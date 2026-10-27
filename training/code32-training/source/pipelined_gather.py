"""Opt-in Trainer.gather adapter; never edits an archived production source.

Install only in a separate test/recovery entrypoint. Recipe, rewards, rollout
batch shape, optimizer and checkpoint schema remain unchanged. One coordinator
owns cloned sampler/RNG state and commits it only after a successful collection.
The code adapter's existing begin_collection health probes still run normally.
"""
import copy
import time

import torch

from batch_grpo_core import informative, state_digest
from ordered_pipeline import Stats, collect_ordered


def install(trainer_class, *, max_pending=16, score_workers=1):
    if score_workers != 1:
        raise ValueError("Existing scorer is stateful; only one group-scoring thread is audited")
    if "_ordered_pipeline_installed" in trainer_class.__dict__:
        raise RuntimeError("Collector already installed")
    original_update, original_sync = trainer_class.update, trainer_class.sync

    def require_safe(self):
        if getattr(self, "_pipeline_poisoned", False):
            raise RuntimeError("Collection incomplete/failed; restore in a fresh trainer before updating")

    def update(self, groups):
        require_safe(self)
        return original_update(self, groups)

    def sync(self):
        require_safe(self)
        return original_sync(self)

    def gather(self, max_attempts=128):
        require_safe(self)
        # Import from the trainer's own pinned module, not a second installation.
        import sys
        module = sys.modules[trainer_class.__module__]
        append, save = module.append, module.save
        behavior_hash = self.behavior_hash
        assert state_digest(self.selector.state_dict()) == behavior_hash
        self._pipeline_poisoned = True
        batch_dir = self.out / "batches" / f"{self.completed + 1:06d}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        save(batch_dir / "started.json", dict(behavior_hash=behavior_hash, time=time.time(),
             completed=self.completed, prior_attempted=self.attempted, limit=max_attempts,
             collector="ordered-pipeline-v1", max_pending=max_pending, score_workers=score_workers))
        sampler = copy.deepcopy(self.sampler)
        noise = torch.Generator(device="cpu")
        noise.set_state(self.noise.get_state())
        attempted_ids, logs = set(), []
        stats = Stats()

        def reserve():
            index = sampler.next(attempted_ids)
            attempted_ids.add(index)
            uniforms = torch.rand((self.recipe.nfe, 8, 256), generator=noise)
            return index, uniforms

        def generate(replica, item):
            index, uniforms = item
            group = replica.generate(self.inputs["train"], [index] * 8, uniforms)
            if group["behavior_hash"] != behavior_hash:
                raise RuntimeError("Rollout behavior policy changed during collection")
            return group

        def score(item, group):
            index, _ = item
            return self.scorer(group["tokens"], [self.rows["train"][index]["gold"]] * 8)

        def commit(record):
            index, _ = record.item
            group, answers = record.group, record.answers
            rewards = [a["reward"] for a in answers]
            take = informative(rewards)
            log = dict(attempt=self.attempted + len(logs) + 1, batch=self.completed + 1,
                question_id=index, rewards=rewards, informative=take, behavior_hash=behavior_hash,
                revisions=group["replay"].revisions.tolist(), seconds=group["seconds"],
                nfe_calls=group["nfe_calls"], answer_tokens=group["tokens"], answers=answers)
            if "task" in self.rows["train"][index]:
                log["task"] = self.rows["train"][index]["task"]
            append(batch_dir / "attempts.jsonl", log)
            logs.append(log)
            if take:
                group.update(question_id=index, rewards=rewards)
            return take

        try:
            records, stats = collect_ordered(workers=self.replicas, generation_pool=self.pool,
                reserve=reserve, generate=generate, score=score, commit=commit,
                target=self.batch_groups, max_attempts=max_attempts, max_pending=max_pending,
                score_workers=score_workers, stats=stats)
            assert state_digest(self.selector.state_dict()) == behavior_hash
            assert all(state_digest(r.selector.state_dict()) == behavior_hash for r in self.replicas)
            save(batch_dir / "pipeline.json", dict(complete=True, **stats.to_dict(),
                speculative_groups=0, discarded_groups=0, behavior_hash=behavior_hash))
        except BaseException as exc:
            save(batch_dir / "failed.json", dict(reason=str(exc), type=type(exc).__name__,
                no_updates=True, state_committed=False, **stats.to_dict()))
            raise
        self.sampler.load_state_dict(sampler.state_dict())
        self.noise.set_state(noise.get_state())
        self.attempted += len(logs)
        self._pipeline_poisoned = False
        return [r.group for r in records], logs, batch_dir

    trainer_class.gather, trainer_class.update, trainer_class.sync = gather, update, sync
    trainer_class._ordered_pipeline_installed = True
