"""Fresh small hidden-policy GRPO with native-vLLM frozen LLaDA rollouts.

Four synchronous replica threads collect distinct question-groups. A single
learner updates only after all workers finish; no asynchronous stale rollouts.
All tensors in replay retain raw, detached backbone features on CPU. Evaluation
uses frozen current checkpoints, never contributes gradients or selection.
"""
import argparse
import concurrent.futures
import copy
import json
import os
import select
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from batch_grpo_core import (QuestionSampler, informative, optimize_collection,
                             replay_error, state_digest)
from batch_grpo_recipe import Recipe, SOTA_SHA, initialize_selector
from benchmark_native import cuda_smoke, load_port
from evaluate_p2 import CountedEngine, P2Scores, draws, save, sha
from src.models.set_selector import SetSelector, SetSelectorConfig
from src.rl.training_state import atomic_torch_save
from worker import collect


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class LegacyScorer:
    def __init__(self, root, assets, log):
        local = Path((assets / "local_root.txt").read_text().strip())
        # Reference Python requires its own module-provided libpython path.
        command = ["bash", "-lc", 'module --force purge; module load release/24.04 '
                   'GCCcore/11.3.0 Python/3.10.4 CUDA/12.6.0; exec "$@"', "scorer",
                   str(local / "venv/bin/python"), str(root / "batch_grpo_reference.py"),
                   "serve", "--assets", str(assets)]
        self.log = log.open("a")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        self.index = 0
        assert self.receive() == {"ready": True}

    def receive(self):
        if not select.select([self.process.stdout], [], [], 300)[0]:
            raise TimeoutError("Pinned scorer did not reply within 300 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Pinned reward scorer terminated")
        return json.loads(line)

    def __call__(self, tokens, golds):
        self.index += 1
        self.process.stdin.write(json.dumps(dict(request_id=self.index, tokens=tokens, golds=golds)) + "\n")
        self.process.stdin.flush()
        response = self.receive()
        assert response["request_id"] == self.index and len(response["answers"]) == len(golds)
        return response["answers"]

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=15)
        self.log.close()


class Replica:
    def __init__(self, gpu, port, checkpoint, selector_config):
        self.device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(self.device)
        self.engine = port.LLaDAOfflineModel.from_local_checkpoint(
            checkpoint, device=self.device, backend="vllm", fuse_projections=True)
        assert sum(p.numel() for p in self.engine.parameters()) == 8015581184
        assert not any(p.requires_grad for p in self.engine.parameters())
        self.selector = SetSelector(selector_config).to(self.device).eval()
        self.behavior_hash = None

    def synchronize(self, state, expected_hash):
        self.selector.load_state_dict(state)
        self.behavior_hash = state_digest(self.selector.state_dict())
        assert self.behavior_hash == expected_hash

    def generate(self, inputs, indices, uniforms, record=True, baseline=False):
        torch.cuda.set_device(self.device)
        started = time.monotonic()
        ids, mask, positions = (inputs[k][indices].to(self.device) for k in ("ids", "mask", "positions"))
        # Remove right padding beyond this minibatch's longest full prompt+answer.
        # No prompt tokens are truncated; answer spans remain absolute and unchanged.
        extent = int(mask.sum(-1).max().item())
        ids, mask = ids[:, :extent], mask[:, :extent]
        counted = CountedEngine(self.engine)
        result = collect(SimpleNamespace(mask_token_id=126336),
                         P2Scores() if baseline else self.selector,
                         ids, mask, positions, uniforms, answer_backbone=counted, record=record)
        assert counted.calls == uniforms.shape[0]
        answer = result.final_ids.gather(1, positions).cpu()
        assert not answer.eq(126336).any()
        assert torch.equal(result.final_ids[~torch.zeros_like(ids, dtype=torch.bool).scatter(1, positions, True)],
                           ids[~torch.zeros_like(ids, dtype=torch.bool).scatter(1, positions, True)])
        result.final_ids = result.final_ids.cpu()
        result.revisions = result.revisions.cpu()
        result.repeat_revisions = result.repeat_revisions.cpu()
        return dict(replay=result, tokens=answer.tolist(), seconds=time.monotonic() - started,
                    behavior_hash=self.behavior_hash, nfe_calls=counted.calls)


class Trainer:
    def __init__(self, root, out, assets, gpus):
        self.root, self.out, self.assets = root, out, assets
        recipe_path = root / "recipe.json"
        self.recipe = Recipe(**json.loads(recipe_path.read_text())) if recipe_path.exists() else Recipe()
        self.total_batches = self.recipe.total_batches
        self.batch_groups = self.recipe.accepted_groups_per_batch
        out.mkdir(parents=True, exist_ok=True)
        self.audit = json.loads((root / "data-audit.json").read_text())
        assert sha(root / "inputs.pt") == self.audit["inputs_sha256"]
        assert sha(root / "rows.json") == self.audit["rows_sha256"]
        self.inputs = torch.load(root / "inputs.pt", map_location="cpu", weights_only=True)
        self.rows = json.loads((root / "rows.json").read_text())
        assert len(self.rows["train"]) == self.audit["train"] == self.recipe.training_questions
        saved = torch.load(assets / "sota-selector.pt", map_location="cpu", weights_only=True)
        assert sha(assets / "sota-selector.pt") == SOTA_SHA
        self.config = self.recipe.selector_config(SetSelectorConfig(**saved["set_selector_config"]))
        torch.manual_seed(7)
        self.device = torch.device(f"cuda:{gpus[0]}")
        self.selector = SetSelector(self.config).to(self.device).eval()
        self.initial_policy_hash = initialize_selector(self.selector, saved, self.recipe.initialization)
        parameter_count = sum(p.numel() for p in self.selector.parameters())
        if self.recipe.policy_width is None:
            assert parameter_count == 284115
        else:
            layers = [m for m in self.selector.pointwise if isinstance(m, torch.nn.Linear)]
            assert [(m.in_features, m.out_features) for m in layers] == [(82,256),(256,256),(256,1)]
        self.optimizer = torch.optim.AdamW(self.selector.parameters(), lr=self.recipe.learning_rate, weight_decay=0.)
        assert len(self.optimizer.state) == 0  # Intentional fresh optimizer for changed batch/data.
        self.sampler = QuestionSampler(self.recipe.training_questions, 7)
        save(out / "resolved-recipe.json", dict(**self.recipe.to_dict(),
             selector_parameters=parameter_count, selector_config=vars(self.config), parent_checkpoint_sha256=SOTA_SHA,
             initial_policy_sha256=self.initial_policy_hash, fresh_optimizer=True,
             trajectories_per_batch=self.batch_groups * 8,
             trajectories_per_update=self.recipe.groups_per_minibatch * 8))
        self.noise = torch.Generator().manual_seed(7001)
        self.shuffle = torch.Generator().manual_seed(7002)
        self.completed, self.attempted = 0, 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus))
        from paper_protocol import Scorer
        self.scorer = Scorer(root, assets, out / "scorer.log")
        local = Path((assets / "local_root.txt").read_text().strip())
        port = load_port(root)
        assert sha(root / "model.py") == "73cac7db80542a3c25e1448332175864b5a9e53cd7c943fe76e107b736b3c0bd"
        torch.cuda.set_device(self.device)
        save(out / "cuda-smoke.json", cuda_smoke(port))
        self.replicas = []
        for gpu in gpus:
            self.replicas.append(Replica(gpu, port, local / "models/LLaDA-8B-Instruct", self.config))
            print("REPLICA_READY", gpu, flush=True)
        torch.cuda.set_device(self.device)
        self.sync()

    def sync(self):
        state = cpu_state(self.selector)
        self.behavior_hash = state_digest(state)
        for replica in self.replicas:
            replica.synchronize(state, self.behavior_hash)

    def snapshot(self):
        return dict(model=cpu_state(self.selector), optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    sampler=self.sampler.state_dict(), noise=self.noise.get_state(),
                    shuffle=self.shuffle.get_state(), completed=self.completed, attempted=self.attempted,
                    total_batches=self.total_batches, selector_config=vars(self.config),
                    source_manifest_sha256=sha(self.root / "source.sha256"),
                    data_audit_sha256=sha(self.root / "data-audit.json"),
                    policy_sha256=state_digest(self.selector.state_dict()),
                    backend="native-vllm-fa2", group_size=8, accepted_groups_per_batch=self.batch_groups,
                    groups_per_minibatch=self.recipe.groups_per_minibatch, passes=self.recipe.passes,
                    optimizer_updates=self.completed * self.recipe.updates_per_batch,
                    recipe=self.recipe.to_dict(), parent_checkpoint_sha256=SOTA_SHA,
                    initial_policy_sha256=self.initial_policy_hash)

    def restore(self, state):
        assert state["total_batches"] == self.total_batches
        assert state["recipe"] == self.recipe.to_dict()
        assert state["initial_policy_sha256"] == self.initial_policy_hash
        assert state["source_manifest_sha256"] == sha(self.root / "source.sha256")
        assert state["data_audit_sha256"] == sha(self.root / "data-audit.json")
        self.selector.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler.load_state_dict(state["sampler"])
        self.noise.set_state(state["noise"])
        self.shuffle.set_state(state["shuffle"])
        self.completed, self.attempted = state["completed"], state["attempted"]
        self.sync()

    def gather(self, max_attempts=1024):
        accepted, attempted_ids, all_rewards, logs = [], set(), [], []
        batch_dir = self.out / "batches" / f"{self.completed + 1:06d}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        save(batch_dir / "started.json", dict(behavior_hash=self.behavior_hash, time=time.time(),
             completed=self.completed, prior_attempted=self.attempted, limit=max_attempts))
        while len(accepted) < self.batch_groups:
            if len(attempted_ids) >= max_attempts:
                save(batch_dir / "failed.json", dict(reason="informative-group retry limit",
                     accepted=len(accepted), attempted=len(attempted_ids), no_updates=True))
                raise RuntimeError("Could not fill informative batch; no partial batch update allowed")
            count = min(len(self.replicas), self.batch_groups - len(accepted), max_attempts - len(attempted_ids))
            futures = []
            for replica in self.replicas[:count]:
                index = self.sampler.next(attempted_ids)
                attempted_ids.add(index)
                uniforms = torch.rand((self.recipe.nfe, 8, 256), generator=self.noise)
                future = self.pool.submit(replica.generate, self.inputs["train"], [index] * 8, uniforms)
                futures.append((index, future))
            # Consume in dispatch order: thread scheduling cannot change RNG/sampler order.
            for index, future in futures:
                group = future.result(timeout=300)
                assert group["behavior_hash"] == self.behavior_hash
                answers = self.scorer(group["tokens"], [self.rows["train"][index]["gold"]] * 8)
                rewards = [a["reward"] for a in answers]
                take = informative(rewards)
                self.attempted += 1
                log = dict(attempt=self.attempted, batch=self.completed + 1, question_id=index,
                           task=self.rows["train"][index]["task"],
                           rewards=rewards, informative=take, behavior_hash=self.behavior_hash,
                           revisions=group["replay"].revisions.tolist(),
                           seconds=group["seconds"], nfe_calls=group["nfe_calls"],
                           answer_tokens=group["tokens"], answers=answers)
                append(batch_dir / "attempts.jsonl", log)
                all_rewards.extend(rewards)
                logs.append(log)
                if take:
                    group.update(question_id=index, rewards=rewards)
                    accepted.append(group)
        assert state_digest(self.selector.state_dict()) == self.behavior_hash
        assert all(state_digest(r.selector.state_dict()) == self.behavior_hash for r in self.replicas)
        return accepted, logs, batch_dir

    def update(self, groups):
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        error = replay_error(self.selector, groups, self.device)
        if error > .003:
            raise RuntimeError(f"Behavior replay mismatch before first update: {error}")
        original = [g["replay"].old_logp.clone() for g in groups]
        lr = self.recipe.learning_rate_at(self.completed)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        updates = optimize_collection(self.selector, self.optimizer, groups, self.device,
                                      self.shuffle, self.batch_groups, self.recipe.passes,
                                      self.recipe.groups_per_minibatch)
        assert all(torch.equal(old, g["replay"].old_logp) for old, g in zip(original, groups))
        used_ids = [qid for u in updates for qid in u["question_ids"]]
        assert len(set(used_ids)) == self.batch_groups
        assert all(used_ids.count(qid) == self.recipe.passes for qid in set(used_ids))
        assert len(updates) == self.recipe.updates_per_batch
        assert all(u["trajectories"] == 8 * self.recipe.groups_per_minibatch for u in updates)
        return dict(lr=lr, replay_error=error, updates=updates)

    def evaluate(self, split, seed):
        name = f"{split}-{self.completed * self.batch_groups:06d}-seed{seed}"
        folder = self.out / "evaluations" / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "summary.json").exists():
            result = json.loads((folder / "summary.json").read_text())
            assert result["policy_sha256"] == self.behavior_hash
            return
        assert not (folder / "answers.jsonl").exists(), "Incomplete eval: inspect before retrying"
        results = []
        total = len(self.rows[split])
        started = time.monotonic()
        for start in range(0, total, len(self.replicas) * 8):
            futures = []
            for j, replica in enumerate(self.replicas):
                indices = list(range(start + j * 8, min(start + (j + 1) * 8, total)))
                if not indices:
                    break
                real = len(indices)
                indices += [indices[-1]] * (8 - real)
                # Full-test uses exact audited per-question random streams.
                uniforms = torch.stack([draws(i, self.recipe.nfe, seed) for i in indices], 1)
                futures.append((indices, real, self.pool.submit(replica.generate, self.inputs[split],
                                                               indices, uniforms, False)))
            for indices, real, future in futures:
                group = future.result(timeout=300)
                answers = self.scorer(group["tokens"][:real],
                                      [self.rows[split][i]["gold"] for i in indices[:real]])
                for i, tokens, answer in zip(indices[:real], group["tokens"][:real], answers):
                    record = dict(index=i, tokens=tokens, **answer)
                    results.append(record)
                    append(folder / "answers.jsonl", record)
        assert [r["index"] for r in results] == list(range(total))
        correct = int(sum(r["reward"] for r in results))
        result = dict(complete=True, split=split, count=total, correct=correct, accuracy=correct / total,
                      accepted_groups=self.completed * self.batch_groups, attempted_groups=self.attempted,
                      optimizer_updates=self.completed * self.recipe.updates_per_batch,
                      held_out=split.startswith("test_"),
                      seed=seed, nfe=self.recipe.nfe, seconds=time.monotonic() - started,
                      policy_sha256=self.behavior_hash, backend="native-vllm-fa2",
                      prompt="APPLe task-specific evaluation template", scorer="APPLe official task-specific evaluator", position_temperature=1.,
                      token_temperature=0., answer_length=256, generation_batch=8)
        save(folder / "summary.json", result)
        print("EVALUATED", name, correct, total, flush=True)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.scorer.close()


def validate(trainer):
    """Real native-backend repeatability, P2 init, replay, gradients, resume gates."""
    initial = trainer.snapshot()
    replica = trainer.replicas[0]
    uniforms = torch.rand((trainer.recipe.nfe, 8, 256), generator=torch.Generator().manual_seed(777))
    first = replica.generate(trainer.inputs["train"], [0] * 8, uniforms)
    repeat = replica.generate(trainer.inputs["train"], [0] * 8, uniforms)
    assert first["tokens"] == repeat["tokens"]
    if trainer.recipe.initialization == "fresh":
        baseline = replica.generate(trainer.inputs["train"], [0] * 8, uniforms, False, True)
        assert first["tokens"] == baseline["tokens"]
        del baseline
    else:
        saved = torch.load(trainer.assets / "sota-selector.pt", map_location="cpu", weights_only=True)
        assert initial["policy_sha256"] == state_digest(saved["model_state"])
    assert torch.equal(first["replay"].old_logp, repeat["replay"].old_logp)
    assert all(torch.equal(a, b) for a, b in zip(first["replay"].orders, repeat["replay"].orders))
    del first, repeat
    longest = max(range(len(trainer.rows["train"])),
                  key=lambda i: len(trainer.rows["train"][i]["prompt_ids"]))
    long_group = replica.generate(trainer.inputs["train"], [longest]*8, uniforms, record=False)
    long_answers = trainer.scorer(long_group["tokens"], [trainer.rows["train"][longest]["gold"]]*8)
    save(trainer.out / "long-prompt-gate.json", dict(index=longest,
         prompt_tokens=len(trainer.rows["train"][longest]["prompt_ids"]),
         nfe=long_group["nfe_calls"], answers=long_answers, prompt_unchanged=True))
    del long_group, long_answers
    groups, logs, _folder = trainer.gather()
    before_update = trainer.snapshot()
    before = cpu_state(trainer.selector)
    one = trainer.update(groups)
    state_one = trainer.snapshot()
    assert state_one["policy_sha256"] != initial["policy_sha256"]
    hidden_changed = [k for k in before if "hidden_projection" in k
                      and not torch.equal(before[k], state_one["model"][k])]
    assert hidden_changed, "Trainable hidden projection did not receive an update"
    atomic_torch_save(before_update, trainer.out / "resume-gate.pt")
    trainer.restore(torch.load(trainer.out / "resume-gate.pt", map_location="cpu", weights_only=True))
    two = trainer.update(groups)
    state_two = trainer.snapshot()
    differences = {k: float((state_one["model"][k] - state_two["model"][k]).abs().max())
                   for k in state_one["model"]}
    save(trainer.out / "resume-comparison.json", dict(first=one, second=two,
         parameter_max_abs=differences, expected_hash=state_one["policy_sha256"],
         actual_hash=state_two["policy_sha256"]))
    atomic_torch_save(state_one, trainer.out / "resume-first.pt")
    atomic_torch_save(state_two, trainer.out / "resume-second.pt")
    assert state_one["policy_sha256"] == state_two["policy_sha256"], "Disk-resume update mismatch"
    for (p1, value1), (p2, value2) in zip(state_one["optimizer"]["state"].items(),
                                       state_two["optimizer"]["state"].items()):
        assert p1 == p2
        assert all(torch.equal(value1[k], value2[k]) for k in value1)
    assert one == two
    def exact(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
        return a == b
    assert exact(state_one, state_two), "Full model/optimizer/sampler/RNG resume mismatch"
    save(trainer.out / "VALIDATED.json", dict(passed=True, native_repeatability="exact",
         initial_policy_matches_native_P2=(trainer.recipe.initialization == "fresh"),
         initial_policy_matches_SOTA=(trainer.recipe.initialization == "sota"),
         collected_informative_groups=trainer.batch_groups,
         attempted_groups=len(logs), trajectories=trainer.batch_groups * 8,
         optimizer_minibatch=8 * trainer.recipe.groups_per_minibatch,
         optimizer_steps=trainer.recipe.updates_per_batch, passes=trainer.recipe.passes, replay=one, disk_resume_optimizer="exact",
         full_resume_state="exact",
         hidden_projection_updated=hidden_changed, initialization_sha256=initial["policy_sha256"],
         source_manifest_sha256=sha(trainer.root / "source.sha256"),
         data_audit_sha256=sha(trainer.root / "data-audit.json")))
    print("VALIDATED", flush=True)


def production(trainer, resume=False):
    gate = json.loads((trainer.root / "validation/VALIDATED.json").read_text())
    assert gate["passed"] and gate["source_manifest_sha256"] == sha(trainer.root / "source.sha256")
    assert gate["data_audit_sha256"] == sha(trainer.root / "data-audit.json")
    if resume:
        trainer.restore(torch.load(trainer.out / "latest.pt", map_location="cpu", weights_only=True))
    else:
        assert not (trainer.out / "latest.pt").exists()
        assert trainer.behavior_hash == gate["initialization_sha256"]
        atomic_torch_save(trainer.snapshot(), trainer.out / "latest.pt")
    for seed in (101, 202, 303):
        if trainer.completed == 0:
            for split in ("test_gsm8k", "test_math"):
                trainer.evaluate(split, seed)
    while trainer.completed < trainer.total_batches:
        started = time.monotonic()
        groups, logs, folder = trainer.gather()
        diagnostics = trainer.update(groups)
        accepted_accuracy = sum(sum(g["rewards"]) for g in groups) / (trainer.batch_groups * 8)
        rewards = [r for log in logs for r in log["rewards"]]
        trainer.completed += 1
        snapshot = trainer.snapshot()
        atomic_torch_save(snapshot, trainer.out / "latest.pt")
        summary = dict(batch=trainer.completed, accepted_groups=trainer.batch_groups * trainer.completed,
                       optimizer_updates=trainer.recipe.updates_per_batch * trainer.completed,
                       attempted_groups=trainer.attempted, attempted_this_batch=len(logs),
                       all_attempt_accuracy=sum(rewards) / len(rewards),
                       accepted_accuracy=accepted_accuracy, acceptance_rate=trainer.batch_groups / len(logs),
                       all_wrong=sum(sum(l["rewards"]) == 0 for l in logs),
                       all_correct=sum(sum(l["rewards"]) == 8 for l in logs),
                       mean_revisions=sum(sum(l["revisions"]) for l in logs) / len(rewards),
                       task_metrics={task: dict(
                           attempted=sum(l["task"] == task for l in logs),
                           accepted=sum(l["task"] == task and l["informative"] for l in logs),
                           correct=sum(sum(l["rewards"]) for l in logs if l["task"] == task),
                           trajectories=8*sum(l["task"] == task for l in logs))
                           for task in ("gsm8k", "math")},
                       seconds=time.monotonic() - started, time=time.time(),
                       behavior_hash=trainer.behavior_hash, new_policy_hash=snapshot["policy_sha256"],
                       **diagnostics)
        save(folder / "complete.json", summary)
        append(trainer.out / "metrics.jsonl", summary)
        trainer.sync()
        print("BATCH", trainer.completed, "accepted", trainer.completed * trainer.batch_groups, "attempted", trainer.attempted,
              "accuracy", round(summary["all_attempt_accuracy"], 4), "seconds", round(summary["seconds"], 2), flush=True)
        del groups, logs, snapshot
        accepted_total = trainer.completed * trainer.batch_groups
        if accepted_total % 512 == 0:
            atomic_torch_save(trainer.snapshot(), trainer.out / f"selector-{accepted_total:06d}.pt")
            trainer.evaluate(trainer.recipe.probe_split, 101)
        if accepted_total in (4096, 8192, 16384, 32768, 49152) or trainer.completed == trainer.total_batches:
            for seed in (101, 202, 303):
                for split in ("test_gsm8k", "test_math"):
                    trainer.evaluate(split, seed)
    save(trainer.out / "COMPLETE.json", dict(completed=time.time(), accepted_groups=trainer.completed * trainer.batch_groups,
         attempted_groups=trainer.attempted, optimizer_updates=trainer.completed * trainer.recipe.updates_per_batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    folder = args.root / ("validation" if args.mode == "validate" else "production")
    if not args.resume:
        assert not folder.exists(), "Existing workflow output: inspect, do not overwrite"
    trainer = Trainer(args.root, folder, args.assets, [int(x) for x in args.gpus.split(",")])
    try:
        validate(trainer) if args.mode == "validate" else production(trainer, args.resume)
    except BaseException as exc:
        save(folder / f"failure-{int(time.time())}.json", dict(type=type(exc).__name__, message=str(exc), time=time.time()))
        raise
    finally:
        trainer.close()
