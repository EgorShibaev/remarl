"""Train fresh ReMaRL policies with accepted-group GRPO."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from .data.prepare import load_prepared, subset
from .io import append_json, save_json, save_checkpoint
from .optimization import (
    QuestionSampler,
    informative,
    optimize_collection,
    replay_error,
    state_digest,
)
from .policy import PolicyConfig, policy_class_for
from .protocols.scoring import Scorer
from .runtime import Generator, configure_compute, backbone_spec, check_prepared_backbone


def policy_config_for_training(cfg):
    hidden = cfg.get("backbone", {}).get("hidden_size", 4096)
    architecture = cfg.get("policy_architecture", "mlp")
    if architecture not in ("mlp", "remarl_jazbec_dit"):
        raise ValueError(f"Unknown policy architecture: {architecture}")
    if architecture != "mlp" and (hidden != 4096 or cfg.get("variant", "full") != "full"):
        raise ValueError("The transformer ablation uses full LLaDA features")
    initialization = cfg.get("policy_initialization", "p2_prime")
    if initialization == "p2_prime":
        return PolicyConfig(backbone_hidden_dim=hidden, architecture=architecture)
    if initialization == "random_mlp_no_confidence_residual":
        if cfg.get("variant", "full") != "full" or architecture != "mlp":
            raise ValueError("The initialization ablation uses the full policy features")
        return PolicyConfig(backbone_hidden_dim=hidden, confidence_residual=False, zero_initialize_output=False)
    raise ValueError(f"Unknown policy initialization: {initialization}")


def validate_config(cfg):
    if (
        cfg["nfe"] not in (4, 8, 16, 32, 64)
        or cfg["group_size"] != 8
        or cfg["accepted_groups_per_batch"] != 16
        or cfg["groups_per_minibatch"] != 16
        or cfg["passes"] != 4
        or cfg["lr_schedule"] != "constant"
        or cfg["target_accepted_groups"] <= 0
        or cfg["target_accepted_groups"] % 16
    ):
        raise ValueError(
            "Expected the paper recipe: groups of 8, batches of 16, four full-batch updates"
        )
    if cfg["initialization"] != "fresh" or cfg["answer_length"] != 256:
        raise ValueError("Training starts from a fresh policy with 256 answer positions")
    policy_config_for_training(cfg)


class Trainer:
    """Dependency-injected generator/scorer support CPU lifecycle checks."""

    def __init__(self, config, rows, manifest, output, replicas, scorer, policy, device):
        validate_config(config)
        self.config, self.rows, self.manifest = config, rows, manifest
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.replicas, self.scorer, self.policy = replicas, scorer, policy
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=config["learning_rate"], weight_decay=0.0
        )
        seed = config["seed"]
        self.sampler = QuestionSampler(len(rows), seed)
        self.noise = torch.Generator().manual_seed(seed * 1000 + 1)
        self.shuffle = torch.Generator().manual_seed(seed * 1000 + 2)
        self.completed = self.attempted = 0
        self.pool = ThreadPoolExecutor(max_workers=len(replicas))
        self.sync()

    def sync(self):
        state = {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}
        for replica in self.replicas:
            replica.policy.load_state_dict(state)
        self.policy_hash = state_digest(state)

    def snapshot(self):
        return dict(
            model={k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()},
            selector_config=asdict(self.policy.config),
            optimizer=self.optimizer.state_dict(),
            sampler=self.sampler.state_dict(),
            noise=self.noise.get_state(),
            shuffle=self.shuffle.get_state(),
            completed=self.completed,
            attempted=self.attempted,
            accepted_groups=self.completed * 16,
            policy_sha256=self.policy_hash,
            config=self.config,
            data_manifest=self.manifest,
        )

    def restore(self, saved):
        previous_config = dict(saved["config"])
        # Older LLaDA checkpoints predate explicit backbone metadata. The model
        # is still checked when train() constructs its generator and dataset.
        if "backbone" not in previous_config and self.config.get("backbone", {}).get("kind") == "llada":
            previous_config["backbone"] = self.config["backbone"]
        if previous_config != self.config or saved["data_manifest"] != self.manifest:
            raise ValueError("Resume requires the same configuration and prepared data")
        if PolicyConfig.from_dict(saved["selector_config"]) != self.policy.config:
            raise ValueError("Resume requires the same policy architecture and scoring rule")
        if state_digest(saved["model"]) != saved["policy_sha256"]:
            raise ValueError("Resume policy hash mismatch")
        self.policy.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.sampler.load_state_dict(saved["sampler"])
        self.noise.set_state(saved["noise"])
        self.shuffle.set_state(saved["shuffle"])
        self.completed, self.attempted = saved["completed"], saved["attempted"]
        self.sync()

    def gather(self):
        self.scorer.begin_collection()
        accepted, logs, attempted_ids = [], [], set()
        # Dispatch and accept in sample order. Rejected groups consume their RNG draws.
        while len(accepted) < 16:
            if len(logs) >= 1024:
                raise RuntimeError("Could not collect 16 informative groups in 1024 attempts")
            work = []
            for replica in self.replicas[: min(16 - len(accepted), 1024 - len(logs))]:
                if len(attempted_ids) == len(self.rows) and self.config.get("per_dataset"):
                    attempted_ids.clear()
                index = self.sampler.next(attempted_ids)
                attempted_ids.add(index)
                uniforms = torch.rand((self.config["nfe"], 8, 256), generator=self.noise)
                future = self.pool.submit(replica.generate, [self.rows[index]] * 8, uniforms, True)
                work.append((index, future))
            for index, future in work:
                result = future.result()
                answers = self.scorer(result["tokens"], [self.rows[index]["gold"]] * 8)
                rewards = [a["reward"] for a in answers]
                take = informative(rewards)
                self.attempted += 1
                log = dict(
                    attempt=self.attempted,
                    question_id=index,
                    accepted=take,
                    rewards=rewards,
                    revisions=result["revisions"],
                )
                logs.append(log)
                append_json(self.output / "attempts.jsonl", log)
                if take:
                    identity = self.attempted if self.config.get("per_dataset") else index
                    accepted.append(
                        dict(
                            question_id=identity,
                            source_question_id=index,
                            rewards=rewards,
                            replay=result["replay"],
                        )
                    )
        return accepted, logs

    def step(self):
        started = time.monotonic()
        groups, attempts = self.gather()
        error = replay_error(self.policy, groups, self.device)
        if error > 0.003:
            raise RuntimeError(f"Behavior-policy replay mismatch: {error}")
        updates = optimize_collection(
            self.policy,
            self.optimizer,
            groups,
            self.device,
            self.shuffle,
            expected_groups=16,
            passes=4,
            groups_per_minibatch=16,
        )
        self.completed += 1
        self.sync()
        record = dict(
            accepted_groups=self.completed * 16,
            attempted_groups=self.attempted,
            optimizer_updates=self.completed * 4,
            policy_sha256=self.policy_hash,
            replay_error=error,
            updates=updates,
            seconds=time.monotonic() - started,
            accuracy=sum(sum(a["rewards"]) for a in attempts) / (8 * len(attempts)),
        )
        save_checkpoint(self.output / "latest.pt", self.snapshot())
        append_json(self.output / "training.jsonl", record)
        print(json.dumps({k: v for k, v in record.items() if k != "updates"}), flush=True)
        return record

    def run(self):
        target = self.config["target_accepted_groups"]
        while self.completed * 16 < target:
            self.step()
            groups = self.completed * 16
            if groups % self.config["save_every_groups"] == 0 or groups == target:
                save_checkpoint(self.output / f"policy-{groups:06d}.pt", self.snapshot())

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.scorer.close()


def train(config, data, model, output, devices, backend="vllm", resume=None):
    configure_compute()
    cfg = json.loads(Path(config).read_text())
    cfg["backend"] = backend
    spec = backbone_spec(model)
    if cfg.get("backbone", spec) != spec:
        raise ValueError("Training recipe specifies a different frozen model")
    cfg["backbone"] = spec
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Provide one or more distinct GPU devices")
    validate_config(cfg)
    rows, manifest = load_prepared(data)
    check_prepared_backbone(manifest, spec)
    if cfg["domain"] != manifest["domain"]:
        raise ValueError("Configuration and dataset domains differ")
    if cfg["domain"] == "code" and cfg["cohort"] != manifest["cohort"]:
        raise ValueError("Use the cohort specified by this checkpoint's recipe")
    training = rows["train"]
    if cfg.get("per_dataset"):
        training, indices = subset(training, cfg["seed"], cfg["per_dataset"])
        manifest = dict(manifest, subset_indices=indices)
    if len(training) != cfg["training_questions"]:
        raise ValueError("Training question count differs from the configuration")
    if (Path(output) / "latest.pt").exists() and resume is None:
        raise FileExistsError("Output contains a run; use --resume or a new directory")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    from .ablations import selector_class

    variant = cfg.get("variant", "full")
    policy_config = policy_config_for_training(cfg)
    base_class = policy_class_for(policy_config)
    policy_class = base_class if variant == "full" else selector_class(base_class, variant)
    policy = policy_class(policy_config).to(devices[0]).eval()
    replicas = [
        Generator(model, policy_class(policy_config), d, backend, manifest["required_canvas"])
        for d in devices
    ]
    scorer = Scorer(replicas[0].tokenizer, cfg["domain"])
    trainer = Trainer(cfg, training, manifest, output, replicas, scorer, policy, devices[0])
    try:
        if resume:
            trainer.restore(torch.load(resume, map_location="cpu", weights_only=True))
        else:
            save_checkpoint(Path(output) / "initial.pt", trainer.snapshot())
        save_json(Path(output) / "config.json", cfg)
        trainer.run()
    finally:
        trainer.close()
