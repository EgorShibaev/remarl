"""Small, resumable online metric stream; never uploads prompts or generated text."""

import hashlib
import json
import math
import os
from collections import deque
from copy import deepcopy
from pathlib import Path

SCHEMA_VERSION = 2
EXPERIMENT_LABELS = {
    "mixed-correctness": "Mixed NFE · correctness",
    "shared-correctness": "Shared NFE · correctness",
    "mixed-separated": "Mixed NFE · separated revision",
    "shared-separated": "Shared NFE · separated revision",
}
QUALITY_FIELDS = ("correctness", "reward", "revisions", "repeat_revisions")
UPDATE_FIELDS = {
    "loss", "kl", "l_remask", "denoiser_loss", "lr_used",
    "gradient_norm_before_clip", "replay_logprob_max_error",
}
GRADIENT_FIELDS = {"policy_gradient_norm", "anchor_gradient_norm"}
POST_UPDATE_FIELDS = {
    "post_update_logratio_abs_mean", "post_update_ratio_outside_clip_fraction",
}


def tracking_metadata(config):
    """Keep a study, an experimental condition, and a random seed distinct."""
    tracking = config["tracking"]
    study = tracking.get("study_id", tracking["group"])
    experiment = config["reward_variant"]
    return {
        "study_id": study,
        "experiment_id": experiment,
        "experiment_label": config.get("experiment_label", EXPERIMENT_LABELS.get(experiment, experiment)),
        "seed": config["seed"],
        "logging_schema_version": SCHEMA_VERSION,
        "metric_window_questions": int(tracking.get("window_questions", 128)),
        "metric_every_questions": int(tracking.get("log_every_questions", 32)),
    }


def measured_metrics(metrics):
    """Drop placeholders, not actual measured zeros. Keep raw local history unchanged."""
    result = {}
    for key, value in metrics.items():
        if key in UPDATE_FIELDS and not metrics.get("optimizer_steps_this_iteration", 0):
            continue
        if key in GRADIENT_FIELDS and not metrics.get("gradient_diagnostics_measured", 0):
            continue
        if key in POST_UPDATE_FIELDS and not metrics.get("post_update_diagnostics_measured", 0):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[key] = value
    return result


def window_metrics(events, *, beta, grad_clip):
    """One complete trailing question window, with explicit conditional denominators."""
    output = {"train_window/questions": len(events), "train_window/seed_present": 1}
    for domain in ("all", "gsm8k", "math"):
        subset = [e for e in events if domain == "all" or e["domain"] == domain]
        if not subset:
            continue
        prefix = "train_window/" if domain == "all" else f"train_domain/{domain}/"
        output[prefix + "questions"] = len(subset)
        for key in QUALITY_FIELDS:
            # Equal question weight, even if the number of samples ever differs.
            output[prefix + ("accuracy" if key == "correctness" else key)] = sum(
                sum(r[key] for r in e["rollouts"]) / len(e["rollouts"]) for e in subset
            ) / len(subset)
    output.update(budget_metrics([e["rollouts"] for e in events]))
    output["train_window/rollouts"] = sum(len(e["rollouts"]) for e in events)
    output["train_window/question_has_correct_sample_fraction"] = sum(
        any(r["correctness"] > 0 for r in e["rollouts"]) for e in events
    ) / len(events)
    output["train_window/all_correct_group_fraction"] = sum(
        all(r["correctness"] > 0 for r in e["rollouts"]) for e in events
    ) / len(events)
    output["train_window/skipped_group_fraction"] = sum(
        e["metrics"]["groups_skipped"] for e in events
    ) / len(events)
    accepted = [e["metrics"] for e in events
                if e["metrics"].get("optimizer_steps_this_iteration", 0) > 0]
    output["optimization_window/accepted_questions"] = len(accepted)
    steps = sum(m["optimizer_steps_this_iteration"] for m in accepted)
    output["optimization_window/optimizer_steps"] = steps
    if steps:
        # The trainer reports sums over PPO epochs (batch_size is one here).
        kl = sum(m["kl"] for m in accepted) / steps
        output["optimization_window/kl"] = kl
        output["optimization_window/weighted_kl"] = beta * kl
        output["optimization_window/gradient_norm_before_clip"] = sum(
            m["gradient_norm_before_clip"] for m in accepted
        ) / len(accepted)
        if grad_clip > 0:
            output["optimization_window/clipped_question_fraction"] = sum(
                m["gradient_norm_before_clip"] > grad_clip for m in accepted
            ) / len(accepted)
    for flag, fields, count_name in (
        ("gradient_diagnostics_measured", GRADIENT_FIELDS, "gradient_measurements"),
        ("post_update_diagnostics_measured", POST_UPDATE_FIELDS, "post_update_measurements"),
    ):
        measured = [e["metrics"] for e in events if e["metrics"].get(flag)]
        output[f"diagnostics_window/{count_name}"] = len(measured)
        for key in sorted(fields):
            values = [m[key] for m in measured if key in m]
            if values:
                output[f"diagnostics_window/{key}"] = sum(values) / len(values)
    gradients = [e["metrics"] for e in events
                 if e["metrics"].get("gradient_diagnostics_measured")
                 and e["metrics"].get("policy_gradient_norm", 0) > 0
                 and "anchor_gradient_norm" in e["metrics"]]
    if gradients:
        # Anchor norm already includes beta in the trainer. Do NOT multiply it again.
        output["diagnostics_window/anchor_to_policy_gradient_ratio"] = sum(
            m["anchor_gradient_norm"] / m["policy_gradient_norm"] for m in gradients
        ) / len(gradients)
    output["diagnostics_window/gradient_ratio_measurements"] = len(gradients)
    experts = [e["expert_metrics"] for e in events if e.get("expert_metrics")]
    if experts:
        for key in experts[0]:
            output[f"expert_window/{key}"] = sum(e[key] for e in experts) / len(experts)
        output["expert_window/questions"] = len(experts)
    return output


def budget_metrics(events):
    metrics = {}
    rows = [r for event in events for r in event]
    for steps in (8, 16, 32, 64):
        subset = [r for r in rows if r["requested_nfe"] == steps]
        if not subset:
            continue
        prefix = f"train_nfe{steps}/"
        for key in ("correctness", "reward", "revisions", "repeat_revisions"):
            metrics[prefix + key] = sum(r[key] for r in subset) / len(subset)
        metrics[prefix + "samples"] = len(subset)
        if all("revision_fraction" in r for r in subset):
            metrics[prefix + "revision_fraction"] = sum(r["revision_fraction"] for r in subset)/len(subset)
        for field in ("accuracy_advantage","revision_advantage"):
            if all(field in r for r in subset):
                metrics[prefix + field + "_abs_mean"] = sum(abs(r[field]) for r in subset)/len(subset)
        correct = [r for r in subset if r["correctness"] > 0]
        metrics[prefix + "correct_samples"] = len(correct)
        if correct:
            metrics[prefix + "correct_zero_reward_fraction"] = sum(
                r["reward"] == 0 for r in correct
            ) / len(correct)
            metrics[prefix + "correct_negative_advantage_fraction"] = sum(
                r["advantage"] < 0 for r in correct
            ) / len(correct)
    return metrics


class TrainingTracker:
    def __init__(self, config, control, runtime, *, resume=False, update=0, state=None):
        import wandb

        self.control = Path(control)
        metadata = tracking_metadata(config)
        self.window = metadata["metric_window_questions"]
        self.every = metadata["metric_every_questions"]
        if self.every <= 0 or self.window < self.every or self.window % self.every:
            raise ValueError("Metric window must be a positive multiple of logging cadence")
        if config["grpo"].get("batch_size", 1) != 1:
            raise ValueError("Question-window tracking requires batch_size=1")
        self.beta = float(config["grpo"].get("beta", .01))
        self.grad_clip = float(config["grpo"].get("grad_clip", 1.0))
        self.events = deque(maxlen=self.window)
        self.pending = None
        self.last_question = update
        self.development_points = {}
        if state is not None:
            if (state["schema_version"] != SCHEMA_VERSION or state["window"] != self.window
                    or state["every"] != self.every or state["last_question"] != update):
                raise ValueError("Tracking state does not match the resumed checkpoint")
            self.events.extend(deepcopy(state["events"]))
            self.development_points = deepcopy(state.get("development_points", {}))
        identity = f"{self.control.resolve()}:{metadata['study_id']}:v{SCHEMA_VERSION}"
        self.run_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
        tracking = config["tracking"]
        offline = os.environ.get("REMASK_TRACKING_MODE") == "offline"
        self.run = wandb.init(
            entity=tracking["entity"], project=tracking["project"],
            group=f"{metadata['study_id']}/{metadata['experiment_id']}",
            name=f"{metadata['experiment_id']} · seed {metadata['seed']}", id=self.run_id,
            resume="allow" if resume else "never", mode="offline" if offline else "online", config={**config, **metadata},
            dir=str(runtime), job_type="training", save_code=False,
            tags=["full-diffusion", "fixed-nfe", "reward-ablation", config["reward_variant"]],
            settings=wandb.Settings(init_timeout=45, disable_git=True, console="off"),
        )
        if self.run is None or (self.run.offline and not offline):
            raise RuntimeError("Online W&B was requested but was not initialized")
        self.run.define_metric("progress/attempted_questions")
        self.run.define_metric("*", step_metric="progress/attempted_questions")
        self.run.define_metric("debug/*", hidden=True)
        self.run.summary["status"] = "running"
        self.run.summary["budget_window_questions"] = self.window
        self.run.summary["uploads_generations"] = False
        self.run.summary["window_restored"] = state is not None
        self.run.log({"progress/attempted_questions": update, "system/online_initialized": int(not offline)})
        (self.control / "wandb.json").write_text(json.dumps({
            "id": self.run_id, "url": self.run.url, "path": self.run.path,
        }, indent=2) + "\n")
        print("WANDB_OFFLINE" if offline else "WANDB_ONLINE", self.run_id, flush=True)

    def observe(self, records, *, domain="unknown", expert_records=None):
        if self.pending is not None or not records:
            raise ValueError("Expected one nonempty question before each training log")
        # Retain only numeric diagnostic fields, never the generated text.
        clean = [{k: r[k] for k in (
            "requested_nfe", "correctness", "reward", "advantage", "revisions",
            "repeat_revisions",
        )} for r in records]
        for original,filtered in zip(records,clean):
            for key in ("revision_fraction","accuracy_advantage","revision_advantage"):
                if key in original: filtered[key]=original[key]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v)
               for r in clean for v in r.values()):
            raise ValueError("Tracking records must contain finite numeric fields only")
        self.pending = {"rollouts": clean, "domain": domain if domain in (
            "gsm8k", "math") else "unknown"}
        if expert_records:
            self.pending["expert_metrics"] = {
                "accuracy": sum(r["correctness"] for r in expert_records) / len(expert_records),
                "rescues_all_wrong_policy_fraction": int(
                    not any(r["correctness"] for r in records)
                    and any(r["correctness"] for r in expert_records)),
                "positive_advantage_fraction": sum(r["advantage"] > 0 for r in expert_records)
                                               / len(expert_records),
                "eligible_decisions": sum(r["eligible_decisions"] for r in expert_records)
                                      / len(expert_records),
            }

    def log(self, metrics):
        update = int(metrics["attempted_groups"])
        if self.pending is None or update != self.last_question + 1:
            raise ValueError("Training question metrics must be contiguous and observed first")
        measured = measured_metrics(metrics)
        self.events.append({**self.pending, "metrics": measured})
        self.pending = None
        self.last_question = update
        output = {"progress/attempted_questions": update,
                  "progress/optimizer_updates": metrics["optimizer_updates"]}
        output.update({f"debug/{key}": value for key, value in measured.items()})
        if measured.get("optimizer_steps_this_iteration", 0):
            output["optimization/kl"] = measured["kl"] / measured["optimizer_steps_this_iteration"]
            output["optimization/weighted_kl"] = self.beta * output["optimization/kl"]
        if update % self.every == 0 and len(self.events) == self.window:
            output.update(window_metrics(self.events, beta=self.beta, grad_clip=self.grad_clip))
        self.run.log(output)

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("Checkpoint tracking only at a completed question boundary")
        return deepcopy({"schema_version": SCHEMA_VERSION, "window": self.window,
                         "every": self.every, "last_question": self.last_question,
                         "events": list(self.events),
                         "development_points": self.development_points})

    def log_development(self, rows, update):
        output = {"progress/attempted_questions": update}
        for method in sorted({r["method"] for r in rows}):
            for domain in ("all", "gsm8k", "math"):
                for steps in (8, 16, 32, 64):
                    subset = [r for r in rows if r["method"] == method and r["steps"] == steps
                              and (domain == "all" or r["domain"] == domain)]
                    if not subset:
                        continue
                    prefix = f"dev/{method}/{domain}/nfe{steps}"
                    point = f"{update}/{prefix}"
                    if self.development_points.get(point) == len(subset):
                        continue
                    self.development_points[point] = len(subset)
                    output[prefix + "/accuracy"] = sum(r["correctness"] for r in subset) / len(subset)
                    output[prefix + "/revisions"] = sum(r["revisions"] for r in subset) / len(subset)
                    output[prefix + "/examples"] = len(subset)
        if len(output) > 1:
            self.run.log(output)

    def checkpoint(self, path, update):
        import wandb

        artifact = wandb.Artifact(f"planner-{self.run_id}", type="model",
                                  metadata={"update": update})
        artifact.add_file(str(path), name="resume.pt")
        selector = Path(path).with_name("selector.pt")
        if selector.is_file():
            artifact.add_file(str(selector), name="selector.pt")
        self.run.log_artifact(artifact, aliases=[f"update-{update}", "latest"])

    def finish(self, status):
        self.run.summary["status"] = status
        self.run.finish(exit_code=1 if status == "failed" else 0)
