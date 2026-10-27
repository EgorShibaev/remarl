"""Plot completed data-size endpoints; never mix partial training seeds."""
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
status_bytes = (BASE / "STATUS.json").read_bytes()
staged_bytes = (BASE / "STAGED.json").read_bytes()
controls_path = BASE.parents[2] / "output/pdf/paper-plan-assets/figures/data-receipt.json"
controls_bytes = controls_path.read_bytes()
controls = json.loads(controls_bytes)["temperature_controls"]
control_temperature = .25
status, staged = json.loads(status_bytes), json.loads(staged_bytes)
sizes = staged["per_dataset"]
target = staged["target_groups"]
seeds = [seed for seed in staged["training_seeds"] if all(
    "COMPLETE.json" in status["runs"].get(f"n{2 * size}-seed{seed}", {})
    for size in sizes)]
assert seeds, "No complete paired training-seed comparison yet."


def score(tests, split):
    rows = [t for t in tests if t["split"] == split and t["accepted_groups"] == target]
    assert sorted(t["seed"] for t in rows) == sorted(staged["inference_seeds"])
    assert all(t["complete"] and t["count"] == staged["tests"][split]
               and t["nfe"] == 8 for t in rows)
    assert len({t["policy_sha256"] for t in rows}) == 1
    return 100 * sum(t["correct"] for t in rows) / sum(t["count"] for t in rows)


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "0.6", "axes.linewidth": .55,
    "xtick.color": "black", "ytick.color": "black",
})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
receipt = {}
for ax, split, title, limits in zip(
        axes, ("test_gsm8k", "test_math"), ("GSM8K", "MATH500"),
        ((35, 57), (15, 22))):
    per_seed = [[score(status["runs"][f"n{2 * size}-seed{seed}"]["tests"], split)
                 for size in sizes] for seed in seeds]
    ys = [mean(values) for values in zip(*per_seed)]
    reference = score(staged["reference_evaluations"].values(), split)
    row = controls[split]
    tuned_p2 = row["scores"][row["temperatures"].index(control_temperature)]
    seed_styles = (("#157CA8", (0, (5, 3)), "s"),
                   ("#157CA8", (0, (2, 2)), "^"),
                   ("#157CA8", (0, (5, 2, 1, 2)), "v"))
    for number, (values, (color, dash, marker)) in enumerate(
            zip(per_seed, seed_styles), 1):
        ax.plot(sizes, values, color=color, linewidth=.8, linestyle=dash,
                marker=marker, markersize=3.6, markerfacecolor="none",
                markeredgewidth=.65, alpha=.5, label=f"Training seed {number}")
    ax.plot(sizes, ys, color="#157CA8", linewidth=1.0, marker="o", markersize=4,
            label="Mean", zorder=4)
    ax.axhline(reference, color="0.4", linewidth=.85, linestyle=(0, (4, 3)),
               label="Full-data reference")
    ax.axhline(tuned_p2, color="#8E4CB6", linewidth=.85,
               linestyle=(0, (7, 3, 1.5, 3)), label="Tuned P2′ (τ = 0.25)")
    ax.text(.98, (reference - limits[0]) / (limits[1] - limits[0]) + .025,
            f"Full data: {reference:.2f}", transform=ax.transAxes,
            ha="right", fontsize=10, color="0.3")
    ax.text(.98, (tuned_p2 - limits[0]) / (limits[1] - limits[0]) - .055,
            f"Tuned P2′: {tuned_p2:.2f}", transform=ax.transAxes,
            ha="right", fontsize=10, color="#8E4CB6")
    ax.set_xscale("log", base=8)
    ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
    ax.set_xlim(2.7, 3000)
    ax.set_ylim(*limits)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Training questions per dataset")
    ax.set_ylabel("Full-test accuracy (%)")
    ax.grid(axis="y", color=".9", linewidth=.5)
    ax.set_axisbelow(True)
    receipt[split] = dict(sizes=sizes, means=ys, per_training_seed=per_seed,
                          full_data_reference=reference, tuned_p2=tuned_p2,
                          control_temperature=control_temperature)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
           frameon=False, bbox_to_anchor=(.5, .99))
seed_note = ("First of three paired subset/training seeds" if len(seeds) == 1
             else f"{len(seeds)} of three paired subset/training seeds")
fig.text(.5, .025, "NFE8 · 16k training groups · " + seed_note +
         "\nThree inference seeds per point; full-data reference reuses the existing 16k run."
         "\nP2′ temperature was selected on training questions and is fixed across dataset sizes.",
         ha="center", fontsize=10)
fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.25, wspace=.24)
fig.savefig(BASE / "data-size-endpoints.png", dpi=180)
plt.close(fig)
(BASE / "endpoint-plot-data.json").write_text(json.dumps(dict(
    status_sha256=hashlib.sha256(status_bytes).hexdigest(),
    staged_sha256=hashlib.sha256(staged_bytes).hexdigest(),
    controls_source=str(controls_path.relative_to(BASE.parents[2])),
    controls_sha256=hashlib.sha256(controls_bytes).hexdigest(),
    paired_training_seeds=seeds, accepted_groups=target, data=receipt), indent=2) + "\n")
print(BASE / "data-size-endpoints.png")
