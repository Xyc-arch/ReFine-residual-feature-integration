import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0) PLOT CONFIGURATION — adjust these!
# ─────────────────────────────────────────────────────────────────────────────
FIGSIZE           = (6, 4)
BAR_WIDTH         = 0.6
ERROR_CAPSIZE     = 5
LEGEND_FONTSIZE   = 14       # e.g. 'small', 8, 12, etc.
TICK_LABELSIZE    = 12
AXIS_LABELSIZE    = 12
TITLE_FONTSIZE    = 14

OUTDIR            = "./results_ablate/ablate_adapters_plots"
os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load data
# ─────────────────────────────────────────────────────────────────────────────
with open("ablate_adapters.json") as f:
    ablate   = json.load(f)
with open("text_dvd2elec.json") as f:
    baseline = json.load(f)

multipliers = [1, 25, 50, 100, 500]
labels      = [f"{m}×" for m in multipliers]
metrics     = ["accuracy", "auc", "f1", "min_class_acc"]

# ─────────────────────────────────────────────────────────────────────────────
# 2) Plot loop
# ─────────────────────────────────────────────────────────────────────────────
for metric in metrics:
    # collect means & stds
    means = [ablate[f"adapter_{m}x"][metric]["mean"] for m in multipliers]
    stds  = [ablate[f"adapter_{m}x"][metric]["std"]  for m in multipliers]

    # baseline lines
    notrans = baseline["notrans"][metric]["mean"]
    refine  = baseline["refine" ][metric]["mean"]

    x = np.arange(len(multipliers))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # mean bars with error‐bars
    bars = ax.bar(
        x,
        means,
        width=BAR_WIDTH,
        yerr=stds,
        capsize=ERROR_CAPSIZE,
        label="Adapter mean ± std",
        edgecolor="black",
        linewidth=0.8
    )

    # dashed baseline lines
    ax.axhline(notrans, color="C1", linestyle="--", label="NoTrans mean")
    ax.axhline(refine,  color="C2", linestyle="--", label="Refine mean")

    # labels & ticks
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_LABELSIZE)
    ax.set_xlabel("Adapter size multiplier", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=AXIS_LABELSIZE)
    ax.set_title(f"{metric.replace('_',' ').title()} vs Adapter size", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)

    # legend
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{metric}.png"))
    plt.close()

print(f"Plots saved under {OUTDIR}/")
