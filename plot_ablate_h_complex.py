# plot_ablate_h_complex.py
# No-arg plotting script (bar charts) for ablation over h with STD overlays and 95% CI (no legend entries).
# Control everything via the variables in the "USER CONTROLS" section below.

import json
import os
from typing import Dict, Any, List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# =========================
# ===== USER CONTROLS =====
# =========================

# 0=param chart, 1=noise(0.8) acc chart, 2=clean(0.0) acc chart, 3=all
MODE = 3

# Paths
JSON_PATH = "./results_ablate/h_complex.json"
OUTDIR    = "./plots/ablate_h_complex"

# Figure styling (enlarged)
FIG_WIDTH   = 11.0
FIG_HEIGHT  = 8
LABEL_SIZE  = 25
TICK_SIZE   = 20
TITLE_SIZE  = 20
LEGEND_SIZE = 20
ROTATE_XTICKS = 25

# Bars
BAR_WIDTH   = 0.35
CAPSIZE_STD = 5
CAPSIZE_CI  = 0    # CI drawn as clean lines without caps

# Seeds used to compute CI = 1.96 * std / sqrt(N_SEEDS)
N_SEEDS = 5
CI_Z = 1.96

# Names / titles / filenames
LABEL_PRETRAINED = "Pretrained"
LABEL_SCRATCH    = "NoTrans"
LABEL_ENHANCED   = "Refine"

TITLE_PARAM  = "Parameter count vs h"
TITLE_NOISE  = "Accuracy vs h (noise=0.8)"
TITLE_CLEAN  = "Accuracy vs h (clean, noise=0.0)"

FILENAME_PARAM = "ablate_h_complex_acc_param.png"
FILENAME_NOISE = "ablate_h_complex_acc_noise_0.8.png"
FILENAME_CLEAN = "ablate_h_complex_acc_noise_0.0.png"

# =========================
# ====== DEFINITIONS ======
# =========================

def get_h_grids() -> List[Dict[str, Any]]:
    return [
        dict(name="h0_tiny",   channels=[16, 32, 32],  fc=128),
        dict(name="h1_small",  channels=[32, 64, 64],  fc=256),
        dict(name="h2_base",   channels=[32, 64, 128], fc=512),
        dict(name="h3_large",  channels=[64, 128,128], fc=768),
        dict(name="h4_xlarge", channels=[64, 128,256], fc=1024),
    ]

class CNNh(nn.Module):
    def __init__(self, channels: List[int], fc_dim: int):
        super().__init__()
        convs = []
        in_c = 3
        for c in channels:
            convs += [nn.Conv2d(in_c, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)]
            in_c = c
        self.conv = nn.Sequential(*convs)
        self.fc_feat = nn.Sequential(nn.Linear(in_c*4*4, fc_dim), nn.ReLU(), nn.Dropout(0.5))
        self.fc_head = nn.Linear(fc_dim, 10)

class EnhancedHead(nn.Module):
    def __init__(self, local_feat_dim: int, ext_dim: int = 2560):
        super().__init__()
        self.final = nn.Linear(local_feat_dim + ext_dim, 10)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_results(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)

def x_labels_and_ticks(hgrids: List[Dict[str, Any]]):
    labels = [g["name"] for g in hgrids]
    xs = np.arange(len(labels))
    return xs, labels

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def apply_style():
    plt.rcParams.update({
        "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_SIZE,
    })

def ci_from_std(std: float, n: int = N_SEEDS, z: float = CI_Z) -> float:
    if n <= 1:
        return 0.0
    return z * (std / np.sqrt(n))

# =========================
# ========= PLOTS =========
# =========================

def plot_param():
    ensure_outdir(OUTDIR)
    apply_style()
    hgrids = get_h_grids()
    xs, labels = x_labels_and_ticks(hgrids)

    scratch_params = []
    enhanced_params = []
    for g in hgrids:
        base = CNNh(g["channels"], g["fc"])
        n_scratch = count_params(base)
        head = EnhancedHead(local_feat_dim=g["fc"], ext_dim=2560)
        n_enh = count_params(base) + count_params(head)
        scratch_params.append(n_scratch)
        enhanced_params.append(n_enh)

    plt.figure()
    # Different color palette for param plot
    plt.bar(xs - BAR_WIDTH/2, scratch_params, width=BAR_WIDTH, label="NoTrans", color="tab:green")
    plt.bar(xs + BAR_WIDTH/2, enhanced_params, width=BAR_WIDTH, label="Refine", color="tab:purple")
    plt.xticks(xs, labels, rotation=ROTATE_XTICKS)
    plt.ylabel("# trainable parameters")
    # plt.title(TITLE_PARAM)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    out = os.path.join(OUTDIR, FILENAME_PARAM)
    plt.tight_layout()
    plt.savefig(out, dpi=240)
    print(f"Saved: {out}")

def _plot_acc_for_ratio(results: Dict[str, Any], ratio_key: str, title: str, outpath: str):
    ensure_outdir(OUTDIR)
    apply_style()
    hgrids = get_h_grids()
    xs, labels = x_labels_and_ticks(hgrids)

    r = results[ratio_key]
    pretrained_acc = r["pretrained"]["acc"]

    scratch_mean = np.array([r["grids"][g["name"]]["scratch"]["acc_mean"] for g in hgrids])
    scratch_std  = np.array([r["grids"][g["name"]]["scratch"]["acc_std"]  for g in hgrids])
    scratch_ci   = np.array([ci_from_std(s) for s in scratch_std])

    enh_mean = np.array([r["grids"][g["name"]]["enhanced"]["acc_mean"] for g in hgrids])
    enh_std  = np.array([r["grids"][g["name"]]["enhanced"]["acc_std"]  for g in hgrids])
    enh_ci   = np.array([ci_from_std(s) for s in enh_std])

    plt.figure()
    # Keep accuracy plots in blue/orange
    plt.bar(xs - BAR_WIDTH/2, scratch_mean, width=BAR_WIDTH, label=LABEL_SCRATCH, yerr=scratch_std,
            capsize=CAPSIZE_STD, alpha=0.9, edgecolor="black", linewidth=0.5, color="tab:blue")
    plt.bar(xs + BAR_WIDTH/2, enh_mean, width=BAR_WIDTH, label=LABEL_ENHANCED, yerr=enh_std,
            capsize=CAPSIZE_STD, alpha=0.9, edgecolor="black", linewidth=0.5, color="tab:orange")

    # CI overlays (no legend entries)
    plt.errorbar(xs - BAR_WIDTH/2, scratch_mean, yerr=scratch_ci, fmt="none",
                 ecolor="k", elinewidth=2.0, capsize=CAPSIZE_CI, alpha=0.8)
    plt.errorbar(xs + BAR_WIDTH/2, enh_mean, yerr=enh_ci, fmt="none",
                 ecolor="dimgray", elinewidth=2.0, capsize=CAPSIZE_CI, alpha=0.8)

    # Bold, obvious pretrained horizontal line
    plt.axhline(pretrained_acc, linestyle="--", linewidth=3.0, color="black",
                label=f"{LABEL_PRETRAINED} ({pretrained_acc:.2f}%)", alpha=0.9)

    plt.xticks(xs, labels, rotation=ROTATE_XTICKS)
    plt.ylabel("Accuracy (%)")
    # plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    print(f"Saved: {outpath}")

def plot_noise():
    results = load_results(JSON_PATH)
    out = os.path.join(OUTDIR, FILENAME_NOISE)
    _plot_acc_for_ratio(results, "0.8", TITLE_NOISE, out)

def plot_clean():
    results = load_results(JSON_PATH)
    out = os.path.join(OUTDIR, FILENAME_CLEAN)
    _plot_acc_for_ratio(results, "0.0", TITLE_CLEAN, out)

# =========================
# ========= MAIN ==========
# =========================

if __name__ == "__main__":
    if MODE == 0:
        plot_param()
    elif MODE == 1:
        plot_noise()
    elif MODE == 2:
        plot_clean()
    elif MODE == 3:
        plot_param()
        plot_noise()
        plot_clean()
    else:
        raise ValueError("MODE must be one of {0:param, 1:noise, 2:clean, 3:all}")
