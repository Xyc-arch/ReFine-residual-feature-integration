# mismatch_tf_test100.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import json
import os
import copy
from torch.utils.data import DataLoader, Subset

from train_eval_test100 import (
    train_model,
    train_linear_prob,
    train_enhanced_model,
    evaluate_model
)
from model_def_test100.model_def10_tf import (
    TransformerClassifier,
    EnhancedTransformer,
    BaselineAdapterTransformer,
    BigTransformer
)

# ─── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------
# Data Loading for CIFAR-100 (Student)
# --------------------------
def load_student_data(seed, raw_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    full_train = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    indices = list(range(len(full_train)))
    random.Random(seed).shuffle(indices)
    raw_idx = indices[:raw_size]
    raw_set = Subset(full_train, raw_idx)
    return raw_set, test_set

# --------------------------
# Data Loading for CIFAR-10 (Teacher)
# --------------------------
def load_teacher_data(pretrain_size=10000, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    indices = list(range(len(full_train)))
    random.Random(seed).shuffle(indices)
    subset = Subset(full_train, indices[:pretrain_size])
    return subset, test_set

# --------------------------
# Main Experiment Loop
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./results_test100/mismatch_tf_reverse.json"

    num_epochs_teacher = 60
    num_epochs_student = 30
    num_runs = 5

    # ─── Train or Load Teacher ───────────────────────────────────────────────────
    teacher_path = "./model_test100/mismatch_tf_teacher.pt"
    if os.path.exists(teacher_path):
        teacher = torch.load(teacher_path).to(device)
        print("Loaded teacher model from:", teacher_path)
    else:
        print("Training teacher model on CIFAR-10 subset...")
        train_ds, _ = load_teacher_data(pretrain_size=10000, seed=42)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        teacher = BigTransformer().to(device)
        train_model(teacher, train_loader, num_epochs_teacher, device)
        os.makedirs(os.path.dirname(teacher_path), exist_ok=True)
        torch.save(teacher, teacher_path)
        print("Saved teacher model to:", teacher_path)

    # ─── Prepare Student Test Set ─────────────────────────────────────────────────
    _, test_student = load_student_data(seed=42, raw_size=2000)
    test_loader = DataLoader(test_student, batch_size=32, shuffle=False, num_workers=2)

    # ─── Metrics Containers ──────────────────────────────────────────────────────
    metrics = {
        "baseline":       {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "linear_prob":    {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "enhanced_concat":{"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "baseline_adapter":{"acc": [], "auc": [], "f1": [], "min_cacc": []}
    }

    # ─── Runs ────────────────────────────────────────────────────────────────────
    for run_idx in range(num_runs):
        seed_run = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_run} ===")

        # Student raw set
        raw_set, _ = load_student_data(seed=seed_run, raw_size=2000)
        raw_loader = DataLoader(raw_set, batch_size=32, shuffle=True, num_workers=2)

        # 1) Baseline student
        print("Training baseline student...")
        student = TransformerClassifier().to(device)
        train_model(student, raw_loader, num_epochs_student, device)
        acc, auc, f1, minc = evaluate_model(student, test_loader, device)
        metrics["baseline"]["acc"].append(acc)
        metrics["baseline"]["auc"].append(auc)
        metrics["baseline"]["f1"].append(f1)
        metrics["baseline"]["min_cacc"].append(minc)

        # 2) Linear probe on teacher with new 100-way head
        print("Training linear probe on teacher...")
        lp = copy.deepcopy(teacher)
        # freeze all pretrained weights
        for p in lp.parameters():
            p.requires_grad = False
        # replace the old 10-class head with a new 100-class layer
        in_feats = lp.classifier.in_features
        lp.classifier = nn.Linear(in_feats, 100).to(device)
        # enable training on the new head
        for p in lp.classifier.parameters():
            p.requires_grad = True
        train_linear_prob(lp, raw_loader, num_epochs_student, device)
        acc, auc, f1, minc = evaluate_model(lp, test_loader, device)
        metrics["linear_prob"]["acc"].append(acc)
        metrics["linear_prob"]["auc"].append(auc)
        metrics["linear_prob"]["f1"].append(f1)
        metrics["linear_prob"]["min_cacc"].append(minc)

        # 3) Enhanced concatenation
        print("Training enhanced (concatenation) model...")
        enh = EnhancedTransformer().to(device)
        train_enhanced_model(enh, raw_loader, teacher, num_epochs_student, device)
        acc, auc, f1, minc = evaluate_model(
            enh, test_loader, device,
            enhanced=True, external_model=teacher
        )
        metrics["enhanced_concat"]["acc"].append(acc)
        metrics["enhanced_concat"]["auc"].append(auc)
        metrics["enhanced_concat"]["f1"].append(f1)
        metrics["enhanced_concat"]["min_cacc"].append(minc)

        # 4) Baseline adapter
        print("Training baseline adapter model...")
        adp = BaselineAdapterTransformer(copy.deepcopy(teacher)).to(device)
        train_model(adp, raw_loader, num_epochs_student, device)
        acc, auc, f1, minc = evaluate_model(adp, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc)
        metrics["baseline_adapter"]["auc"].append(auc)
        metrics["baseline_adapter"]["f1"].append(f1)
        metrics["baseline_adapter"]["min_cacc"].append(minc)

        print(f"Run {run_idx+1} results: "
              f"Base={metrics['baseline']['acc'][-1]:.2f}%, "
              f"LP={metrics['linear_prob']['acc'][-1]:.2f}%, "
              f"ENH={metrics['enhanced_concat']['acc'][-1]:.2f}%, "
              f"ADP={metrics['baseline_adapter']['acc'][-1]:.2f}%")

    # ─── Aggregate and Save ──────────────────────────────────────────────────────
    final = {}
    for method, m in metrics.items():
        acc = np.array(m["acc"]); auc = np.array(m["auc"])
        f1  = np.array(m["f1"]);  mn  = np.array(m["min_cacc"])
        final[method] = {
            "acc_mean": acc.mean().item(),    "acc_std": acc.std().item(),
            "auc_mean": auc.mean().item(),    "auc_std": auc.std().item(),
            "f1_mean":  f1.mean().item(),     "f1_std":  f1.std().item(),
            "minc_mean": mn.mean().item(),    "minc_std": mn.std().item()
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final, fp, indent=2)

    print("\nAll runs complete. Results saved to", save_path)

if __name__ == "__main__":
    main()
