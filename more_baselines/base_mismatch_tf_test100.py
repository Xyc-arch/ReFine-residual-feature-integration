#!/usr/bin/env python3
# more_baselines/base_mismatch_tf_test100.py
# Reverse mismatch (Transformer): Teacher on CIFAR-10 -> Adapt to CIFAR-100 with LoRA-head and DANN-Gate-head.

import os
import sys
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Make repo root importable when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Teacher net: Transformer defined for CIFAR-10 (10 classes)
# Adjust import to your actual location if needed.
from model_def_test10.model_def_tf import BigTransformer as BigTransformer10

# CIFAR-100 train/eval utilities (expect 100-way logits)
from train_eval_test100 import train_model, evaluate_model

# LoRA / DANN-Gate trainers for CIFAR-100 (Transformer-head versions)
from more_baselines.base_train_eval_test100 import train_lora_tf, train_dann_gate_tf

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# --------------------------
# CIFAR-100 (student) split
# --------------------------
def load_cifar100_student(seed, raw_size=4000):
    """
    Returns a raw CIFAR-100 subset (raw_size) and the full CIFAR-100 test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    idx = list(range(len(trainset)))
    random.Random(seed).shuffle(idx)
    raw_idx = idx[:raw_size]
    raw_set = Subset(trainset, raw_idx)
    return raw_set, testset


# --------------------------
# CIFAR-10 (teacher) subset
# --------------------------
def load_cifar10_teacher_subset(pretrain_size=10000, seed=42):
    """
    Returns a CIFAR-10 train subset (pretrain_size) for teacher pretraining.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    idx = list(range(len(trainset)))
    random.Random(seed).shuffle(idx)
    subset = Subset(trainset, idx[:pretrain_size])
    return subset


# --------------------------
# Convert teacher head: 10 -> 100 (for CIFAR-100)
# --------------------------
def convert_teacher_to_100_tf(teacher_10cls: nn.Module) -> nn.Module:
    """
    Deep-copy the 10-class Transformer teacher and REPLACE the final linear head with 100-class.
    Assumes teacher has `.classifier` (nn.Linear) with in_features=2560.
    """
    model = copy.deepcopy(teacher_10cls)
    if not (hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear)):
        raise RuntimeError("Expected Transformer with a `.classifier` nn.Linear head.")
    in_dim = model.classifier.in_features  # typically 2560 from your BigTransformer
    model.classifier = nn.Linear(in_dim, 100)
    return model


def main():
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size      = 64
    pretrain_epochs = 60   # teacher on CIFAR-10 subset
    adapt_epochs    = 30   # LoRA / DANN-Gate on CIFAR-100
    num_runs        = 5

    # Paths
    save_path    = "./results_test100_base/mismatch_tf_test100.json"
    teacher_ckpt = "./model_test100/mismatch_tf_teacher_cifar10.pt"

    # 1) Pretrain teacher Transformer on CIFAR-10 subset (10k)
    print("\n=== Pretraining Teacher (BigTransformer-10) on CIFAR-10 subset ===")
    teacher_subset = load_cifar10_teacher_subset(pretrain_size=10000, seed=42)
    teacher_loader = DataLoader(teacher_subset, batch_size=batch_size, shuffle=True, num_workers=2)

    if os.path.exists(teacher_ckpt):
        teacher_10 = torch.load(teacher_ckpt).to(device)
        print("Loaded teacher transformer from:", teacher_ckpt)
    else:
        teacher_10 = BigTransformer10().to(device)
        train_model(teacher_10, teacher_loader, pretrain_epochs, device)
        os.makedirs(os.path.dirname(teacher_ckpt), exist_ok=True)
        torch.save(teacher_10, teacher_ckpt)
        print("Trained and saved teacher transformer to:", teacher_ckpt)

    # 2) CIFAR-100 test loader
    _, cifar100_test = load_cifar100_student(seed=42, raw_size=4000)
    test_loader_100 = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=2)

    # Metrics (base variants only)
    metrics = {
        "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    # 3) Adapt on multiple CIFAR-100 raw splits
    for run_idx in range(num_runs):
        seed_run = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_run} ===")
        raw_set_100, _ = load_cifar100_student(seed=seed_run, raw_size=4000)
        raw_loader_100 = DataLoader(raw_set_100, batch_size=batch_size, shuffle=True, num_workers=2)

        # LoRA-head: convert to 100-class head, then LoRA injected at the head
        print("Training LoRA (convert teacher head to 100, then LoRA-head)...")
        teacher_for_lora = convert_teacher_to_100_tf(teacher_10).to(device)
        acc, auc, f1, minc = train_lora_tf(teacher_for_lora, raw_loader_100, test_loader_100, device, epochs=adapt_epochs)
        metrics["lora"]["acc"].append(acc)
        metrics["lora"]["auc"].append(auc)
        metrics["lora"]["f1"].append(f1)
        metrics["lora"]["min_cacc"].append(minc)

        # DANN-Gate-head: convert to 100-class head, then adversarial gate where D sees logits
        print("Training DANN-Gate (convert teacher head to 100, head-only training)...")
        teacher_for_dann = convert_teacher_to_100_tf(teacher_10).to(device)
        acc, auc, f1, minc = train_dann_gate_tf(teacher_for_dann, raw_loader_100, test_loader_100, device, epochs=adapt_epochs)
        metrics["dann_gate"]["acc"].append(acc)
        metrics["dann_gate"]["auc"].append(auc)
        metrics["dann_gate"]["f1"].append(f1)
        metrics["dann_gate"]["min_cacc"].append(minc)

        print(f"[Run {run_idx+1}] "
              f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
              f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

    # 4) Aggregate and save
    final = {}
    for method, m in metrics.items():
        acc = np.array(m["acc"]);   auc = np.array(m["auc"])
        f1  = np.array(m["f1"]);    mn  = np.array(m["min_cacc"])
        final[method] = {
            "acc_mean": float(acc.mean()),   "acc_std": float(acc.std()),
            "auc_mean": float(auc.mean()),   "auc_std": float(auc.std()),
            "f1_mean":  float(f1.mean()),    "f1_std":  float(f1.std()),
            "min_cacc_mean": float(mn.mean()), "min_cacc_std": float(mn.std()),
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final, fp, indent=2)

    print(f"\nAll runs complete. Results saved to {save_path}")


if __name__ == "__main__":
    main()
