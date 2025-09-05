#!/usr/bin/env python3
# more_baselines/base_mismatch_tf.py
# Domain mismatch (Transformer): Teacher on CIFAR-100 -> Adapt to CIFAR-10 with (1) LoRA-head and (2) DANN-Gate-head

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

# Teacher (Transformer) defined for CIFAR-100
# If your BigTransformer lives elsewhere, adjust this import accordingly.
from model_def_test100.model_def_tf import BigTransformer as BigTransformer100

# CIFAR-10 train/eval utilities (metrics infer class-count from logits)
from train_eval import train_model, evaluate_model

# LoRA & DANN-Gate (Transformer-head versions)
from more_baselines.base_train_eval import train_lora_tf, train_dann_gate_tf

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# --------------------------
# CIFAR-10 (student) split
# --------------------------
def load_and_split_cifar10(seed_for_split, raw_size=4000, augment_size=4000):
    """
    Splits CIFAR-10 train set into:
      - raw_set: clean labeled subset (size=raw_size)
      - augment_set: extra subset (kept for parity; not used here)
    Returns raw_set, augment_set, testset.
    """
    rng = random.Random(seed_for_split)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    idx = list(range(len(trainset)))
    rng.shuffle(idx)
    raw_idx     = idx[:raw_size]
    augment_idx = idx[raw_size:raw_size+augment_size]

    raw_set     = Subset(trainset, raw_idx)
    augment_set = Subset(trainset, augment_idx)
    return raw_set, augment_set, testset


# --------------------------
# CIFAR-100 (teacher) subset
# --------------------------
def load_cifar100_teacher_subset(pretrain_size=10000, seed=42):
    """
    Returns a CIFAR-100 train subset of size `pretrain_size`, and the CIFAR-100 test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,  download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    idx = list(range(len(trainset)))
    random.Random(seed).shuffle(idx)
    subset_idx = idx[:pretrain_size]
    subset = Subset(trainset, subset_idx)
    return subset, testset


# --------------------------
# Replace teacher head: 100 -> 10 classes (for CIFAR-10)
# --------------------------
def make_teacher_for_cifar10_tf(external_model_100: nn.Module) -> nn.Module:
    """
    Deep-copy the CIFAR-100 Transformer teacher and REPLACE the final
    classifier (2560->100) with a new 10-class layer (2560->10).
    """
    model = copy.deepcopy(external_model_100)
    if not (hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear)):
        raise RuntimeError("Expected Transformer with a `.classifier` nn.Linear head.")
    in_dim = model.classifier.in_features  # expected 2560
    model.classifier = nn.Linear(in_dim, 10)
    return model


def main():
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size      = 64
    pretrain_epochs = 60   # teacher on CIFAR-100 subset
    adapt_epochs    = 30   # LoRA / DANN-Gate on CIFAR-10
    num_runs        = 5

    save_path    = "./results_test10_base/mismatch_tf.json"
    teacher_ckpt = "./model_test10/mismatch_tf_teacher_cifar100.pt"

    # 1) Pretrain Transformer teacher on CIFAR-100 subset
    print("\n=== Pretraining Teacher (BigTransformer-100) on CIFAR-100 subset ===")
    teacher_subset, _ = load_cifar100_teacher_subset(pretrain_size=10000, seed=42)
    teacher_loader = DataLoader(teacher_subset, batch_size=batch_size, shuffle=True, num_workers=2)

    if os.path.exists(teacher_ckpt):
        teacher_model_100 = torch.load(teacher_ckpt).to(device)
        print("Loaded teacher transformer from:", teacher_ckpt)
    else:
        teacher_model_100 = BigTransformer100().to(device)
        train_model(teacher_model_100, teacher_loader, pretrain_epochs, device)
        os.makedirs(os.path.dirname(teacher_ckpt), exist_ok=True)
        torch.save(teacher_model_100, teacher_ckpt)
        print("Trained and saved teacher transformer to:", teacher_ckpt)

    # 2) CIFAR-10 test loader
    _, _, cifar10_test = load_and_split_cifar10(seed_for_split=42)
    test_loader_10 = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2)

    # Metrics
    metrics = {
        "lora_10":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate_10": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    # 3) Adapt on multiple CIFAR-10 raw splits
    for run_idx in range(num_runs):
        seed_for_split = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_for_split} ===")
        raw_set_10, _, _ = load_and_split_cifar10(seed_for_split, raw_size=4000, augment_size=4000)
        raw_loader_10 = DataLoader(raw_set_10, batch_size=batch_size, shuffle=True, num_workers=2)

        # 3a) LoRA: convert head -> 10, then LoRA-on-head training
        print("Training LoRA (teacher -> 10-class head + LoRA) ...")
        lora_teacher = make_teacher_for_cifar10_tf(teacher_model_100).to(device)
        acc, auc, f1, minc = train_lora_tf(lora_teacher, raw_loader_10, test_loader_10, device, epochs=adapt_epochs)
        metrics["lora_10"]["acc"].append(acc)
        metrics["lora_10"]["auc"].append(auc)
        metrics["lora_10"]["f1"].append(f1)
        metrics["lora_10"]["min_cacc"].append(minc)

        # 3b) DANN-Gate: convert head -> 10, then adversarial gate training (head-only)
        print("Training DANN-Gate (teacher -> 10-class head) ...")
        dann_teacher = make_teacher_for_cifar10_tf(teacher_model_100).to(device)
        acc, auc, f1, minc = train_dann_gate_tf(dann_teacher, raw_loader_10, test_loader_10, device, epochs=adapt_epochs)
        metrics["dann_gate_10"]["acc"].append(acc)
        metrics["dann_gate_10"]["auc"].append(auc)
        metrics["dann_gate_10"]["f1"].append(f1)
        metrics["dann_gate_10"]["min_cacc"].append(minc)

        print(f"[Run {run_idx+1}] "
              f"LoRA Acc={metrics['lora_10']['acc'][-1]:.2f}% | "
              f"DANN-Gate Acc={metrics['dann_gate_10']['acc'][-1]:.2f}%")

    # 4) Aggregate and save
    final_results = {}
    for method, m in metrics.items():
        acc_arr  = np.array(m["acc"])
        auc_arr  = np.array(m["auc"])
        f1_arr   = np.array(m["f1"])
        minc_arr = np.array(m["min_cacc"])
        final_results[method] = {
            "acc_mean": float(acc_arr.mean()),   "acc_std": float(acc_arr.std()),
            "auc_mean": float(auc_arr.mean()),   "auc_std": float(auc_arr.std()),
            "f1_mean":  float(f1_arr.mean()),    "f1_std":  float(f1_arr.std()),
            "min_cacc_mean": float(minc_arr.mean()), "min_cacc_std": float(minc_arr.std()),
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final_results, fp, indent=2)

    print(f"\nAll done. Final mean/std results saved to: {save_path}")


if __name__ == "__main__":
    main()
