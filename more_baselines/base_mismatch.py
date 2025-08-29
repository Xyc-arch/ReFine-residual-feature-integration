# more_baselines/base_mismatch.py
# Domain mismatch: Teacher on CIFAR-100 -> Adapt to CIFAR-10 with (1) LoRA and (2) DANN-Gate
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

# Robust import for BigCNN (teacher on CIFAR-100)
try:
    # Some repos keep CIFAR-100 model under this path
    from model_def_test10.model_def100 import BigCNN
except Exception:
    # Fallback to the standard CIFAR-100 path
    from model_def_test100.model_def import BigCNN

# Train/eval utilities (CIFAR-10 evaluation is fine; metric heads are determined by model output size)
from train_eval import train_model, evaluate_model

# LoRA & DANN-Gate implementations reused from your base utilities
from more_baselines.base_train_eval import train_lora, train_dann_gate

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
      - augment_set: additional subset (kept for parity; not used here)
    Returns raw_set, augment_set, testset.
    """
    rng = random.Random(seed_for_split)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
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
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
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
def make_teacher_for_cifar10(external_model_100):
    """
    Deep-copy the CIFAR-100 teacher and REPLACE the final FC layer (2560->100)
    with a new 10-class layer (2560->10).
    """
    model = copy.deepcopy(external_model_100)
    last = model.fc_layers[-1]
    if not isinstance(last, nn.Linear):
        raise RuntimeError("Expected final layer of BigCNN.fc_layers to be nn.Linear.")
    in_dim = last.in_features
    model.fc_layers[-1] = nn.Linear(in_dim, 10)
    return model


def main():
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size      = 64
    pretrain_epochs = 60   # teacher on CIFAR-100 subset
    adapt_epochs    = 30   # LoRA / DANN-Gate epochs on CIFAR-10
    num_runs        = 5

    save_path    = "./results_test10_base/mismatch.json"
    teacher_ckpt = "./model_test10/base_mismatch_teacher_cifar100.pt"

    # 1) Pretrain teacher on CIFAR-100 subset
    print("\n=== Pretraining Teacher (BigCNN) on CIFAR-100 subset ===")
    teacher_subset, _ = load_cifar100_teacher_subset(pretrain_size=10000, seed=42)
    teacher_loader = DataLoader(teacher_subset, batch_size=batch_size, shuffle=True, num_workers=2)

    if os.path.exists(teacher_ckpt):
        teacher_model_100 = torch.load(teacher_ckpt).to(device)
        print("Loaded teacher model from:", teacher_ckpt)
    else:
        teacher_model_100 = BigCNN().to(device)
        train_model(teacher_model_100, teacher_loader, pretrain_epochs, device)
        os.makedirs(os.path.dirname(teacher_ckpt), exist_ok=True)
        torch.save(teacher_model_100, teacher_ckpt)
        print("Trained and saved teacher model to:", teacher_ckpt)

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

        # 3a) LoRA: head -> 10 classes, then inject LoRA at the head
        print("Training LoRA (teacher -> 10-class head + LoRA) ...")
        lora_teacher = make_teacher_for_cifar10(teacher_model_100).to(device)
        acc, auc, f1, minc = train_lora(lora_teacher, raw_loader_10, test_loader_10, device, epochs=adapt_epochs)
        metrics["lora_10"]["acc"].append(acc)
        metrics["lora_10"]["auc"].append(auc)
        metrics["lora_10"]["f1"].append(f1)
        metrics["lora_10"]["min_cacc"].append(minc)

        # 3b) DANN-Gate: head -> 10 classes
        print("Training DANN-Gate (teacher -> 10-class head) ...")
        dann_teacher = make_teacher_for_cifar10(teacher_model_100).to(device)
        acc, auc, f1, minc = train_dann_gate(dann_teacher, raw_loader_10, test_loader_10, device, epochs=adapt_epochs)
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
            "min_cacc_mean": float(minc_arr.mean()), "min_cacc_std": float(minc_arr.std())
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final_results, fp, indent=2)

    print(f"\nAll done. Final mean/std results saved to: {save_path}")


if __name__ == "__main__":
    main()
