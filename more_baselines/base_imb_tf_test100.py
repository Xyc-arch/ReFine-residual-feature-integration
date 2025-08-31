#!/usr/bin/env python3
# more_baselines/base_im_tf_test100.py

import os, sys, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Ensure project root is on the path so imports work when run from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_def_test100.model_def_tf import BigTransformer
from train_eval_test100 import train_model, evaluate_model
from more_baselines.base_train_eval_test100 import train_lora_tf, train_dann_gate_tf

# Repro
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# --------------------------
# 1) Imbalanced pretraining dataset for CIFAR-100
# --------------------------
def load_data_split_imb_cifar100(seed,
                                 majority_classes=list(range(10)),
                                 majority_count=400,
                                 minority_count=100,
                                 target_total=10000):
    """
    Builds:
      • pretrain_subset: ~10k samples, imbalanced by (majority_classes, majority_count, minority_count)
      • raw_set: 4k clean samples (random)
      • testset: CIFAR-100 test
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )

    num_classes = 100
    indices_by_class = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset):
        indices_by_class[label].append(idx)

    rng = np.random.RandomState(seed)
    for c in range(num_classes):
        rng.shuffle(indices_by_class[c])

    pretrain_indices = []
    for c in range(num_classes):
        n = majority_count if c in majority_classes else minority_count
        pretrain_indices.extend(indices_by_class[c][:n])

    # Trim (or allow slight underfill) to hit target_total
    pretrain_indices = pretrain_indices[:target_total]
    pretrain_subset = Subset(trainset, pretrain_indices)

    # Build raw set (clean 4k)
    all_indices = np.arange(len(trainset))
    rng.shuffle(all_indices)
    raw_indices = all_indices[:4000]
    raw_set = Subset(trainset, raw_indices)

    return pretrain_subset, raw_set, testset


# --------------------------
# 2) Main (LoRA + DANN-Gate baselines only)
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path       = "./results_test100_base/imbalance_tf_cifar100.json"
    model_save_path = "./model_test100/base_imbalance_tf_cifar100.pt"

    pretrain_epochs = 60
    other_epochs    = 30
    num_runs        = 5

    # Build imbalanced pretraining set + test once
    pretrain_dataset, _, testset = load_data_split_imb_cifar100(seed=42)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True,  num_workers=2)
    test_loader     = DataLoader(testset,         batch_size=64, shuffle=False, num_workers=2)

    # 1) Pretrain external BigTransformer (on imbalanced data)
    if os.path.exists(model_save_path):
        external_model = torch.load(model_save_path).to(device)
        print("Loaded external transformer from:", model_save_path)
    else:
        external_model = BigTransformer().to(device)
        train_model(external_model, pretrain_loader, pretrain_epochs, device)
        torch.save(external_model, model_save_path)
        print("Trained and saved external transformer to:", model_save_path)

    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Transformer (Pretrained on Imbalanced Data): "
          f"Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    # Metrics (base variants only)
    metrics = {
        "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    # 2) Runs with different raw-set splits (clean)
    for run_idx in range(num_runs):
        seed_split = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_split} ===")

        _, raw_set, _ = load_data_split_imb_cifar100(seed=seed_split)
        raw_loader    = DataLoader(raw_set, batch_size=64, shuffle=True, num_workers=2)

        # LoRA on Transformer head (classifier only; backbone frozen)
        print("Training LoRA (Transformer head)...")
        acc, auc, f1, mc = train_lora_tf(external_model, raw_loader, test_loader, device, epochs=other_epochs)
        metrics["lora"]["acc"].append(acc)
        metrics["lora"]["auc"].append(auc)
        metrics["lora"]["f1"].append(f1)
        metrics["lora"]["min_cacc"].append(mc)

        # DANN-Gate (discriminator over logits; GRL updates LoRA head only)
        print("Training DANN-Gate (Transformer head)...")
        acc, auc, f1, mc = train_dann_gate_tf(external_model, raw_loader, test_loader, device, epochs=other_epochs)
        metrics["dann_gate"]["acc"].append(acc)
        metrics["dann_gate"]["auc"].append(auc)
        metrics["dann_gate"]["f1"].append(f1)
        metrics["dann_gate"]["min_cacc"].append(mc)

        print(f"[Run {run_idx+1}] "
              f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
              f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

    # 3) Aggregate and save
    final = {}
    for name, vals in metrics.items():
        a = np.array(vals["acc"])
        u = np.array(vals["auc"])
        f = np.array(vals["f1"])
        m = np.array(vals["min_cacc"])
        final[name] = {
            "acc_mean": float(a.mean()),     "acc_std": float(a.std()),
            "auc_mean": float(u.mean()),     "auc_std": float(u.std()),
            "f1_mean":  float(f.mean()),     "f1_std": float(f.std()),
            "min_cacc_mean": float(m.mean()), "min_cacc_std": float(m.std()),
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final, fp, indent=2)

    print(f"\nAll done. Results saved to: {save_path}")


if __name__ == "__main__":
    main()
