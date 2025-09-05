#!/usr/bin/env python3
# more_baselines/base_imb_tf.py

import os, sys, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms

# Ensure project root is on the path so imports work when run from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_def_test10.model_def_tf import BigTransformer
from train_eval import train_model, evaluate_model
from more_baselines.base_train_eval import train_lora_tf, train_dann_gate_tf

# Repro
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def normalize_dict(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


def load_data_split_imbalanced(seed, imbalance_dict):
    """
    CIFAR-10:
      - pretrain_set: first 10,000 samples, resampled by imbalance_dict (keys 0..9 -> proportions)
      - raw_set: next 4,000 samples with TRUE labels (no resampling)
      - testset: standard CIFAR-10 test set
    Only the pretrain_set is imbalanced; raw_set/testset are clean.
    """
    imbalance_dict = normalize_dict(imbalance_dict)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    total_indices = np.arange(len(trainset))
    rng = np.random.RandomState(seed)
    rng.shuffle(total_indices)

    pretrain_indices = total_indices[:10000]
    raw_indices      = total_indices[10000:10000+4000]

    pretrain_subset = Subset(trainset, pretrain_indices)
    raw_set         = Subset(trainset, raw_indices)

    # group indices by true label within pretrain_subset
    indices_by_label = {i: [] for i in range(10)}
    for i in range(len(pretrain_subset)):
        _, label = pretrain_subset[i]
        indices_by_label[label].append(i)

    total_pretrain = len(pretrain_subset)
    desired_counts = {c: int(total_pretrain * imbalance_dict.get(c, 0)) for c in range(10)}

    sampled_indices = []
    for c in range(10):
        available = indices_by_label[c]
        count = desired_counts[c]
        if len(available) >= count:
            sampled = rng.choice(available, size=count, replace=False).tolist()
        else:
            sampled = rng.choice(available, size=count, replace=True).tolist()
        sampled_indices.extend(sampled)

    pretrain_dataset = Subset(pretrain_subset, sampled_indices)
    return pretrain_dataset, raw_set, testset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_epochs = 60
    other_epochs    = 30
    num_runs        = 5

    # Desired imbalance distribution (must sum to 1; we normalize anyway)
    imbalance_dict = {
        0: 0.35,
        1: 0.30,
        2: 0.10,
        3: 0.07,
        4: 0.06,
        5: 0.045,
        6: 0.03,
        7: 0.02,
        8: 0.015,
        9: 0.01,
    }

    save_path       = "./results_test10_base/imb_tf.json"
    model_save_path = "./model_test10/imb_tf.pt"

    # Build datasets/loaders
    pretrain_dataset, raw_set, test_dataset = load_data_split_imbalanced(seed=42, imbalance_dict=imbalance_dict)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True,  num_workers=2)
    test_loader     = DataLoader(test_dataset,   batch_size=64, shuffle=False, num_workers=2)

    print("\n=== Pretraining External Model (BigTransformer) on 10k Imbalanced Samples ===")
    if os.path.exists(model_save_path):
        external_model = torch.load(model_save_path).to(device)
        print("Loaded external model from:", model_save_path)
    else:
        external_model = BigTransformer().to(device)
        train_model(external_model, pretrain_loader, pretrain_epochs, device)
        torch.save(external_model, model_save_path)
        print("Trained and saved external model to:", model_save_path)

    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Transformer Eval: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    metrics = {
        "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    for run_idx in range(num_runs):
        run_seed = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, raw-set seed={run_seed} ===")
        _, raw_set_run, _ = load_data_split_imbalanced(seed=run_seed, imbalance_dict=imbalance_dict)
        run_loader = DataLoader(raw_set_run, batch_size=64, shuffle=True, num_workers=2)

        # LoRA on Transformer head (classifier only)
        print("Training LoRA (Transformer head; imbalanced pretrain)...")
        acc, auc, f1, mc = train_lora_tf(external_model, run_loader, test_loader, device, epochs=other_epochs)
        metrics["lora"]["acc"].append(acc)
        metrics["lora"]["auc"].append(auc)
        metrics["lora"]["f1"].append(f1)
        metrics["lora"]["min_cacc"].append(mc)

        # DANN-Gate on Transformer head
        print("Training DANN-Gate (Transformer head; imbalanced pretrain)...")
        acc, auc, f1, mc = train_dann_gate_tf(external_model, run_loader, test_loader, device, epochs=other_epochs)
        metrics["dann_gate"]["acc"].append(acc)
        metrics["dann_gate"]["auc"].append(auc)
        metrics["dann_gate"]["f1"].append(f1)
        metrics["dann_gate"]["min_cacc"].append(mc)

        print(f"[Run {run_idx+1}] "
              f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
              f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

    # Aggregate mean/std
    final_results = {}
    for method, m in metrics.items():
        a = np.array(m["acc"])
        u = np.array(m["auc"])
        f = np.array(m["f1"])
        c = np.array(m["min_cacc"])
        final_results[method] = {
            "acc_mean": float(a.mean()),     "acc_std": float(a.std()),
            "auc_mean": float(u.mean()),     "auc_std": float(u.std()),
            "f1_mean":  float(f.mean()),     "f1_std": float(f.std()),
            "min_cacc_mean": float(c.mean()), "min_cacc_std": float(c.std()),
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final_results, fp, indent=2)

    print(f"\nAll done. Final mean/std results saved to: {save_path}")


if __name__ == "__main__":
    main()
