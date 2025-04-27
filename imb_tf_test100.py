import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import random
import json
import os
import copy

from model_def_test100.model_def_tf import (
    TransformerClassifier,
    EnhancedTransformer,
    BaselineAdapterTransformer,
    BigTransformer
)
from train_eval_test100 import (
    train_model,
    train_linear_prob,
    train_enhanced_model,
    train_distillation,
    evaluate_model
)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------
# 1. Function to build an imbalanced pretraining dataset for CIFAR-100
# --------------------------
def load_data_split_imb_cifar100(seed,
                                 majority_classes=list(range(10)),
                                 majority_count=400,
                                 minority_count=100,
                                 target_total=10000):
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

    # group indices by class
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
    pretrain_indices = pretrain_indices[:target_total]
    pretrain_subset = Subset(trainset, pretrain_indices)

    # raw set for student training
    all_indices = np.arange(len(trainset))
    rng.shuffle(all_indices)
    raw_indices = all_indices[:4000]
    raw_set = Subset(trainset, raw_indices)

    return pretrain_subset, raw_set, testset

# --------------------------
# 2. Main experiment loop (Imbalanced pretraining on CIFAR-100 using Transformers)
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./results_test100/imbalance_tf_cifar100.json"
    model_save_path = "./model_test100/imbalance_tf_cifar100.pt"

    pretrain_epochs = 60
    other_epochs = 30
    num_runs = 5

    # load imbalanced pretraining + test once
    pretrain_dataset, _, testset = load_data_split_imb_cifar100(seed=42)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader     = DataLoader(testset,         batch_size=64, shuffle=False, num_workers=2)

    # 1. Pretrain external BigTransformer
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

    # metrics container
    metrics = {
        "baseline":         {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "linear_prob":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "enhanced_concat":  {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "baseline_adapter": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "distillation":     {"acc": [], "auc": [], "f1": [], "min_cacc": []}
    }

    for run_idx in range(num_runs):
        seed_split = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_split} ===")

        # re-split raw set for student
        _, raw_set, _ = load_data_split_imb_cifar100(seed=seed_split)
        raw_loader = DataLoader(raw_set, batch_size=64, shuffle=True, num_workers=2)

        # 2.1 Baseline: TransformerClassifier on raw_set only
        print("Training baseline TransformerClassifier...")
        baseline = TransformerClassifier().to(device)
        train_model(baseline, raw_loader, other_epochs, device)
        acc, auc, f1, mc = evaluate_model(baseline, test_loader, device)
        metrics["baseline"]["acc"].append(acc)
        metrics["baseline"]["auc"].append(auc)
        metrics["baseline"]["f1"].append(f1)
        metrics["baseline"]["min_cacc"].append(mc)

        # 2.2 Linear Probe: fine-tune only classifier head
        print("Training linear probe (freeze all but classifier)...")
        lp_model = copy.deepcopy(external_model)
        for p in lp_model.parameters():
            p.requires_grad = False
        for p in lp_model.classifier.parameters():
            p.requires_grad = True
        train_linear_prob(lp_model, raw_loader, other_epochs, device)
        acc, auc, f1, mc = evaluate_model(lp_model, test_loader, device)
        metrics["linear_prob"]["acc"].append(acc)
        metrics["linear_prob"]["auc"].append(auc)
        metrics["linear_prob"]["f1"].append(f1)
        metrics["linear_prob"]["min_cacc"].append(mc)

        # 2.3 Enhanced Concatenation
        print("Training EnhancedTransformer (concat features)...")
        enh = EnhancedTransformer().to(device)
        train_enhanced_model(enh, raw_loader, external_model, other_epochs, device)
        acc, auc, f1, mc = evaluate_model(
            enh, test_loader, device, enhanced=True, external_model=external_model
        )
        metrics["enhanced_concat"]["acc"].append(acc)
        metrics["enhanced_concat"]["auc"].append(auc)
        metrics["enhanced_concat"]["f1"].append(f1)
        metrics["enhanced_concat"]["min_cacc"].append(mc)

        # 2.4 Baseline Adapter
        print("Training BaselineAdapterTransformer...")
        ba = BaselineAdapterTransformer(copy.deepcopy(external_model)).to(device)
        train_model(ba, raw_loader, other_epochs, device)
        acc, auc, f1, mc = evaluate_model(ba, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc)
        metrics["baseline_adapter"]["auc"].append(auc)
        metrics["baseline_adapter"]["f1"].append(f1)
        metrics["baseline_adapter"]["min_cacc"].append(mc)

        # 2.5 Knowledge Distillation
        print("Training distillation student TransformerClassifier...")
        student = TransformerClassifier().to(device)
        train_distillation(student, external_model, raw_loader,
                           other_epochs, device, temperature=2.0, alpha=0.5)
        acc, auc, f1, mc = evaluate_model(student, test_loader, device)
        metrics["distillation"]["acc"].append(acc)
        metrics["distillation"]["auc"].append(auc)
        metrics["distillation"]["f1"].append(f1)
        metrics["distillation"]["min_cacc"].append(mc)

        print(f"[Run {run_idx+1} results] "
              f"Base={metrics['baseline']['acc'][-1]:.2f}% | "
              f"LP={metrics['linear_prob']['acc'][-1]:.2f}% | "
              f"ENH={metrics['enhanced_concat']['acc'][-1]:.2f}% | "
              f"ADP={metrics['baseline_adapter']['acc'][-1]:.2f}% | "
              f"DIST={metrics['distillation']['acc'][-1]:.2f}%")

    # aggregate and save
    final = {}
    for name, vals in metrics.items():
        a = np.array(vals["acc"])
        u = np.array(vals["auc"])
        f = np.array(vals["f1"])
        m = np.array(vals["min_cacc"])
        final[name] = {
            "acc_mean": float(a.mean()), "acc_std": float(a.std()),
            "auc_mean": float(u.mean()), "auc_std": float(u.std()),
            "f1_mean": float(f.mean()), "f1_std": float(f.std()),
            "min_cacc_mean": float(m.mean()), "min_cacc_std": float(m.std())
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final, fp, indent=2)

    print(f"\nAll done. Results saved to: {save_path}")

if __name__ == "__main__":
    main()
