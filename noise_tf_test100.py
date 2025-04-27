import torch
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

def load_data_split(seed, flip_ratio=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )

    total_indices = np.arange(len(trainset))
    rng = np.random.RandomState(seed)
    rng.shuffle(total_indices)

    pretrain_indices = total_indices[:10000]
    raw_indices      = total_indices[10000:10000+4000]

    pretrain_subset = Subset(trainset, pretrain_indices)
    raw_set         = Subset(trainset, raw_indices)

    class RandomLabelDataset(Dataset):
        def __init__(self, subset, num_classes=100, flip_ratio=1.0):
            self.subset = subset
            self.num_classes = num_classes
            self.flip_ratio = flip_ratio
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            image, true_label = self.subset[idx]
            if np.random.rand() < self.flip_ratio:
                wrong_label = np.random.randint(0, self.num_classes - 1)
                if wrong_label >= true_label:
                    wrong_label += 1
                return image, wrong_label
            else:
                return image, true_label

    pretrain_dataset = RandomLabelDataset(pretrain_subset, num_classes=100, flip_ratio=flip_ratio)
    return pretrain_dataset, raw_set, testset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_epochs = 60
    other_epochs    = 30
    num_runs        = 5

    for flip_ratio in [0.8]:
        save_path       = f"./results_test100/noise_tf_cifar100_{flip_ratio}.json"
        model_save_path = f"./model_test100/noise_tf_cifar100_{flip_ratio}.pt"

        pretrain_dataset, raw_set_common, testset = load_data_split(seed=42, flip_ratio=flip_ratio)
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
        print(f"External Transformer Eval: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

        metrics = {
            "baseline":         {"acc": [], "auc": [], "f1": [], "min_cacc": []},
            "linear_prob":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
            "enhanced_concat":  {"acc": [], "auc": [], "f1": [], "min_cacc": []},
            "baseline_adapter": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
            "distillation":     {"acc": [], "auc": [], "f1": [], "min_cacc": []}
        }

        for run_idx in range(num_runs):
            run_seed = 42 + run_idx
            print(f"\n=== Run {run_idx+1}/{num_runs}, raw-set seed={run_seed} ===")
            _, raw_set, _ = load_data_split(seed=run_seed, flip_ratio=flip_ratio)
            run_loader    = DataLoader(raw_set, batch_size=64, shuffle=True, num_workers=2)

            # 2.1 Baseline Transformer
            print("Training baseline TransformerClassifier on raw set...")
            baseline = TransformerClassifier().to(device)
            train_model(baseline, run_loader, other_epochs, device)
            acc, auc, f1, mc = evaluate_model(baseline, test_loader, device)
            metrics["baseline"]["acc"].append(acc)
            metrics["baseline"]["auc"].append(auc)
            metrics["baseline"]["f1"].append(f1)
            metrics["baseline"]["min_cacc"].append(mc)

            # 2.2 Linear Probe
            print("Training linear probe (freeze all but classifier)...")
            lp_model = copy.deepcopy(external_model)
            for p in lp_model.parameters():
                p.requires_grad = False
            for p in lp_model.classifier.parameters():
                p.requires_grad = True
            train_linear_prob(lp_model, run_loader, other_epochs, device)
            acc, auc, f1, mc = evaluate_model(lp_model, test_loader, device)
            metrics["linear_prob"]["acc"].append(acc)
            metrics["linear_prob"]["auc"].append(auc)
            metrics["linear_prob"]["f1"].append(f1)
            metrics["linear_prob"]["min_cacc"].append(mc)

            # 2.3 Enhanced Concat
            print("Training EnhancedTransformer (concat features)...")
            enh = EnhancedTransformer().to(device)
            train_enhanced_model(enh, run_loader, external_model, other_epochs, device)
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
            train_model(ba, run_loader, other_epochs, device)
            acc, auc, f1, mc = evaluate_model(ba, test_loader, device)
            metrics["baseline_adapter"]["acc"].append(acc)
            metrics["baseline_adapter"]["auc"].append(auc)
            metrics["baseline_adapter"]["f1"].append(f1)
            metrics["baseline_adapter"]["min_cacc"].append(mc)

            # 2.5 Knowledge Distillation
            print("Training distillation student TransformerClassifier...")
            student = TransformerClassifier().to(device)
            train_distillation(student, external_model, run_loader,
                               other_epochs, device, temperature=2.0, alpha=0.5)
            acc, auc, f1, mc = evaluate_model(student, test_loader, device)
            metrics["distillation"]["acc"].append(acc)
            metrics["distillation"]["auc"].append(auc)
            metrics["distillation"]["f1"].append(f1)
            metrics["distillation"]["min_cacc"].append(mc)

            print(f"[Run {run_idx+1} Results] "
                  f"Base={metrics['baseline']['acc'][-1]:.2f}% | "
                  f"LP={metrics['linear_prob']['acc'][-1]:.2f}% | "
                  f"ENH={metrics['enhanced_concat']['acc'][-1]:.2f}% | "
                  f"ADP={metrics['baseline_adapter']['acc'][-1]:.2f}% | "
                  f"DIST={metrics['distillation']['acc'][-1]:.2f}%")

        # aggregate results
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
                "min_cacc_mean": float(m.mean()), "min_cacc_std": float(m.std())
            }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as fp:
            json.dump(final, fp, indent=2)

        print(f"\nAll done. Final mean/std results saved to: {save_path}")

if __name__ == "__main__":
    main()
