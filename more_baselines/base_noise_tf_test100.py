#!/usr/bin/env python3
# more_baselines/base_noise_tf_test100.py

import os, sys, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
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


def load_data_split(seed, flip_ratio=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    # Match your original script: only clean pretraining
    flip_ratios = [0.8, 0.0]  # change to [0.8, 0.0] if you also want a noisy-pretrain variant

    for flip_ratio in flip_ratios:
        save_path       = f"./results_test100_base/noise_tf_cifar100_{flip_ratio}.json"
        model_save_path = f"./model_test100/base_noise_tf_cifar100_{flip_ratio}.pt"

        pretrain_dataset, raw_set_common, testset = load_data_split(seed=42, flip_ratio=flip_ratio)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True,  num_workers=2)
        test_loader     = DataLoader(testset,         batch_size=64, shuffle=False, num_workers=2)

        # 1) Pretrain external BigTransformer (on possibly corrupted labels)
        if os.path.exists(model_save_path):
            external_model = torch.load(model_save_path).to(device)
            print("Loaded external transformer from:", model_save_path)
        else:
            external_model = BigTransformer(num_classes=100).to(device)
            train_model(external_model, pretrain_loader, pretrain_epochs, device)
            torch.save(external_model, model_save_path)
            print("Trained and saved external transformer to:", model_save_path)

        ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
        print(f"External Transformer Eval: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

        metrics = {
            "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
            "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        }

        for run_idx in range(num_runs):
            run_seed = 42 + run_idx
            print(f"\n=== flip_ratio={flip_ratio} | Run {run_idx+1}/{num_runs}, seed={run_seed} ===")
            _, raw_set, _ = load_data_split(seed=run_seed, flip_ratio=flip_ratio)
            run_loader    = DataLoader(raw_set, batch_size=64, shuffle=True, num_workers=2)

            # 2) LoRA on TF head (only final classifier is LoRA-ized)
            print("Training LoRA (Transformer head; CIFAR-100)...")
            acc, auc, f1, mc = train_lora_tf(external_model, run_loader, test_loader, device, epochs=other_epochs)
            metrics["lora"]["acc"].append(acc)
            metrics["lora"]["auc"].append(auc)
            metrics["lora"]["f1"].append(f1)
            metrics["lora"]["min_cacc"].append(mc)

            # 3) DANN-Gate on TF head
            print("Training DANN-Gate (Transformer head; CIFAR-100)...")
            acc, auc, f1, mc = train_dann_gate_tf(external_model, run_loader, test_loader, device, epochs=other_epochs)
            metrics["dann_gate"]["acc"].append(acc)
            metrics["dann_gate"]["auc"].append(auc)
            metrics["dann_gate"]["f1"].append(f1)
            metrics["dann_gate"]["min_cacc"].append(mc)

            print(f"[Run {run_idx+1}] "
                  f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
                  f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

        # aggregate mean/std
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

        print(f"\nAll done. Final mean/std results saved to: {save_path}")


if __name__ == "__main__":
    main()
