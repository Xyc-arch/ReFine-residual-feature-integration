#!/usr/bin/env python3
# more_baselines/base_adversarial_tf.py

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


# -----------------------------------------------------------------------------
# 1) Adversarial wrapper: paired label flips + additive white noise
# -----------------------------------------------------------------------------
class PairedLabelCorruptedDataset(Dataset):
    """
    Wraps a dataset and applies:
      • Paired-class label flips with probability p_flip
      • Additive white Gaussian noise (std = noise_std) to each image

    CIFAR-10 pairing:
      3<->5 (cat<->dog), 4<->7 (deer<->horse), 1<->9 (auto<->truck), 0<->8 (airplane<->ship)
    """
    def __init__(self, dataset, p_flip=0.5, noise_std=0.5, seed=42):
        self.dataset = dataset
        self.p_flip = p_flip
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self.paired = {
            3: 5, 5: 3,
            4: 7, 7: 4,
            1: 9, 9: 1,
            0: 8, 8: 0,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # image already transformed tensor
        if label in self.paired and self.rng.random() < self.p_flip:
            label = self.paired[label]
        noise = torch.randn_like(image) * self.noise_std
        return image + noise, label


# -----------------------------------------------------------------------------
# 2) Data loading / splitting (matches your original logic)
# -----------------------------------------------------------------------------
def load_and_split_data(seed_for_split, use_adversarial=False, p_flip=0.5, noise_std=0.1):
    """
    Loads CIFAR-10 and splits training into:
      • raw_set:   4,000 samples (clean)
      • augment_set: 10,000 samples (optionally adversarially corrupted)
    """
    rg = random.Random(seed_for_split)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    total_size   = len(trainset)
    raw_size     = 4000
    augment_size = 10000

    indices = list(range(total_size))
    rg.shuffle(indices)

    raw_indices     = indices[:raw_size]
    augment_indices = indices[raw_size:raw_size+augment_size]

    raw_set     = Subset(trainset, raw_indices)
    augment_set = Subset(trainset, augment_indices)

    if use_adversarial:
        augment_set = PairedLabelCorruptedDataset(augment_set, p_flip=p_flip, noise_std=noise_std, seed=seed_for_split)

    return raw_set, augment_set, testset


# -----------------------------------------------------------------------------
# 3) Main: pretrain external BigTransformer on adversarial augment set,
#          then evaluate LoRA & DANN-Gate on raw_set across seeds
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path       = "./results_test10_base/adversarial_tf.json"
    model_save_path = "./model_test10/base_adversarial_tf.pt"

    pretrain_epochs = 60
    other_epochs    = 30
    num_runs        = 5

    # Adversarial corruption hyperparams (match original file)
    p_flip   = 0.5   # 50% chance to flip paired classes
    noise_std = 0.2  # std of additive white noise

    # Test set (common)
    _, _, testset = load_and_split_data(seed_for_split=42)
    test_loader   = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # ---- Pretrain external model on adversarial augment set ----
    print("\n=== Pretraining External Model (BigTransformer) on Adversarial Augment Set ===")
    if os.path.exists(model_save_path):
        external_model = torch.load(model_save_path).to(device)
        print("Loaded external model from:", model_save_path)
    else:
        _, augment_set_ext, _ = load_and_split_data(
            seed_for_split=42, use_adversarial=True, p_flip=p_flip, noise_std=noise_std
        )
        augment_loader_ext = DataLoader(augment_set_ext, batch_size=32, shuffle=True, num_workers=2)
        external_model = BigTransformer().to(device)
        train_model(external_model, augment_loader_ext, pretrain_epochs, device)
        torch.save(external_model, model_save_path)
        print("Trained and saved external model to:", model_save_path)

    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Transformer Eval: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    # Metrics containers (base variants only)
    metrics = {
        "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    # ---- Runs with different raw_set splits (clean) ----
    for run_idx in range(num_runs):
        seed_for_split = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_for_split} ===")

        raw_set, _, _ = load_and_split_data(seed_for_split)
        raw_loader    = DataLoader(raw_set, batch_size=32, shuffle=True, num_workers=2)

        # 1) LoRA on Transformer head (classifier only; backbone frozen)
        print("Training LoRA (Transformer head) ...")
        acc, auc, f1, mc = train_lora_tf(external_model, raw_loader, test_loader, device, epochs=other_epochs)
        metrics["lora"]["acc"].append(acc)
        metrics["lora"]["auc"].append(auc)
        metrics["lora"]["f1"].append(f1)
        metrics["lora"]["min_cacc"].append(mc)

        # 2) DANN-Gate on Transformer head
        print("Training DANN-Gate (Transformer head) ...")
        acc, auc, f1, mc = train_dann_gate_tf(external_model, raw_loader, test_loader, device, epochs=other_epochs)
        metrics["dann_gate"]["acc"].append(acc)
        metrics["dann_gate"]["auc"].append(auc)
        metrics["dann_gate"]["f1"].append(f1)
        metrics["dann_gate"]["min_cacc"].append(mc)

        print(f"[Run {run_idx+1}] "
              f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
              f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

    # ---- Aggregate mean/std and save ----
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
