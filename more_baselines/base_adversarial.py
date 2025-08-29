# more_baselines/base_adversarial.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import random, os, json, sys

# Ensure project root is on the path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_def_test10.model_def import BigCNN
from train_eval import train_model, evaluate_model
from more_baselines.base_train_eval import train_lora, train_dann_gate

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# --------------------------
# 1) PairedLabelCorruptedDataset (same logic as adversarial.py)
# --------------------------
class PairedLabelCorruptedDataset(Dataset):
    """
    Applies adversarial paired label flipping and additive white Gaussian noise.

    Paired classes for CIFAR-10:
      - 3<->5 (cat<->dog), 4<->7 (deer<->horse),
        1<->9 (automobile<->truck), 0<->8 (airplane<->ship)

    With probability p_flip, a sample of a paired class flips to its pair.
    Always adds N(0, noise_std^2) pixel noise.
    """
    def __init__(self, dataset, p_flip=0.5, noise_std=0.2, seed=42):
        self.dataset = dataset
        self.p_flip = p_flip
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self.paired = {3:5, 5:3, 4:7, 7:4, 1:9, 9:1, 0:8, 8:0}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if label in self.paired and self.rng.random() < self.p_flip:
            label = self.paired[label]
        noise = torch.randn_like(image) * self.noise_std
        image = image + noise
        return image, label


# --------------------------
# 2) Data loading & split (mirrors adversarial.py)
# --------------------------
def load_and_split_data(seed_for_split, use_adversarial=False, p_flip=0.5, noise_std=0.2):
    """
    Split CIFAR-10 train into:
      - raw_set (4k)      : clean subset for adaptation
      - augment_set (10k) : pretraining subset (optionally adversarial)
    """
    rng = random.Random(seed_for_split)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    total = len(trainset)
    raw_size = 4000
    augment_size = 10000

    indices = list(range(total))
    rng.shuffle(indices)

    raw_idx     = indices[:raw_size]
    augment_idx = indices[raw_size:raw_size+augment_size]

    raw_set     = Subset(trainset, raw_idx)
    augment_set = Subset(trainset, augment_idx)

    if use_adversarial:
        augment_set = PairedLabelCorruptedDataset(augment_set, p_flip=p_flip, noise_std=noise_std, seed=seed_for_split)

    return raw_set, augment_set, testset


# --------------------------
# 3) Main: pretrain BigCNN then run LoRA & DANN-Gate on raw_set
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Settings aligned with adversarial.py
    pretrain_epochs = 60
    other_epochs    = 30
    num_runs        = 5
    batch_size      = 32

    # Adversarial corruption parameters for the pretraining subset
    p_flip   = 0.5
    noise_sd = 0.2

    # Results path (base version)
    save_path = "./results_test10_base/adversarial.json"

    # Load test set once
    _, _, testset = load_and_split_data(seed_for_split=42)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # ----- Pretrain or load external model (BigCNN) on adversarial augment_set -----
    print("\n=== Pretraining External Model (BigCNN) on Adversarial Pretraining Data ===")
    model_ckpt = "./model_test10/base_adversarial.pt"
    if os.path.exists(model_ckpt):
        external_model = torch.load(model_ckpt).to(device)
        print("Loaded external model from:", model_ckpt)
    else:
        _, augment_set_ext, _ = load_and_split_data(
            seed_for_split=42, use_adversarial=True, p_flip=p_flip, noise_std=noise_sd
        )
        augment_loader = DataLoader(augment_set_ext, batch_size=batch_size, shuffle=True)
        external_model = BigCNN().to(device)
        train_model(external_model, augment_loader, pretrain_epochs, device)
        torch.save(external_model, model_ckpt)
        print("Trained and saved external model to:", model_ckpt)

    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Model (Pretrained) Evaluation: "
          f"Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    # Metrics for base baselines only
    metrics = {
        "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    # ----- Runs over different raw splits -----
    for run_idx in range(num_runs):
        seed_for_split = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_for_split} ===")
        raw_set, _, _ = load_and_split_data(seed_for_split, use_adversarial=False)
        raw_loader = DataLoader(raw_set, batch_size=batch_size, shuffle=True)

        # LoRA: replace final classifier (head-only)
        print("Training LoRA...")
        acc, auc, f1, minc = train_lora(external_model, raw_loader, test_loader, device, epochs=other_epochs)
        metrics["lora"]["acc"].append(acc)
        metrics["lora"]["auc"].append(auc)
        metrics["lora"]["f1"].append(f1)
        metrics["lora"]["min_cacc"].append(minc)

        # DANN-Gate: adversarial gate on features with GRL
        print("Training DANN-Gate...")
        acc, auc, f1, minc = train_dann_gate(external_model, raw_loader, test_loader, device, epochs=other_epochs)
        metrics["dann_gate"]["acc"].append(acc)
        metrics["dann_gate"]["auc"].append(auc)
        metrics["dann_gate"]["f1"].append(f1)
        metrics["dann_gate"]["min_cacc"].append(minc)

        print(f"[Run {run_idx+1}] "
              f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
              f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

    # ----- Aggregate mean/std and save -----
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
