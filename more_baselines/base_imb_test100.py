# more_baselines/base_imb_test100.py

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random, os, json, sys

# Ensure project root is importable if running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_def_test100.model_def import BigCNN
from train_eval_test100 import train_model, evaluate_model
from more_baselines.base_train_eval_test100 import train_lora, train_dann_gate

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# --------------------------
# Imbalanced CIFAR-100 split (pretrain + raw)
# --------------------------
def load_data_split_imb_cifar100(
    seed,
    majority_classes=list(range(10)),
    majority_count=400,
    minority_count=100,
    target_total=10000,
    raw_size=4000,
):
    """
    Build:
      - pretrain_subset: imbalanced CIFAR-100 subset (target_total samples).
      - raw_set        : uniformly sampled raw subset (raw_size samples).
      - testset        : CIFAR-100 test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,  download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Group indices by class
    num_classes = 100
    indices_by_class = {c: [] for c in range(num_classes)}
    for idx, (_, y) in enumerate(trainset):
        indices_by_class[y].append(idx)

    rng = np.random.RandomState(seed)
    for c in range(num_classes):
        rng.shuffle(indices_by_class[c])

    pretrain_indices = []
    for c in range(num_classes):
        cap = majority_count if c in majority_classes else minority_count
        take = min(len(indices_by_class[c]), cap)
        pretrain_indices.extend(indices_by_class[c][:take])

    # Trim/pad to target_total
    pretrain_indices = pretrain_indices[:target_total]
    pretrain_subset = Subset(trainset, pretrain_indices)

    # Raw set (uniform random sample)
    all_idx = np.arange(len(trainset))
    rng.shuffle(all_idx)
    raw_indices = all_idx[:raw_size]
    raw_set = Subset(trainset, raw_indices)

    return pretrain_subset, raw_set, testset


# --------------------------
# Main experiment: LoRA + DANN-Gate
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_epochs = 60
    other_epochs    = 30
    num_runs        = 5
    batch_size      = 64

    save_path  = "./results_test100_base/imbalance_cifar100.json"
    model_ckpt = "./model_test100/base_imbalance_cifar100.pt"

    # Build fixed imbalanced pretrain set and test loader
    pretrain_dataset, raw_set_seed42, testset = load_data_split_imb_cifar100(seed=42)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader     = DataLoader(testset,       batch_size=batch_size, shuffle=False, num_workers=2)

    print("\n=== Pretraining External Model (BigCNN) on Imbalanced CIFAR-100 ===")
    if os.path.exists(model_ckpt):
        external_model = torch.load(model_ckpt).to(device)
        print("Loaded external model from:", model_ckpt)
    else:
        external_model = BigCNN().to(device)
        train_model(external_model, pretrain_loader, pretrain_epochs, device)
        torch.save(external_model, model_ckpt)
        print("Trained and saved external model to:", model_ckpt)

    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Model Evaluation: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | "
          f"F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    metrics = {
        "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    # Vary raw subset across runs
    for run_idx in range(num_runs):
        run_seed = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={run_seed} ===")
        _, raw_set_run, _ = load_data_split_imb_cifar100(seed=run_seed)
        raw_loader = DataLoader(raw_set_run, batch_size=batch_size, shuffle=True, num_workers=2)

        print("Training LoRA...")
        acc, auc, f1, minc = train_lora(external_model, raw_loader, test_loader, device, epochs=other_epochs)
        metrics["lora"]["acc"].append(acc)
        metrics["lora"]["auc"].append(auc)
        metrics["lora"]["f1"].append(f1)
        metrics["lora"]["min_cacc"].append(minc)

        print("Training DANN-Gate...")
        acc, auc, f1, minc = train_dann_gate(external_model, raw_loader, test_loader, device, epochs=other_epochs)
        metrics["dann_gate"]["acc"].append(acc)
        metrics["dann_gate"]["auc"].append(auc)
        metrics["dann_gate"]["f1"].append(f1)
        metrics["dann_gate"]["min_cacc"].append(minc)

        print(f"[Run {run_idx+1}] "
              f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
              f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

    # Aggregate and save
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
