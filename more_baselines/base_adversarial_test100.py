# more_baselines/base_adversarial_test100.py

import torch
from torch.utils.data import DataLoader, Subset, Dataset
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
# 1) PairedLabelCorruptedDataset for CIFAR-100
# --------------------------
class PairedLabelCorruptedDataset(Dataset):
    """
    Wraps a dataset to apply confusion-based paired label flipping (prob p_flip)
    and additive white Gaussian noise (std=noise_std) to the images.
    """
    def __init__(self, dataset, p_flip=0.5, noise_std=0.2, seed=42, custom_mapping=None):
        if custom_mapping is None:
            raise ValueError("Please provide a custom mapping of confusing class pairs for CIFAR-100.")
        self.dataset = dataset
        self.p_flip = p_flip
        self.noise_std = noise_std
        self.paired_mapping = custom_mapping
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if label in self.paired_mapping and self.rng.random() < self.p_flip:
            label = self.paired_mapping[label]
        image = image + torch.randn_like(image) * self.noise_std
        return image, label

# --------------------------
# 2) Confusion mapping (pairs) for CIFAR-100
# --------------------------
custom_confusion_mapping = {
    0: 52,  52: 0,   # apple ↔ orange
    1: 31,  31: 1,   # aquarium_fish ↔ flatfish
    2: 99,  99: 2,   # baby ↔ bat
    3: 96,  96: 3,   # bear ↔ wolf
    4: 54,  54: 4,   # beaver ↔ otter
    5: 24,  24: 5,   # bed ↔ couch
    6: 7,   7: 6,    # bee ↔ beetle
    8: 47,  47: 8,   # bicycle ↔ motorcycle
    9: 27,  27: 9,   # bottle ↔ cup
    10: 60, 60: 10,  # bowl ↔ plate
    11: 34, 34: 11,  # boy ↔ girl
    12: 16, 16: 12,  # bridge ↔ can
    14: 18, 18: 14,  # butterfly ↔ caterpillar
    17: 36, 36: 17,  # castle ↔ house
    19: 37, 37: 19,  # cattle ↔ kangaroo
    20: 45, 45: 20,  # chimpanzee ↔ man
    23: 78, 78: 23,  # cockroach ↔ spider
    25: 44, 44: 25,  # crab ↔ lobster
    26: 28, 28: 26,  # dinosaur ↔ crocodile
    29: 94, 94: 29,  # dolphin ↔ whale
    30: 15, 15: 30,  # elephant ↔ camel
    32: 48, 48: 32,  # forest ↔ mountain
    33: 65, 65: 33,  # fox ↔ raccoon
    35: 49, 49: 35,  # hamster ↔ mouse
    38: 39, 39: 38,  # keyboard ↔ lamp
    40: 88, 88: 40,  # lawn_mower ↔ tractor
    41: 97, 97: 41,  # leopard ↔ woman
    42: 87, 87: 42,  # lion ↔ tiger
    43: 77, 77: 43,  # lizard ↔ snake
    46: 51, 51: 46,  # maple_tree ↔ oak_tree
    53: 61, 61: 53,  # orchid ↔ poppy
    55: 56, 56: 55,  # palm_tree ↔ pear
    57: 13, 13: 57,  # pickup_truck ↔ bus
    58: 59, 59: 58,  # pine_tree ↔ plain
    62: 63, 63: 62,  # porcupine ↔ possum
    64: 79, 79: 64,  # rabbit ↔ squirrel
    66: 72, 72: 66,  # ray ↔ shark
    67: 68, 68: 67,  # road ↔ rocket
    69: 91, 91: 69,  # rose ↔ tulip
    70: 71, 71: 70,  # sea ↔ seal
    73: 74, 74: 73,  # shrew ↔ skunk
    75: 80, 80: 75,  # skyscraper ↔ streetcar
    76: 98, 98: 76,  # snail ↔ worm
    81: 82, 82: 81,  # sunflower ↔ sweet_pepper
    83: 84, 84: 83,  # table ↔ tank
    89: 90, 90: 89,  # train ↔ trout
    93: 95, 95: 93,  # wardrobe ↔ willow_tree
}

# --------------------------
# 3) CIFAR-100 split helper
# --------------------------
def load_and_split_data(seed_for_split, use_adversarial=False, p_flip=0.5, noise_std=0.2, custom_mapping=None):
    """
    Splits CIFAR-100 train into:
      - raw_set: 4,000 clean samples
      - augment_set: 10,000 samples (optionally adversarially corrupted)
    """
    rng = random.Random(seed_for_split)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,  download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    total = len(trainset)
    raw_size = 4000
    augment_size = 10000

    idx = list(range(total))
    rng.shuffle(idx)

    raw_idx     = idx[:raw_size]
    augment_idx = idx[raw_size:raw_size+augment_size]

    raw_set     = Subset(trainset, raw_idx)
    augment_set = Subset(trainset, augment_idx)

    if use_adversarial:
        augment_set = PairedLabelCorruptedDataset(
            augment_set,
            p_flip=p_flip,
            noise_std=noise_std,
            seed=seed_for_split,
            custom_mapping=custom_mapping
        )

    return raw_set, augment_set, testset

# --------------------------
# 4) Main: pretrain on adversarial augment set, run LoRA & DANN-Gate
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_epochs = 60
    other_epochs    = 30
    num_runs        = 5
    batch_size      = 32

    p_flip   = 0.5
    noise_sd = 0.2

    save_path  = "./results_test100_base/adversarial_cifar100_confusion.json"
    model_ckpt = "./model_test100/base_adversarial_cifar100.pt"

    # Test loader (common)
    _, _, testset = load_and_split_data(seed_for_split=42)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Pretrain or load external BigCNN on adversarial augment subset
    print("\n=== Pretraining External Model (BigCNN) on Adversarial CIFAR-100 ===")
    if os.path.exists(model_ckpt):
        external_model = torch.load(model_ckpt).to(device)
        print("Loaded external model from:", model_ckpt)
    else:
        _, augment_set_ext, _ = load_and_split_data(
            seed_for_split=42,
            use_adversarial=True,
            p_flip=p_flip,
            noise_std=noise_sd,
            custom_mapping=custom_confusion_mapping
        )
        augment_loader = DataLoader(augment_set_ext, batch_size=batch_size, shuffle=True)
        external_model = BigCNN().to(device)
        train_model(external_model, augment_loader, pretrain_epochs, device)
        torch.save(external_model, model_ckpt)
        print("Trained and saved external model to:", model_ckpt)

    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Model (Pretrained) Evaluation: "
          f"Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    metrics = {
        "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    # Runs over different raw splits
    for run_idx in range(num_runs):
        run_seed = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={run_seed} ===")
        raw_set, _, _ = load_and_split_data(seed_for_split=run_seed)
        raw_loader = DataLoader(raw_set, batch_size=batch_size, shuffle=True)

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
