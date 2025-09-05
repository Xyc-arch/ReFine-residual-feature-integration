#!/usr/bin/env python3
# more_baselines/base_adversarial_tf_test100.py

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

# --------------------------
# 1) Paired-label corruption + noise wrapper (CIFAR-100)
# --------------------------
class PairedLabelCorruptedDataset(Dataset):
    """
    Applies paired label flipping and adds white noise to the images
    using a custom confusion mapping (dict: class_i -> paired_class_j).
    """
    def __init__(self, dataset, p_flip=0.5, noise_std=0.5, seed=42, custom_mapping=None):
        self.dataset = dataset
        self.p_flip = p_flip
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        if custom_mapping is None:
            raise ValueError("Please supply a custom mapping of confusing class pairs for CIFAR-100.")
        self.paired_mapping = custom_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if label in self.paired_mapping and self.rng.random() < self.p_flip:
            label = self.paired_mapping[label]
        noise = torch.randn_like(image) * self.noise_std
        noisy_image = image + noise
        return noisy_image, label

# --------------------------
# 2) Confusion mapping (CIFAR-100)
# --------------------------
custom_confusion_mapping = {
    0: 52,   52: 0,    # apple  ↔ orange
    1: 31,   31: 1,    # aquarium_fish  ↔ flatfish
    2: 99,   99: 2,    # baby  ↔ bat
    3: 96,   96: 3,    # bear  ↔ wolf
    4: 54,   54: 4,    # beaver  ↔ otter
    5: 24,   24: 5,    # bed  ↔ couch
    6: 7,    7: 6,     # bee  ↔ beetle
    8: 47,   47: 8,    # bicycle  ↔ motorcycle
    9: 27,   27: 9,    # bottle  ↔ cup
    10: 60,  60: 10,   # bowl  ↔ plate
    11: 34,  34: 11,   # boy  ↔ girl
    12: 16,  16: 12,   # bridge  ↔ can
    14: 18,  18: 14,   # butterfly  ↔ caterpillar
    17: 36,  36: 17,   # castle  ↔ house
    19: 37,  37: 19,   # cattle  ↔ kangaroo
    20: 45,  45: 20,   # chimpanzee  ↔ man
    23: 78,  78: 23,   # cockroach  ↔ spider
    25: 44,  44: 25,   # crab  ↔ lobster
    26: 28,  28: 26,   # dinosaur  ↔ crocodile
    29: 94,  94: 29,   # dolphin  ↔ whale
    30: 15,  15: 30,   # elephant  ↔ camel
    32: 48,  48: 32,   # forest  ↔ mountain
    33: 65,  65: 33,   # fox  ↔ raccoon
    35: 49,  49: 35,   # hamster  ↔ mouse
    38: 39,  39: 38,   # keyboard  ↔ lamp
    40: 88,  88: 40,   # lawn_mower  ↔ tractor
    42: 87,  87: 42,   # lion  ↔ tiger
    41: 97,  97: 41,   # leopard  ↔ woman
    43: 77,  77: 43,   # lizard  ↔ snake
    46: 51,  51: 46,   # maple_tree  ↔ oak_tree
    53: 61,  61: 53,   # orchid  ↔ poppy
    55: 56,  56: 55,   # palm_tree  ↔ pear
    57: 13,  13: 57,   # pickup_truck  ↔ bus
    58: 59,  59: 58,   # pine_tree  ↔ plain
    62: 63,  63: 62,   # porcupine  ↔ possum
    64: 79,  79: 64,   # rabbit  ↔ squirrel
    66: 72,  72: 66,   # ray  ↔ shark
    67: 68,  68: 67,   # road  ↔ rocket
    69: 91,  91: 69,   # rose  ↔ tulip
    70: 71,  71: 70,   # sea  ↔ seal
    73: 74,  74: 73,   # shrew  ↔ skunk
    75: 80,  80: 75,   # skyscraper  ↔ streetcar
    76: 98,  98: 76,   # snail  ↔ worm
    81: 82,  82: 81,   # sunflower  ↔ sweet_pepper
    83: 84,  84: 83,   # table  ↔ tank
    89: 90,  90: 89,   # train  ↔ trout
    93: 95,  95: 93    # wardrobe  ↔ willow_tree
}

# --------------------------
# 3) Data loading & split for CIFAR-100
# --------------------------
def load_and_split_data(seed_for_split, use_adversarial=False, p_flip=0.5, noise_std=0.2, custom_mapping=None):
    rng = random.Random(seed_for_split)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )

    indices = list(range(len(trainset)))
    rng.shuffle(indices)
    raw_set     = Subset(trainset, indices[:4000])
    augment_set = Subset(trainset, indices[4000:14000])

    if use_adversarial:
        augment_set = PairedLabelCorruptedDataset(
            augment_set, p_flip=p_flip, noise_std=noise_std,
            seed=seed_for_split, custom_mapping=custom_mapping
        )
    return raw_set, augment_set, testset

# --------------------------
# 4) Main (LoRA + DANN-Gate baselines only)
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path       = "./results_test100_base/adversarial_tf_cifar100_confusion.json"
    model_save_path = "./model_test100/adversarial_tf_cifar100.pt"

    num_epochs      = 30
    pretrain_epochs = 60
    num_runs        = 5

    # adversarial corruption settings
    p_flip    = 0.5
    noise_std = 0.8

    # Load test set once
    _, _, testset = load_and_split_data(42)
    test_loader   = DataLoader(testset, batch_size=32, shuffle=False)

    # Pretrain external BigTransformer on adversarial augment set
    print("\n=== Pretraining BigTransformer on adversarial CIFAR-100 (confusion pairs + noise) ===")
    if os.path.exists(model_save_path):
        external_model = torch.load(model_save_path).to(device)
        print("Loaded external transformer from:", model_save_path)
    else:
        _, augment_ext, _ = load_and_split_data(
            42, use_adversarial=True, p_flip=p_flip, noise_std=noise_std,
            custom_mapping=custom_confusion_mapping
        )
        aug_loader_ext = DataLoader(augment_ext, batch_size=32, shuffle=True)
        external_model = BigTransformer().to(device)
        train_model(external_model, aug_loader_ext, pretrain_epochs, device)
        torch.save(external_model, model_save_path)
        print("Saved pretrained transformer to:", model_save_path)

    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Transformer Eval: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    # Metrics (base variants only)
    metrics = {
        "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    # Repeat experiments with different raw-set splits
    for run_idx in range(num_runs):
        seed = 42 + run_idx
        print(f"\n--- Run {run_idx+1}/{num_runs}, seed={seed} ---")
        raw_set, _, _ = load_and_split_data(seed)
        raw_loader = DataLoader(raw_set, batch_size=32, shuffle=True)

        print("Training LoRA (Transformer head)...")
        acc, auc, f1, mc = train_lora_tf(external_model, raw_loader, test_loader, device, epochs=num_epochs)
        metrics["lora"]["acc"].append(acc)
        metrics["lora"]["auc"].append(auc)
        metrics["lora"]["f1"].append(f1)
        metrics["lora"]["min_cacc"].append(mc)

        print("Training DANN-Gate (Transformer head)...")
        acc, auc, f1, mc = train_dann_gate_tf(external_model, raw_loader, test_loader, device, epochs=num_epochs)
        metrics["dann_gate"]["acc"].append(acc)
        metrics["dann_gate"]["auc"].append(auc)
        metrics["dann_gate"]["f1"].append(f1)
        metrics["dann_gate"]["min_cacc"].append(mc)

        print(f"[Run {run_idx+1}] "
              f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
              f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

    # Aggregate and save
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
