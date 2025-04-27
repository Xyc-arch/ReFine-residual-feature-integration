import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
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
from train_eval_test100 import train_model, train_linear_prob, train_enhanced_model, train_distillation, evaluate_model

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------
# 1. Custom PairedLabelCorruptedDataset for CIFAR-100 with confusion-based pairs
# --------------------------
class PairedLabelCorruptedDataset(Dataset):
    """
    Applies paired label flipping and adds white noise to the images based on a custom confusion mapping.
    """
    def __init__(self, dataset, p_flip=0.5, noise_std=0.5, seed=42, custom_mapping=None):
        self.dataset = dataset
        self.p_flip = p_flip
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        if custom_mapping is None:
            raise ValueError("For CIFAR-100 please supply a custom mapping of confusing class pairs.")
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
# 2. Define custom confusion mapping for CIFAR-100
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
# 3. Data Loading & Splitting for CIFAR-100
# --------------------------
def load_and_split_data(seed_for_split, use_adversarial=False, p_flip=0.5, noise_std=0.2, custom_mapping=None):
    rand_gen = random.Random(seed_for_split)
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

    indices = list(range(len(trainset)))
    rand_gen.shuffle(indices)
    raw_set = Subset(trainset, indices[:4000])
    augment_set = Subset(trainset, indices[4000:14000])

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
# 4. Main Experiment Loop for CIFAR-100 Transformer
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./results_test100/adversarial_tf_cifar100_confusion.json"
    num_epochs = 30
    pretrain_epochs = 60
    num_runs = 5

    p_flip = 0.5
    noise_std = 0.2

    # Load test set once
    _, _, testset = load_and_split_data(42)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    # Pretrain external BigTransformer
    print("\n=== Pretraining BigTransformer on adversarial CIFAR-100 ===")
    adv_tf_save_path = "./model_test100/mismatch_tf.pt"
    if os.path.exists(adv_tf_save_path):
        external_model = torch.load(adv_tf_save_path).to(device)
        print("Loaded external transformer from:", adv_tf_save_path)
    else:
        _, augment_ext, _ = load_and_split_data(
            42, use_adversarial=True, p_flip=p_flip, noise_std=noise_std,
            custom_mapping=custom_confusion_mapping
        )
        aug_loader_ext = DataLoader(augment_ext, batch_size=32, shuffle=True)
        external_model = BigTransformer().to(device)
        train_model(external_model, aug_loader_ext, pretrain_epochs, device)
        torch.save(external_model, adv_tf_save_path)
        print("Saved pretrained transformer to:", adv_tf_save_path)

    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Transformer Eval: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    # Metrics container
    metrics = {m: {"acc": [], "auc": [], "f1": [], "min_cacc": []}
               for m in ["baseline", "linear_prob", "enhanced_concat", "baseline_adapter", "distillation"]}

    # Repeat experiments
    for run_idx in range(num_runs):
        seed = 42 + run_idx
        print(f"\n--- Run {run_idx+1}/{num_runs}, seed={seed} ---")
        raw_set, _, _ = load_and_split_data(seed)
        raw_loader = DataLoader(raw_set, batch_size=32, shuffle=True)

        # 1. Baseline Transformer
        print("Training baseline TransformerClassifier...")
        baseline = TransformerClassifier().to(device)
        train_model(baseline, raw_loader, num_epochs, device)
        acc, auc, f1, mc = evaluate_model(baseline, test_loader, device)
        metrics["baseline"]["acc"].append(acc)
        metrics["baseline"]["auc"].append(auc)
        metrics["baseline"]["f1"].append(f1)
        metrics["baseline"]["min_cacc"].append(mc)

        # 2. Linear Probe
        print("Training linear probe on classifier head...")
        lp_model = copy.deepcopy(external_model)
        for p in lp_model.parameters():
            p.requires_grad = False
        for p in lp_model.classifier.parameters():
            p.requires_grad = True
        train_linear_prob(lp_model, raw_loader, num_epochs, device)
        acc, auc, f1, mc = evaluate_model(lp_model, test_loader, device)
        metrics["linear_prob"]["acc"].append(acc)
        metrics["linear_prob"]["auc"].append(auc)
        metrics["linear_prob"]["f1"].append(f1)
        metrics["linear_prob"]["min_cacc"].append(mc)

        # 3. Enhanced (Concat)
        print("Training EnhancedTransformer (concat features)...")
        enh = EnhancedTransformer().to(device)
        train_enhanced_model(enh, raw_loader, external_model, num_epochs, device)
        acc, auc, f1, mc = evaluate_model(enh, test_loader, device, enhanced=True, external_model=external_model)
        metrics["enhanced_concat"]["acc"].append(acc)
        metrics["enhanced_concat"]["auc"].append(auc)
        metrics["enhanced_concat"]["f1"].append(f1)
        metrics["enhanced_concat"]["min_cacc"].append(mc)

        # 4. Baseline Adapter
        print("Training BaselineAdapterTransformer...")
        ba = BaselineAdapterTransformer(copy.deepcopy(external_model)).to(device)
        train_model(ba, raw_loader, num_epochs, device)
        acc, auc, f1, mc = evaluate_model(ba, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc)
        metrics["baseline_adapter"]["auc"].append(auc)
        metrics["baseline_adapter"]["f1"].append(f1)
        metrics["baseline_adapter"]["min_cacc"].append(mc)

        # 5. Knowledge Distillation
        print("Training distillation student TransformerClassifier...")
        student = TransformerClassifier().to(device)
        train_distillation(student, external_model, raw_loader, num_epochs, device,
                           temperature=2.0, alpha=0.5)
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

    # Aggregate results
    final = {}
    for name, vals in metrics.items():
        arr_acc = np.array(vals["acc"])
        arr_auc = np.array(vals["auc"])
        arr_f1 = np.array(vals["f1"])
        arr_mc = np.array(vals["min_cacc"])
        final[name] = {
            "acc_mean": float(arr_acc.mean()), "acc_std": float(arr_acc.std()),
            "auc_mean": float(arr_auc.mean()), "auc_std": float(arr_auc.std()),
            "f1_mean": float(arr_f1.mean()), "f1_std": float(arr_f1.std()),
            "min_cacc_mean": float(arr_mc.mean()), "min_cacc_std": float(arr_mc.std())
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final, fp, indent=2)

    print(f"\nAll done. Final mean/std results saved to: {save_path}")

if __name__ == "__main__":
    main()
