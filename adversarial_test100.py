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

from model_def_test100.model_def import CNN, EnhancedCNN, BaselineAdapter, BigCNN
from train_eval import train_model, train_linear_prob, train_enhanced_model, train_distillation, evaluate_model

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------
# 1. Custom PairedLabelCorruptedDataset for CIFAR-100 with confusion-based pairs
# --------------------------
class PairedLabelCorruptedDataset(Dataset):
    """
    A dataset wrapper that applies paired label flipping and adds white noise to the images.

    Instead of pairing classes sequentially, this version uses a custom mapping based on classes
    that are known to be confused. For example, if samples of class A (e.g. "dolphin") are often mis‐
    classified as class B (e.g. "whale"), then with probability `p_flip` the label A is flipped to B.

    The custom mapping should be provided as a dictionary where each key-value pair indicates that
    the label should flip to its paired value and vice versa.
    
    Note: The custom mapping below assumes the CIFAR-100 canonical ordering. For example:
      - "dolphin" is assumed to have index 30 and "whale" index 93.
      - "shark" (index 73) and "ray" (index 67) are paired.
      - "rose" (index 70) and "tulip" (index 90) are paired.
      - "bottle" (index 9) and "cup" (index 28) are paired.
      - "apple" (index 0) and "orange" (index 53) are paired.
      - "telephone" (index 84) and "television" (index 85) are paired.
      - "bed" (index 5) and "couch" (index 25) are paired.
      - "caterpillar" (index 18) and "butterfly" (index 14) are paired.
      - "lion" (index 43) and "tiger" (index 86) are paired.
      - "castle" (index 17) and "house" (index 37) are paired.
      - "chimpanzee" (index 21) and "man" (index 46) are paired.
      - "fox" (index 34) and "raccoon" (index 66) are paired.
      - "crab" (index 26) and "lobster" (index 45) are paired.
      - "boy" (index 11) and "girl" (index 35) are paired.
      - "lizard" (index 44) and "snake" (index 76) are paired.
      - "hamster" (index 36) and "mouse" (index 50) are paired.
      - "maple_tree" (index 47) and "oak_tree" (index 52) are paired.
      - "bicycle" (index 8) and "motorcycle" (index 48) are paired.
      - "bus" (index 13) and "pickup_truck" (index 58) are paired.
    
    With probability `p_flip`, if the sample’s label is in this mapping, its label is replaced by the paired label.
    Additionally, white Gaussian noise (std: noise_std) is added to the image.
    """
    def __init__(self, dataset, p_flip=0.5, noise_std=0.5, seed=42, num_classes=100, custom_mapping=None):
        self.dataset = dataset
        self.p_flip = p_flip
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self.num_classes = num_classes

        # Use the supplied custom_mapping if provided; otherwise, raise an error.
        if custom_mapping is None:
            raise ValueError("For CIFAR-100 please supply a custom mapping of confusing class pairs.")
        else:
            self.paired_mapping = custom_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Flip the label if it is in our custom mapping and the random chance hits.
        if label in self.paired_mapping and self.rng.random() < self.p_flip:
            label = self.paired_mapping[label]
        noise = torch.randn(image.size()) * self.noise_std
        noisy_image = image + noise
        return noisy_image, label

# --------------------------
# 2. Define a custom confusion mapping for CIFAR-100
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
# 3. Data Loading & Splitting Functions for CIFAR-100
# --------------------------
def load_and_split_data(seed_for_split, use_adversarial=False, p_flip=0.5, noise_std=0.1, custom_mapping=None):
    """
    Loads CIFAR-100 data and splits the training set into:
      - raw_set: uncorrupted samples (size=4000)
      - augment_set: pretraining samples (size=10000)
    
    When use_adversarial is True, the augment_set is wrapped with PairedLabelCorruptedDataset,
    which applies paired label flipping with probability p_flip and adds white noise (std: noise_std).
    """
    rand_gen = random.Random(seed_for_split)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    total_size = len(trainset)
    raw_size = 4000
    augment_size = 10000
    indices = list(range(total_size))
    rand_gen.shuffle(indices)
    
    raw_indices = indices[:raw_size]
    augment_indices = indices[raw_size:raw_size+augment_size]
    
    raw_set = Subset(trainset, raw_indices)
    augment_set = Subset(trainset, augment_indices)
    
    if use_adversarial:
        # For CIFAR-100, we supply the custom confusion mapping.
        augment_set = PairedLabelCorruptedDataset(
            augment_set,
            p_flip=p_flip,
            noise_std=noise_std,
            seed=seed_for_split,
            num_classes=100,
            custom_mapping=custom_mapping
        )
    
    return raw_set, augment_set, testset

# --------------------------
# 4. Main Experiment Loop for CIFAR-100 (using confusion-based paired corruption)
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./results_test100/adversarial_cifar100_confusion.json"
    num_epochs = 30
    pretrain_epochs = 60
    num_runs = 5

    # Adversarial parameters.
    p_flip = 0.5    # 50% chance to flip paired class labels
    noise_std = 0.1 # Noise standard deviation

    # Load the test set once (common to all runs)
    _, _, testset = load_and_split_data(seed_for_split=42)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    # ---------------------------
    # Pretrain external model (BigCNN) on adversarial pretraining data (using confusion pairs)
    # ---------------------------
    print("\n=== Training External Model (BigCNN) on Adversarial Pretraining Data (CIFAR-100) ===")
    mismatch_save_path = "./model_test100/mismatch.pt"
    if os.path.exists(mismatch_save_path):
        external_model = torch.load(mismatch_save_path).to(device)
        print("Loaded external model from:", mismatch_save_path)
    else:
        _, augment_set_ext, _ = load_and_split_data(
            seed_for_split=42,
            use_adversarial=True,
            p_flip=p_flip,
            noise_std=noise_std,
            custom_mapping=custom_confusion_mapping
        )
        augment_loader_ext = DataLoader(augment_set_ext, batch_size=32, shuffle=True)
        external_model = BigCNN().to(device)
        train_model(external_model, augment_loader_ext, pretrain_epochs, device)
        torch.save(external_model, mismatch_save_path)
        print("Trained and saved external model to:", mismatch_save_path)
    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Model (Pretrained) Evaluation: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    
    # Prepare metrics containers for different training methods.
    metrics = {
        "baseline": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "linear_prob": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "enhanced_concat": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "baseline_adapter": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "distillation": {"acc": [], "auc": [], "f1": [], "min_cacc": []}
    }
    
    # Run experiments over different raw_set splits.
    for run_idx in range(num_runs):
        seed_for_split_run = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_for_split_run} ===")
        # Load raw_set (uncorrupted) for student training.
        raw_set, _, _ = load_and_split_data(seed_for_split_run)
        raw_loader = DataLoader(raw_set, batch_size=32, shuffle=True)
        
        # 1. Baseline: Train CNN on raw_set only.
        print("Training baseline model (raw only)...")
        baseline_model = CNN().to(device)
        train_model(baseline_model, raw_loader, num_epochs, device)
        acc_b, auc_b, f1_b, min_cacc_b = evaluate_model(baseline_model, test_loader, device)
        metrics["baseline"]["acc"].append(acc_b)
        metrics["baseline"]["auc"].append(auc_b)
        metrics["baseline"]["f1"].append(f1_b)
        metrics["baseline"]["min_cacc"].append(min_cacc_b)
        
        # 2. Linear Probe: Fine-tune only the final layer of external_model on raw_set.
        print("Training linear probe model (external fine-tuned on raw_set)...")
        linear_model = copy.deepcopy(external_model)
        for param in linear_model.parameters():
            param.requires_grad = False
        for param in linear_model.fc_layers[-1].parameters():
            param.requires_grad = True
        train_linear_prob(linear_model, raw_loader, num_epochs, device)
        acc_lp, auc_lp, f1_lp, min_cacc_lp = evaluate_model(linear_model, test_loader, device)
        metrics["linear_prob"]["acc"].append(acc_lp)
        metrics["linear_prob"]["auc"].append(auc_lp)
        metrics["linear_prob"]["f1"].append(f1_lp)
        metrics["linear_prob"]["min_cacc"].append(min_cacc_lp)
        
        # 3. Enhanced (Concatenation): Train EnhancedCNN on raw_set using features from external_model.
        print("Training enhanced model (concatenation)...")
        enhanced_concat_model = EnhancedCNN().to(device)
        train_enhanced_model(enhanced_concat_model, raw_loader, external_model, num_epochs, device)
        acc_ec, auc_ec, f1_ec, min_cacc_ec = evaluate_model(
            enhanced_concat_model, test_loader, device, enhanced=True, external_model=external_model
        )
        metrics["enhanced_concat"]["acc"].append(acc_ec)
        metrics["enhanced_concat"]["auc"].append(auc_ec)
        metrics["enhanced_concat"]["f1"].append(f1_ec)
        metrics["enhanced_concat"]["min_cacc"].append(min_cacc_ec)
        
        # 4. Baseline Adapter: Freeze external_model and fine-tune adapter + classifier on raw_set.
        print("Training baseline adapter model (external frozen with adapter) on raw_set...")
        baseline_adapter_model = BaselineAdapter(copy.deepcopy(external_model)).to(device)
        train_model(baseline_adapter_model, raw_loader, num_epochs, device)
        acc_ba, auc_ba, f1_ba, min_cacc_ba = evaluate_model(baseline_adapter_model, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc_ba)
        metrics["baseline_adapter"]["auc"].append(auc_ba)
        metrics["baseline_adapter"]["f1"].append(f1_ba)
        metrics["baseline_adapter"]["min_cacc"].append(min_cacc_ba)
        
        # 5. Knowledge Distillation: Train a CNN student using external_model as teacher.
        print("Training knowledge distillation model (CNN student with teacher external)...")
        student_model = CNN().to(device)
        train_distillation(student_model, external_model, raw_loader, num_epochs, device, temperature=2.0, alpha=0.5)
        acc_kd, auc_kd, f1_kd, min_cacc_kd = evaluate_model(student_model, test_loader, device)
        metrics["distillation"]["acc"].append(acc_kd)
        metrics["distillation"]["auc"].append(auc_kd)
        metrics["distillation"]["f1"].append(f1_kd)
        metrics["distillation"]["min_cacc"].append(min_cacc_kd)
        
        print(f"\n[Run {run_idx+1} Results]")
        print(f"Baseline:          Acc={acc_b:.2f}% | AUC={auc_b:.4f} | F1={f1_b:.4f} | MinCAcc={min_cacc_b:.2f}%")
        print(f"Linear Probe:      Acc={acc_lp:.2f}% | AUC={auc_lp:.4f} | F1={f1_lp:.4f} | MinCAcc={min_cacc_lp:.2f}%")
        print(f"Enhanced (Concat): Acc={acc_ec:.2f}% | AUC={auc_ec:.4f} | F1={f1_ec:.4f} | MinCAcc={min_cacc_ec:.2f}%")
        print(f"Baseline Adapter:  Acc={acc_ba:.2f}% | AUC={auc_ba:.4f} | F1={f1_ba:.4f} | MinCAcc={min_cacc_ba:.2f}%")
        print(f"Distillation:      Acc={acc_kd:.2f}% | AUC={auc_kd:.4f} | F1={f1_kd:.4f} | MinCAcc={min_cacc_kd:.2f}%")
    
    final_results = {}
    for method, m_dict in metrics.items():
        acc_arr = np.array(m_dict["acc"])
        auc_arr = np.array(m_dict["auc"])
        f1_arr = np.array(m_dict["f1"])
        minc_arr = np.array(m_dict["min_cacc"])
        final_results[method] = {
            "acc_mean": float(acc_arr.mean()),
            "acc_std": float(acc_arr.std()),
            "auc_mean": float(auc_arr.mean()),
            "auc_std": float(auc_arr.std()),
            "f1_mean": float(f1_arr.mean()),
            "f1_std": float(f1_arr.std()),
            "min_cacc_mean": float(minc_arr.mean()),
            "min_cacc_std": float(minc_arr.std())
        }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final_results, fp, indent=2)
    
    print(f"\nAll done. Final mean/std results saved to: {save_path}")

if __name__ == "__main__":
    main()
