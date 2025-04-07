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

from model_def import CNN, EnhancedCNN, BaselineAdapter, BigCNN
from train_eval import train_model, train_linear_prob, train_enhanced_model, train_distillation, evaluate_model

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------
# 1. Custom PairedLabelCorruptedDataset
# --------------------------
class PairedLabelCorruptedDataset(Dataset):
    """
    A dataset wrapper that applies adversarial label flipping for paired classes
    and adds white noise to the images.
    
    For CIFAR-10 the paired classes are:
      - Cat (3) <-> Dog (5)
      - Deer (4) <-> Horse (7)
      - Automobile (1) <-> Truck (9)
      - Airplane (0) <-> Ship (8)
      
    With probability `p_flip` (default 0.5), samples belonging to a paired class
    have their label flipped to the corresponding pair. Other classes remain unchanged.
    
    Additionally, white Gaussian noise is added to each image with standard deviation `noise_std`.
    """
    def __init__(self, dataset, p_flip=0.5, noise_std=0.5, seed=42):
        self.dataset = dataset
        self.p_flip = p_flip
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self.paired_mapping = {
            3: 5,  # cat -> dog
            5: 3,  # dog -> cat
            4: 7,  # deer -> horse
            7: 4,  # horse -> deer
            1: 9,  # automobile -> truck
            9: 1,  # truck -> automobile
            0: 8,  # airplane -> ship
            8: 0   # ship -> airplane
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if label in self.paired_mapping and self.rng.random() < self.p_flip:
            label = self.paired_mapping[label]
        noise = torch.randn(image.size()) * self.noise_std
        noisy_image = image + noise
        return noisy_image, label

# --------------------------
# 2. Data Loading & Splitting Functions
# --------------------------
def load_and_split_data(seed_for_split, use_adversarial=False, p_flip=0.5, noise_std=0.1):
    """
    Loads CIFAR-10 data and splits the training set into:
      - raw_set: uncorrupted samples (size=4000)
      - augment_set: pretraining samples (size=10000)
    
    If use_adversarial is True, the augment_set is wrapped with PairedLabelCorruptedDataset,
    which applies paired label flipping with probability p_flip and adds white noise with std noise_std.
    Data is always downloaded under './data'.
    """
    rand_gen = random.Random(seed_for_split)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
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
        augment_set = PairedLabelCorruptedDataset(augment_set, p_flip=p_flip, noise_std=noise_std, seed=seed_for_split)
    
    return raw_set, augment_set, testset

# --------------------------
# 5. Main Experiment Loop
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./results/adversarial.json"
    num_epochs = 30
    pretrain_epochs = 60
    num_runs = 5

    # Adversarial parameters.
    p_flip = 0.5    # 50% chance to flip paired class labels
    noise_std = 0.1 # Noise standard deviation

    # Load test set once (common to all runs)
    _, _, testset = load_and_split_data(seed_for_split=42)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    # ---------------------------
    # Pretrain external model (BigCNN) on adversarial pretraining data
    # ---------------------------
    print("\n=== Training External Model (BigCNN) on Adversarial Pretraining Data ===")
    # Set use_adversarial=True to apply paired label flipping and noise.
    _, augment_set_ext, _ = load_and_split_data(seed_for_split=42, use_adversarial=True, p_flip=p_flip, noise_std=noise_std)
    augment_loader_ext = DataLoader(augment_set_ext, batch_size=32, shuffle=True)
    external_model = BigCNN().to(device)
    train_model(external_model, augment_loader_ext, pretrain_epochs, device)
    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Model (Pretrained) Evaluation: Acc={ext_acc:.2f}%, AUC={ext_auc:.4f}, F1={ext_f1:.4f}, MinCAcc={ext_minc:.2f}%")
    
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
        seed_for_split = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_for_split} ===")
        # Load raw_set (uncorrupted) for student training.
        raw_set, _, _ = load_and_split_data(seed_for_split)
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
        acc_ec, auc_ec, f1_ec, min_cacc_ec = evaluate_model(enhanced_concat_model, test_loader, device, enhanced=True, external_model=external_model)
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
