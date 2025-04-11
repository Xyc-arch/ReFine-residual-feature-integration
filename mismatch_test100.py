import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, Subset
import random
import json
import os
import copy
from train_eval_test100 import *

# For reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Import models from the new model_def10.py file.
from model_def_test100.model_def10 import CNN, EnhancedCNN, BaselineAdapter, BigCNN

# --------------------------
# Data Loading for CIFAR-100 (Student)
# --------------------------
def load_student_data(seed_for_split, raw_size=2000, augment_size=4000):
    """
    Splits CIFAR-100 training data (for student) into:
      - raw_set: clean samples (size=raw_size)
      - augment_set: samples with random (corrupted) labels (size=augment_size)
    All data is downloaded under ./data.
    """
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
    total_size = len(trainset)
    indices = list(range(total_size))
    rand_gen.shuffle(indices)
    raw_indices = indices[:raw_size]

    # We define a dataset for the raw (clean) training set.
    raw_set = Subset(trainset, raw_indices)
    return raw_set, testset

# --------------------------
# Data Loading for CIFAR-10 (Teacher)
# --------------------------
def load_teacher_data(pretrain_size=2000, seed=42):
    """
    Loads CIFAR-10 training data and creates a subset of size pretrain_size.
    All data is downloaded under ./data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    teacher_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    teacher_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    total_size = len(teacher_train)
    indices = list(range(total_size))
    random.Random(seed).shuffle(indices)
    subset_indices = indices[:pretrain_size]
    teacher_train_subset = Subset(teacher_train, subset_indices)
    return teacher_train_subset, teacher_test

# --------------------------
# Main Experiment Loop
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./results_test100/mismatch_reverse.json"
    num_epochs = 30
    num_epoch_teacher = 60
    num_runs = 5

    # ---------------------------
    # Train Teacher on CIFAR-10 Subset
    # ---------------------------
    print("\n=== Training Teacher (BigCNN) on CIFAR-10 Subset ===")
    teacher_model_save_path = "./model_test100/mismatch_reverse.pt"
    if os.path.exists(teacher_model_save_path):
        teacher_model = torch.load(teacher_model_save_path).to(device)
        print("Loaded teacher model from:", teacher_model_save_path)
    else:
        pretrain_size = 10000
        teacher_train_subset, teacher_test = load_teacher_data(pretrain_size=pretrain_size, seed=42)
        teacher_loader = DataLoader(teacher_train_subset, batch_size=32, shuffle=True, num_workers=2)
        teacher_model = BigCNN().to(device)
        train_model(teacher_model, teacher_loader, num_epoch_teacher, device)
        torch.save(teacher_model, teacher_model_save_path)
        print("Trained and saved teacher model to:", teacher_model_save_path)
    # Teacher is trained to classify 10 classes. We will use teacher_model.get_features() to extract features.
    
    # ---------------------------
    # Load CIFAR-100 Testset for Student Evaluation
    # ---------------------------
    _, testset = load_student_data(seed_for_split=42, raw_size=4000)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    # Prepare metrics containers for four experiments:
    # baseline, linear probe, enhanced (concatenation), and adapter.
    metrics = {
        "baseline": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "linear_prob": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "enhanced_concat": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "baseline_adapter": {"acc": [], "auc": [], "f1": [], "min_cacc": []}
    }
    
    # Run experiments over different raw_set splits (CIFAR-100)
    for run_idx in range(num_runs):
        seed_for_split = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_for_split} ===")
        # Load student raw set (using CIFAR-100)
        raw_set, _ = load_student_data(seed_for_split, raw_size=4000)
        raw_loader = DataLoader(raw_set, batch_size=32, shuffle=True, num_workers=2)
        
        # 1. Baseline: Train CNN (student CIFAR-100 model) on raw_set only.
        print("Training baseline model (raw only)...")
        baseline_model = CNN().to(device)
        train_model(baseline_model, raw_loader, num_epochs, device)
        acc_b, auc_b, f1_b, min_cacc_b = evaluate_model(baseline_model, test_loader, device)
        metrics["baseline"]["acc"].append(acc_b)
        metrics["baseline"]["auc"].append(auc_b)
        metrics["baseline"]["f1"].append(f1_b)
        metrics["baseline"]["min_cacc"].append(min_cacc_b)
        
        # 2. Linear Probe: Replace teacher's head with a new 100-class head and fine-tune on raw_set.
        print("Training linear probe model (teacher fine-tuned on raw_set)...")
        linear_model = copy.deepcopy(teacher_model)
        for param in linear_model.parameters():
            param.requires_grad = False
        # Replace teacher's final classification layer (originally 10 classes) with new layer for 100 classes.
        linear_model.fc_layers[-1] = nn.Linear(2560, 100)
        for param in linear_model.fc_layers[-1].parameters():
            param.requires_grad = True
        linear_model = linear_model.to(device)
        train_linear_prob(linear_model, raw_loader, num_epochs, device)
        acc_lp, auc_lp, f1_lp, min_cacc_lp = evaluate_model(linear_model, test_loader, device)
        metrics["linear_prob"]["acc"].append(acc_lp)
        metrics["linear_prob"]["auc"].append(auc_lp)
        metrics["linear_prob"]["f1"].append(f1_lp)
        metrics["linear_prob"]["min_cacc"].append(min_cacc_lp)
        
        # 3. Enhanced (Concatenation): Train EnhancedCNN on raw_set using teacher features.
        print("Training enhanced model (concatenation)...")
        enhanced_concat_model = EnhancedCNN().to(device)
        train_enhanced_model(enhanced_concat_model, raw_loader, teacher_model, num_epochs, device)
        acc_ec, auc_ec, f1_ec, min_cacc_ec = evaluate_model(
            enhanced_concat_model, test_loader, device, enhanced=True, external_model=teacher_model
        )
        metrics["enhanced_concat"]["acc"].append(acc_ec)
        metrics["enhanced_concat"]["auc"].append(auc_ec)
        metrics["enhanced_concat"]["f1"].append(f1_ec)
        metrics["enhanced_concat"]["min_cacc"].append(min_cacc_ec)
        
        # 4. Baseline Adapter: Use teacher's frozen feature extractor with an adapter and a new head.
        print("Training baseline adapter model (teacher frozen with adapter) on raw_set...")
        adapter_model = BaselineAdapter(copy.deepcopy(teacher_model)).to(device)
        train_model(adapter_model, raw_loader, num_epochs, device)
        acc_ba, auc_ba, f1_ba, min_cacc_ba = evaluate_model(adapter_model, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc_ba)
        metrics["baseline_adapter"]["auc"].append(auc_ba)
        metrics["baseline_adapter"]["f1"].append(f1_ba)
        metrics["baseline_adapter"]["min_cacc"].append(min_cacc_ba)
        
        print(f"\n[Run {run_idx+1} Results]")
        print(f"Baseline:         Acc={acc_b:.2f}% | AUC={auc_b:.4f} | F1={f1_b:.4f} | MinCAcc={min_cacc_b:.2f}%")
        print(f"Linear Probe:     Acc={acc_lp:.2f}% | AUC={auc_lp:.4f} | F1={f1_lp:.4f} | MinCAcc={min_cacc_lp:.2f}%")
        print(f"Enhanced (Concat):Acc={acc_ec:.2f}% | AUC={auc_ec:.4f} | F1={f1_ec:.4f} | MinCAcc={min_cacc_ec:.2f}%")
        print(f"Baseline Adapter: Acc={acc_ba:.2f}% | AUC={auc_ba:.4f} | F1={f1_ba:.4f} | MinCAcc={min_cacc_ba:.2f}%")
    
    # Compute final mean and standard deviation of metrics across runs.
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
