import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
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
# 1. Function to Build an Imbalanced Pretraining Dataset for CIFAR-100
# --------------------------
def load_data_split_imb_cifar100(seed, 
                                 majority_classes=list(range(10)), 
                                 majority_count=400, 
                                 minority_count=100, 
                                 target_total=10000):
    """
    Loads CIFAR-100 data and constructs two training subsets:
      - pretrain_subset: an imbalanced subset.
          * For classes in majority_classes, take up to majority_count samples.
          * For all other classes (minority), take up to minority_count samples.
          * Finally, trim (or pad) the collected indices to have target_total samples.
      - raw_set: a uniformly sampled subset (e.g. 4000 samples) used later for student training.
    
    Also returns the CIFAR-100 test set.
    
    Parameters:
      seed              : int, for reproducibility.
      majority_classes  : list of int, indices of classes that are designated as majority.
      majority_count    : int, max number of samples per majority class.
      minority_count    : int, max number of samples per minority class.
      target_total      : int, target size of the pretraining set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # Load the full training and test sets (CIFAR-100).
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Group training indices by class.
    num_classes = 100
    indices_by_class = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset):
        indices_by_class[label].append(idx)

    # Shuffle indices in each class.
    rng = np.random.RandomState(seed)
    for c in range(num_classes):
        rng.shuffle(indices_by_class[c])

    pretrain_indices = []
    for c in range(num_classes):
        if c in majority_classes:
            n = min(len(indices_by_class[c]), majority_count)
        else:
            n = min(len(indices_by_class[c]), minority_count)
        pretrain_indices.extend(indices_by_class[c][:n])
    
    # Trim the pretrain_indices if we collected more than target_total samples.
    pretrain_indices = pretrain_indices[:target_total]

    pretrain_subset = Subset(trainset, pretrain_indices)

    # Build a raw (uncorrupted) set for student training.
    total_indices = np.arange(len(trainset))
    rng.shuffle(total_indices)
    raw_indices = total_indices[:4000]
    raw_set = Subset(trainset, raw_indices)

    return pretrain_subset, raw_set, testset

# --------------------------
# 2. Main Experiment Loop (Imbalanced Pretraining on CIFAR-100)
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Save path for the final results.
    save_path = "./results_test100/imbalance_cifar100.json"

    pretrain_epochs = 60  # Number of epochs for pretraining on imbalanced data.
    other_epochs = 30     # Epochs for the remaining phases.
    num_runs = 5          # Number of different raw-set splits.

    # Load the imbalanced pretraining data (use a fixed seed for consistent pretraining set).
    pretrain_dataset, raw_set_common, testset = load_data_split_imb_cifar100(seed=42)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    
    model_save_path = "./model_test100/imbalance_cifar100.pt"
    if os.path.exists(model_save_path):
        external_model = torch.load(model_save_path).to(device)
        print("Loaded external model from:", model_save_path)
    else:
        external_model = BigCNN().to(device)
        train_model(external_model, pretrain_loader, pretrain_epochs, device)
        torch.save(external_model, model_save_path)
        print("Trained and saved external model to:", model_save_path)
    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Model (Pretrained on Imbalanced Data): Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

    
    # Prepare metrics containers for different training methods.
    metrics = {
        "baseline": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "linear_prob": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "enhanced_concat": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "baseline_adapter": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "distillation": {"acc": [], "auc": [], "f1": [], "min_cacc": []}
    }
    
    # Run experiments over different raw_set splits for student training.
    for run_idx in range(num_runs):
        seed_for_split = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, seed={seed_for_split} ===")
        
        # Reload the raw (uncorrupted) set using a new seed (to vary the raw data).
        _, raw_set, _ = load_data_split_imb_cifar100(seed=seed_for_split)
        raw_loader = DataLoader(raw_set, batch_size=64, shuffle=True, num_workers=2)
        
        # 1. Baseline: Train a CNN on raw_set only.
        print("Training baseline model (CNN on raw data)...")
        baseline_model = CNN().to(device)
        train_model(baseline_model, raw_loader, other_epochs, device)
        acc_b, auc_b, f1_b, min_cacc_b = evaluate_model(baseline_model, test_loader, device)
        metrics["baseline"]["acc"].append(acc_b)
        metrics["baseline"]["auc"].append(auc_b)
        metrics["baseline"]["f1"].append(f1_b)
        metrics["baseline"]["min_cacc"].append(min_cacc_b)
        
        # 2. Linear Probe: Fine-tune only the final layer of external_model on raw_set.
        print("Training linear probe model (external fine-tuning on raw data)...")
        linear_model = copy.deepcopy(external_model)
        for param in linear_model.parameters():
            param.requires_grad = False
        for param in linear_model.fc_layers[-1].parameters():
            param.requires_grad = True
        train_linear_prob(linear_model, raw_loader, other_epochs, device)
        acc_lp, auc_lp, f1_lp, min_cacc_lp = evaluate_model(linear_model, test_loader, device)
        metrics["linear_prob"]["acc"].append(acc_lp)
        metrics["linear_prob"]["auc"].append(auc_lp)
        metrics["linear_prob"]["f1"].append(f1_lp)
        metrics["linear_prob"]["min_cacc"].append(min_cacc_lp)
        
        # 3. Enhanced Concatenation: Train EnhancedCNN on raw_set using features from external_model.
        print("Training enhanced model (concatenation)...")
        enhanced_concat_model = EnhancedCNN().to(device)
        train_enhanced_model(enhanced_concat_model, raw_loader, external_model, other_epochs, device)
        acc_ec, auc_ec, f1_ec, min_cacc_ec = evaluate_model(
            enhanced_concat_model, test_loader, device, enhanced=True, external_model=external_model
        )
        metrics["enhanced_concat"]["acc"].append(acc_ec)
        metrics["enhanced_concat"]["auc"].append(auc_ec)
        metrics["enhanced_concat"]["f1"].append(f1_ec)
        metrics["enhanced_concat"]["min_cacc"].append(min_cacc_ec)
        
        # 4. Baseline Adapter: Freeze external_model and fine-tune an adapter network on raw_set.
        print("Training baseline adapter model (external frozen with adapter) on raw data...")
        baseline_adapter_model = BaselineAdapter(copy.deepcopy(external_model)).to(device)
        train_model(baseline_adapter_model, raw_loader, other_epochs, device)
        acc_ba, auc_ba, f1_ba, min_cacc_ba = evaluate_model(baseline_adapter_model, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc_ba)
        metrics["baseline_adapter"]["auc"].append(auc_ba)
        metrics["baseline_adapter"]["f1"].append(f1_ba)
        metrics["baseline_adapter"]["min_cacc"].append(min_cacc_ba)
        
        # 5. Knowledge Distillation: Train a CNN student using external_model as teacher.
        print("Training knowledge distillation model (CNN student with teacher external)...")
        student_model = CNN().to(device)
        train_distillation(student_model, external_model, raw_loader, other_epochs, device, temperature=2.0, alpha=0.5)
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
    
    # Compute mean and standard deviation metrics across runs.
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
