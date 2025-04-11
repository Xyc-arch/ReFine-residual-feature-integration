import torch
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

from model_def_test10.model_def_tf import (
    TransformerClassifier,
    EnhancedTransformer,
    BaselineAdapterTransformer,
    BigTransformer
)
from train_eval import train_model, train_linear_prob, train_enhanced_model, train_distillation, evaluate_model

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def normalize_dict(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}

def load_data_split_imbalanced(seed, imbalance_dict):
    """
    Loads CIFAR-10 data and splits the training set into:
      - pretrain_set: first 10,000 samples, re-sampled to follow the desired class distribution.
      - raw_set: next 4,000 samples with true labels.
    The imbalance_dict specifies the desired proportion for each class (keys 0-9).
    """
    imbalance_dict = normalize_dict(imbalance_dict)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    total_indices = np.arange(len(trainset))
    rng = np.random.RandomState(seed)
    rng.shuffle(total_indices)
    
    # Use first 10k indices for pretraining and next 4k for raw set.
    pretrain_indices = total_indices[:10000]
    raw_indices = total_indices[10000:10000+4000]
    
    pretrain_subset = Subset(trainset, pretrain_indices)
    raw_set = Subset(trainset, raw_indices)
    
    # Group indices in pretrain_subset by true label.
    indices_by_label = {i: [] for i in range(10)}
    for i in range(len(pretrain_subset)):
        _, label = pretrain_subset[i]
        indices_by_label[label].append(i)
    
    total_pretrain = len(pretrain_subset)
    desired_counts = {c: int(total_pretrain * imbalance_dict.get(c, 0)) for c in range(10)}
    
    sampled_indices = []
    for c in range(10):
        available = indices_by_label[c]
        count = desired_counts[c]
        if len(available) >= count:
            sampled = rng.choice(available, size=count, replace=False).tolist()
        else:
            sampled = rng.choice(available, size=count, replace=True).tolist()
        sampled_indices.extend(sampled)
    
    pretrain_dataset = Subset(pretrain_subset, sampled_indices)
    return pretrain_dataset, raw_set, testset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_epochs = 60
    num_epochs = 30
    num_runs = 5
    # Define desired imbalance distribution.
    imbalance_dict = {
        0: 0.35,
        1: 0.3,
        2: 0.1,
        3: 0.07,
        4: 0.06,
        5: 0.045,
        6: 0.03,
        7: 0.02,
        8: 0.015,
        9: 0.01
    }
    save_path = "./results_test10/imb_tf.json"

    pretrain_dataset, raw_set, test_dataset = load_data_split_imbalanced(seed=42, imbalance_dict=imbalance_dict)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=2)
    raw_loader = DataLoader(raw_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    print("\n=== Pretraining External Model (BigTransformer) on 10k Imbalanced Samples ===")
    imb_tf_save_path = "./model_test10/imb_tf.pt"
    if os.path.exists(imb_tf_save_path):
        external_model = torch.load(imb_tf_save_path).to(device)
        print("Loaded external model from:", imb_tf_save_path)
    else:
        external_model = BigTransformer().to(device)
        train_model(external_model, pretrain_loader, pretrain_epochs, device)
        torch.save(external_model, imb_tf_save_path)
        print("Trained and saved external model to:", imb_tf_save_path)
    ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
    print(f"External Model Evaluation: Acc={ext_acc:.2f}%, AUC={ext_auc:.4f}, F1={ext_f1:.4f}, MinCAcc={ext_minc:.2f}%")

    
    metrics = {
        "baseline": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "linear_prob": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "enhanced_concat": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "baseline_adapter": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "distillation": {"acc": [], "auc": [], "f1": [], "min_cacc": []}
    }
    
    for run_idx in range(num_runs):
        run_seed = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{num_runs}, raw-set seed={run_seed} ===")
        _, raw_set_run, _ = load_data_split_imbalanced(seed=run_seed, imbalance_dict=imbalance_dict)
        run_loader = DataLoader(raw_set_run, batch_size=64, shuffle=True, num_workers=2)
        
        print("Training baseline model (TransformerClassifier on raw set)...")
        baseline_model = TransformerClassifier().to(device)
        train_model(baseline_model, run_loader, num_epochs, device)
        acc_b, auc_b, f1_b, min_cacc_b = evaluate_model(baseline_model, test_loader, device)
        metrics["baseline"]["acc"].append(acc_b)
        metrics["baseline"]["auc"].append(auc_b)
        metrics["baseline"]["f1"].append(f1_b)
        metrics["baseline"]["min_cacc"].append(min_cacc_b)
        
        print("Training linear probe model (fine-tuning external model's final classifier)...")
        linear_model = copy.deepcopy(external_model)
        for param in linear_model.parameters():
            param.requires_grad = False
        for param in linear_model.classifier.parameters():
            param.requires_grad = True
        train_linear_prob(linear_model, run_loader, num_epochs, device)
        acc_lp, auc_lp, f1_lp, min_cacc_lp = evaluate_model(linear_model, test_loader, device)
        metrics["linear_prob"]["acc"].append(acc_lp)
        metrics["linear_prob"]["auc"].append(auc_lp)
        metrics["linear_prob"]["f1"].append(f1_lp)
        metrics["linear_prob"]["min_cacc"].append(minc_lp if (minc_lp := min_cacc_lp) is not None else min_cacc_lp)
        
        print("Training enhanced model (concatenation)...")
        enhanced_concat_model = EnhancedTransformer().to(device)
        train_enhanced_model(enhanced_concat_model, run_loader, external_model, num_epochs, device)
        acc_ec, auc_ec, f1_ec, min_cacc_ec = evaluate_model(enhanced_concat_model, test_loader, device, enhanced=True, external_model=external_model)
        metrics["enhanced_concat"]["acc"].append(acc_ec)
        metrics["enhanced_concat"]["auc"].append(auc_ec)
        metrics["enhanced_concat"]["f1"].append(f1_ec)
        metrics["enhanced_concat"]["min_cacc"].append(min_cacc_ec)
        
        print("Training baseline adapter model (external frozen with adapter)...")
        baseline_adapter_model = BaselineAdapterTransformer(copy.deepcopy(external_model)).to(device)
        train_model(baseline_adapter_model, run_loader, num_epochs, device)
        acc_ba, auc_ba, f1_ba, min_cacc_ba = evaluate_model(baseline_adapter_model, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc_ba)
        metrics["baseline_adapter"]["auc"].append(auc_ba)
        metrics["baseline_adapter"]["f1"].append(f1_ba)
        metrics["baseline_adapter"]["min_cacc"].append(min_cacc_ba)
        
        print("Training knowledge distillation model (Transformer student with teacher external)...")
        student_model = TransformerClassifier().to(device)
        train_distillation(student_model, external_model, run_loader, num_epochs, device, temperature=2.0, alpha=0.5)
        acc_kd, auc_kd, f1_kd, min_cacc_kd = evaluate_model(student_model, test_loader, device)
        metrics["distillation"]["acc"].append(acc_kd)
        metrics["distillation"]["auc"].append(auc_kd)
        metrics["distillation"]["f1"].append(f1_kd)
        metrics["distillation"]["min_cacc"].append(min_cacc_kd)
        
        print(f"\n[Run {run_idx+1} Results]")
        print(f"Baseline:         Acc={acc_b:.2f}% | AUC={auc_b:.4f} | F1={f1_b:.4f} | MinCAcc={min_cacc_b:.2f}%")
        print(f"Linear Probe:     Acc={acc_lp:.2f}% | AUC={auc_lp:.4f} | F1={f1_lp:.4f} | MinCAcc={min_cacc_lp:.2f}%")
        print(f"Enhanced (Concat):Acc={acc_ec:.2f}% | AUC={auc_ec:.4f} | F1={f1_ec:.4f} | MinCAcc={min_cacc_ec:.2f}%")
        print(f"Baseline Adapter: Acc={acc_ba:.2f}% | AUC={auc_ba:.4f} | F1={f1_ba:.4f} | MinCAcc={min_cacc_ba:.2f}%")
        print(f"Distillation:     Acc={acc_kd:.2f}% | AUC={auc_kd:.4f} | F1={f1_kd:.4f} | MinCAcc={min_cacc_kd:.2f}%")
    
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
