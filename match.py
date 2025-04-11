import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
import json
import os
import copy
from train_eval_test100 import *  # Expects train_model, train_linear_prob, train_enhanced_model, evaluate_model

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Control parameters
PRETRAIN_SIZE = 10000  # Number of CIFAR-10 examples for teacher pretraining
RAW_SIZE = 2000        # Number of STL-10 training examples for finetuning
NUM_EPOCHS = 30
NUM_EPOCH_TEACHER = 60
NUM_RUNS = 5

# Import models
from model_def_test100.model_def10 import BigCNN        # Teacher (pretrained on CIFAR-10)
from model_def_test100.model_def_stl import CNN, EnhancedCNN, BaselineAdapter  # STL student models

class ResizeWrapper(nn.Module):
    """
    Resizes input images to target_size before passing them into the wrapped model.
    Also provides get_features() forwarding.
    """
    def __init__(self, model, target_size=(32, 32)):
        super().__init__()
        self.model = model
        self.target_size = target_size

    def forward(self, x):
        x_resized = torch.nn.functional.interpolate(
            x, size=self.target_size, mode='bilinear', align_corners=False
        )
        return self.model(x_resized)
    
    def get_features(self, x):
        x_resized = torch.nn.functional.interpolate(
            x, size=self.target_size, mode='bilinear', align_corners=False
        )
        return self.model.get_features(x_resized)

def download_stl():
    """Download STL-10 train and test splits if not present."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    torchvision.datasets.STL10(root='./data', split="train", download=True, transform=transform)
    torchvision.datasets.STL10(root='./data', split="test", download=True, transform=transform)

def load_teacher_data(pretrain_size=PRETRAIN_SIZE, seed=42):
    """Load and subsample CIFAR-10 for teacher pretraining."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:pretrain_size])

def load_stl_data(split="train", raw_size=None, seed=42):
    """
    Load STL-10 data. For the training split, subsample raw_size examples using the specified seed.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.STL10(root='./data', split=split, download=False, transform=transform)
    if split == "train" and raw_size is not None and raw_size < len(dataset):
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        dataset = Subset(dataset, indices[:raw_size])
    return dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./results_test100/match_results.json"

    # Download STL-10 data
    print("Checking STL-10 dataset...")
    download_stl()

    # ---------------------------
    # Teacher Pretraining on CIFAR-10
    # ---------------------------
    teacher_model_save_path = "./model_test100/match.pt"
    if os.path.exists(teacher_model_save_path):
        teacher_model = torch.load(teacher_model_save_path).to(device)
        print("Loaded teacher model from:", teacher_model_save_path)
    else:
        teacher_trainset = load_teacher_data(pretrain_size=PRETRAIN_SIZE, seed=42)
        teacher_loader = DataLoader(teacher_trainset, batch_size=32, shuffle=True, num_workers=2)
        teacher_model = BigCNN().to(device)
        train_model(teacher_model, teacher_loader, NUM_EPOCH_TEACHER, device)
        torch.save(teacher_model, teacher_model_save_path)
        print("Trained and saved teacher model to:", teacher_model_save_path)
    teacher_stl = ResizeWrapper(teacher_model, target_size=(32, 32)).to(device)


    # Create a ResizeWrapper for using the teacher on STL-10 images (96x96 -> 32x32).
    teacher_stl = ResizeWrapper(teacher_model, target_size=(32, 32)).to(device)

    # Constant STL test set
    stl_testset = load_stl_data(split="test")
    stl_test_loader = DataLoader(stl_testset, batch_size=32, shuffle=False, num_workers=2)

    # Initialize metrics for experiments
    metrics = {
        "baseline": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "linear_prob": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "enhanced_concat": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "baseline_adapter": {"acc": [], "auc": [], "f1": [], "min_cacc": []}
    }

    # ---------------------------
    # Run experiments: each run uses a different STL training subset.
    # ---------------------------
    for run_idx in range(NUM_RUNS):
        run_seed = 42 + run_idx
        print(f"\n=== Run {run_idx+1}/{NUM_RUNS}, seed={run_seed} ===")
        stl_trainset = load_stl_data(split="train", raw_size=RAW_SIZE, seed=run_seed)
        stl_train_loader = DataLoader(stl_trainset, batch_size=32, shuffle=True, num_workers=2)

        # 1. Baseline STL model finetuning (student CNN)
        print("Training baseline STL model...")
        baseline_model = CNN().to(device)
        train_model(baseline_model, stl_train_loader, NUM_EPOCHS, device)
        acc, auc, f1, min_cacc = evaluate_model_stl(baseline_model, stl_test_loader, device)
        metrics["baseline"]["acc"].append(acc)
        metrics["baseline"]["auc"].append(auc)
        metrics["baseline"]["f1"].append(f1)
        metrics["baseline"]["min_cacc"].append(min_cacc)

        # 2. Linear Probe: Fine-tune teacher model (wrapped) on STL
        print("Training linear probe model...")
        # Copy the teacher model, freeze parameters, replace final head with 10-class output.
        linear_model = copy.deepcopy(teacher_model)
        for param in linear_model.parameters():
            param.requires_grad = False
        linear_model.fc_layers[-1] = nn.Linear(2560, 10)
        for param in linear_model.fc_layers[-1].parameters():
            param.requires_grad = True
        # Wrap to resize STL inputs
        linear_model = ResizeWrapper(linear_model, target_size=(32, 32)).to(device)
        train_linear_prob(linear_model, stl_train_loader, NUM_EPOCHS, device)
        acc, auc, f1, min_cacc = evaluate_model_stl(linear_model, stl_test_loader, device)
        metrics["linear_prob"]["acc"].append(acc)
        metrics["linear_prob"]["auc"].append(auc)
        metrics["linear_prob"]["f1"].append(f1)
        metrics["linear_prob"]["min_cacc"].append(min_cacc)

        # 3. Enhanced Concatenation: Train student model using teacher features.
        print("Training enhanced concatenation model...")
        enhanced_model = EnhancedCNN().to(device)
        train_enhanced_model(enhanced_model, stl_train_loader, teacher_stl, NUM_EPOCHS, device)
        acc, auc, f1, min_cacc = evaluate_model_stl(
            enhanced_model, stl_test_loader, device, enhanced=True, external_model=teacher_stl
        )
        metrics["enhanced_concat"]["acc"].append(acc)
        metrics["enhanced_concat"]["auc"].append(auc)
        metrics["enhanced_concat"]["f1"].append(f1)
        metrics["enhanced_concat"]["min_cacc"].append(min_cacc)

        # 4. Baseline Adapter: Fine-tune an adapter model using frozen teacher features.
        print("Training baseline adapter model...")
        # Use teacher_stl to ensure proper input resize.
        adapter_model = BaselineAdapter(copy.deepcopy(teacher_stl)).to(device)
        train_model(adapter_model, stl_train_loader, NUM_EPOCHS, device)
        acc, auc, f1, min_cacc = evaluate_model_stl(adapter_model, stl_test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc)
        metrics["baseline_adapter"]["auc"].append(auc)
        metrics["baseline_adapter"]["f1"].append(f1)
        metrics["baseline_adapter"]["min_cacc"].append(min_cacc)

        print(f"\n[Run {run_idx+1} Results]")
        print(f"Baseline:          Acc={metrics['baseline']['acc'][-1]:.2f}%")
        print(f"Linear Probe:      Acc={metrics['linear_prob']['acc'][-1]:.2f}%")
        print(f"Enhanced (Concat): Acc={metrics['enhanced_concat']['acc'][-1]:.2f}%")
        print(f"Baseline Adapter:  Acc={metrics['baseline_adapter']['acc'][-1]:.2f}%")

    # Aggregate results and save JSON.
    final_results = {}
    for method, m in metrics.items():
        final_results[method] = {
            "acc_mean": float(np.mean(m["acc"])),
            "acc_std": float(np.std(m["acc"])),
            "auc_mean": float(np.mean(m["auc"])),
            "auc_std": float(np.std(m["auc"])),
            "f1_mean": float(np.mean(m["f1"])),
            "f1_std": float(np.std(m["f1"])),
            "min_cacc_mean": float(np.mean(m["min_cacc"])),
            "min_cacc_std": float(np.std(m["min_cacc"]))
        }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()
