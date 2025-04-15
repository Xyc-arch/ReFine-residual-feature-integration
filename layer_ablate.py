import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset

# For metrics
from sklearn.metrics import roc_auc_score, f1_score

# Import BigCNN from the original model definitions (noise.py)
from model_def_test10.model_def import BigCNN

# Import the three enhanced ablation models from our new module
from model_def_test10.model_def_layer_ablate import EnhancedCNN_L1, EnhancedCNN_L2, EnhancedCNN_L3

# --- Data Preparation Function (same as in noise.py) ---
def load_data_split(seed, flip_ratio=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    total_indices = np.arange(len(trainset))
    rng = np.random.RandomState(seed)
    rng.shuffle(total_indices)
    
    pretrain_indices = total_indices[:10000]
    raw_indices = total_indices[10000:10000+4000]
    
    pretrain_subset = Subset(trainset, pretrain_indices)
    raw_set = Subset(trainset, raw_indices)

    class RandomLabelDataset(Dataset):
        def __init__(self, subset, num_classes=10, flip_ratio=flip_ratio):
            self.subset = subset
            self.num_classes = num_classes
            self.flip_ratio = flip_ratio
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            image, true_label = self.subset[idx]
            if np.random.rand() < self.flip_ratio:
                wrong_label = np.random.randint(0, self.num_classes - 1)
                if wrong_label >= true_label:
                    wrong_label += 1
                return image, wrong_label
            else:
                return image, true_label

    pretrain_dataset = RandomLabelDataset(pretrain_subset, flip_ratio=flip_ratio)
    
    return pretrain_dataset, raw_set, testset

# --- Training and Evaluation functions for the enhanced models ---

def train_enhanced_ablate(model, train_loader, external_model, epochs, device):
    model.train()
    # Replace Adam with SGD with lr 0.01 and momentum 0.9.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    external_model.eval()  # BigCNN remains fixed
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                ext_feats = external_model.get_features(images)
            optimizer.zero_grad()
            outputs = model(images, ext_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"[Enhanced Train] Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

def evaluate_enhanced_ablate(model, test_loader, external_model, device):
    model.eval()
    all_labels = []
    all_preds = []
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ext_feats = external_model.get_features(images)
            outputs = model(images, ext_feats)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
    accuracy = 100.0 * correct / total
    
    try:
        from torch.nn import functional as F
        probs = F.softmax(torch.tensor(all_preds), dim=1).numpy()
        num_classes = 10
        all_labels_np = np.array(all_labels)
        all_labels_oh = np.eye(num_classes)[all_labels_np]
        auc = roc_auc_score(all_labels_oh, probs, average='macro', multi_class='ovr')
    except Exception as e:
        auc = 0.0

    f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='macro')
    
    # Compute minimum per-class accuracy.
    all_labels_np = np.array(all_labels)
    predictions = np.argmax(all_preds, axis=1)
    class_acc = []
    for c in range(10):
        idx = (all_labels_np == c)
        if idx.sum() > 0:
            acc_c = 100.0 * (predictions[idx] == c).sum() / idx.sum()
            class_acc.append(acc_c)
    min_cacc = min(class_acc) if class_acc else 0.0
    
    return accuracy, auc, f1, min_cacc

# --- Main Experiment ---
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_epochs = 60    # Pretraining epochs for BigCNN
    other_epochs = 30       # Training epochs for enhanced models
    num_runs = 5
    
    # Loop over different flip_ratio values.
    for flip_ratio in [0.8, 0]:
        print(f"\n========== Experiments with flip_ratio = {flip_ratio} ==========")
        # --- Pretrain BigCNN (or load if existing) using noise pretrain data ---
        # Use the same naming convention from noise.py for pretrained models.
        noise_model_path = "./model_test10/noise_{}.pt".format(flip_ratio)
        pretrain_dataset, raw_set_global, testset = load_data_split(seed=42, flip_ratio=flip_ratio)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    
        if os.path.exists(noise_model_path):
            external_model = torch.load(noise_model_path, map_location=device)
            external_model.to(device)
            print(f"Loaded pretrained BigCNN from: {noise_model_path}")
        else:
            external_model = BigCNN().to(device)
            # Replace Adam with SGD with lr 0.01 and momentum 0.9.
            optimizer = torch.optim.SGD(external_model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            external_model.train()
            for epoch in range(pretrain_epochs):
                running_loss = 0.0
                for images, labels in pretrain_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = external_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                avg_loss = running_loss / len(pretrain_loader)
                print(f"[Pretrain BigCNN] Epoch {epoch+1}/{pretrain_epochs}: Loss = {avg_loss:.4f}")
            torch.save(external_model, noise_model_path)
            print(f"Trained and saved pretrained BigCNN to: {noise_model_path}")
    
        # --- Ablation Study on Enhanced Models ---
        metrics = {
            "enhanced_l1": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
            "enhanced_l2": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
            "enhanced_l3": {"acc": [], "auc": [], "f1": [], "min_cacc": []}
        }
    
        for run_idx in range(num_runs):
            run_seed = 42 + run_idx
            print(f"\n=== Run {run_idx+1}/{num_runs}, raw-set seed = {run_seed} ===")
            _, raw_set_run, _ = load_data_split(seed=run_seed, flip_ratio=flip_ratio)
            run_loader = DataLoader(raw_set_run, batch_size=64, shuffle=True, num_workers=2)
            
            # For each ablation variant, create, train, and evaluate a new model.
            for variant, ModelClass in zip(
                    ["enhanced_l1", "enhanced_l2", "enhanced_l3"],
                    [EnhancedCNN_L1, EnhancedCNN_L2, EnhancedCNN_L3]):
                
                print(f"--- Training model variant: {variant} ---")
                model_variant = ModelClass().to(device)
                train_enhanced_ablate(model_variant, run_loader, external_model, other_epochs, device)
                acc, auc, f1, min_cacc = evaluate_enhanced_ablate(model_variant, test_loader, external_model, device)
                
                metrics[variant]["acc"].append(acc)
                metrics[variant]["auc"].append(auc)
                metrics[variant]["f1"].append(f1)
                metrics[variant]["min_cacc"].append(min_cacc)
                
                print(f"Results for {variant}: Acc={acc:.2f}% | AUC={auc:.4f} | F1={f1:.4f} | MinCAcc={min_cacc:.2f}%")
    
        # Aggregate final results for this flip_ratio.
        final_results = {}
        for variant, m_dict in metrics.items():
            acc_arr = np.array(m_dict["acc"])
            auc_arr = np.array(m_dict["auc"])
            f1_arr = np.array(m_dict["f1"])
            minc_arr = np.array(m_dict["min_cacc"])
            final_results[variant] = {
                "acc_mean": float(acc_arr.mean()),
                "acc_std": float(acc_arr.std()),
                "auc_mean": float(auc_arr.mean()),
                "auc_std": float(auc_arr.std()),
                "f1_mean": float(f1_arr.mean()),
                "f1_std": float(f1_arr.std()),
                "min_cacc_mean": float(minc_arr.mean()),
                "min_cacc_std": float(minc_arr.std())
            }
    
        # Save JSON results with the flip_ratio appended to the filename.
        results_path = "./results_ablate/layer_ablate_{}.json".format(flip_ratio)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as fp:
            json.dump(final_results, fp, indent=2)
        print(f"\nAll done for flip_ratio = {flip_ratio}. Final ablation study results saved to: {results_path}")

if __name__ == "__main__":
    main()
