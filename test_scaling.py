# train_scaling_experiment.py
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import json
import time
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

# Import the enhanced model and its evaluation function.
# The EnhancedScalingCNN should be designed to accept one (or more) external feature set(s)
# and incorporate them (e.g. via concatenation) in its forward pass.
from model_def_test10.model_def_scaling import EnhancedScalingCNN, evaluate_enhanced

# Import the simple CNN for base and external models.
from model_def_test10.model_def import CNN

# Set seeds for reproducibility.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

###############################################################################
# DATASET LOADING FUNCTIONS
###############################################################################

def get_full_trainset():
    # Load the full CIFAR-10 training set.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

def load_subset(dataset, start_idx, num_samples):
    # Returns a subset of the dataset from a given start index.
    indices = np.arange(start_idx, start_idx + num_samples)
    return Subset(dataset, indices)

def load_test_sets():
    # Divide the CIFAR-10 test set into 5 subsets (each 2000 samples).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    indices = np.arange(len(testset))
    copies = []
    for i in range(5):
        subset_indices = indices[i * 2000:(i + 1) * 2000]
        copies.append(Subset(testset, subset_indices))
    return copies

###############################################################################
# DATA CORRUPTION
###############################################################################

class CorruptedDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that flips the label for each sample.
    For CIFAR-10, the label is flipped by doing: (label + 1) % 10.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        corrupted_label = (label + 1) % 10  # simple flip
        return img, corrupted_label

    def __len__(self):
        return len(self.dataset)

###############################################################################
# TRAINING FUNCTIONS
###############################################################################

def train_model(model, dataloader, epochs, device):
    # Generic training loop using SGD.
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def train_enhanced(model, external_models, dataloader, epochs, device):
    """
    Training loop for the enhanced model.
    external_models is a list (typically of one) pre-trained external models.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # Set external models to eval mode.
    for ext_model in external_models:
        ext_model.eval()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=len(external_models) if external_models else 1)
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            futures = []
            # Extract external features concurrently.
            for ext_model in external_models:
                futures.append(executor.submit(lambda m, inp: m.get_features(inp.to(device)), ext_model, inputs))
            external_features_list = [future.result() for future in futures]
            # Enhanced model takes base input and external features.
            outputs = model(inputs, external_features_list if external_features_list else None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    executor.shutdown()

###############################################################################
# NAIVE MODEL BENCHMARK FUNCTIONS
###############################################################################

class FeatureDatasetConcat(torch.utils.data.Dataset):
    """
    A dataset that precomputes and concatenates features obtained from a list of external models.
    Each sample is a tuple (concatenated_features, label).
    """
    def __init__(self, base_dataset, external_models, device):
        self.base_dataset = base_dataset
        self.external_models = external_models
        self.device = device
        self.features = []
        self.labels = []
        self.precompute_features()

    def precompute_features(self):
        for model in self.external_models:
            model.eval()
        loader = DataLoader(self.base_dataset, batch_size=64, shuffle=False, num_workers=2)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                feats_list = []
                for ext_model in self.external_models:
                    feats = ext_model.get_features(inputs)
                    feats_list.append(feats.cpu())
                # Concatenate features along dimension 1.
                concat_feats = torch.cat(feats_list, dim=1)
                self.features.append(concat_feats)
                self.labels.append(labels)
        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

class NaiveClassifier(nn.Module):
    """
    A simple MLP classifier that accepts feature vectors as input.
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=10):
        super(NaiveClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

def train_naive(model, dataloader, epochs, device):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for feats, labels in dataloader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

###############################################################################
# EVALUATION FUNCTIONS (NO INFERENCE TIME)
###############################################################################

def evaluate_classifier(model, dataloader, device):
    model.eval()
    total_samples = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            probs = F.softmax(outputs, dim=1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='macro')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    auc = roc_auc_score(all_labels_binarized, all_probs, average='macro', multi_class='ovr')
    return accuracy, auc, f1

def evaluate_naive(model, dataloader, device):
    model.eval()
    total_samples = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for feats, labels in dataloader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            _, predicted = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='macro')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    auc = roc_auc_score(all_labels_binarized, all_probs, average='macro', multi_class='ovr')
    return accuracy, auc, f1

###############################################################################
# MAIN FUNCTION
###############################################################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # All models and results will be saved under this directory.
    os.makedirs("./model_scaling_ablate", exist_ok=True)
    os.makedirs("./model_scaling_ablate/model", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # ----- Parameters -----
    train_size = 2000         # Number of samples per disjoint dataset.
    pretrain_epochs = 30      # Training epochs for base and external models.
    enhanced_epochs = 30      # Training epochs for enhanced models and naive models.
    # -----------------------
    
    full_trainset = get_full_trainset()
    
    # -------------------------------
    # DEFINE DISJOINT DATASETS
    # -------------------------------
    # Base model uses the first 2000 samples.
    base0_dataset = load_subset(full_trainset, 0, train_size)
    
    # External models: 8 disjoint subsets. They start right after the base dataset.
    external_datasets = []
    for i in range(8):
        start_idx = train_size + i * train_size
        subset = load_subset(full_trainset, start_idx, train_size)
        # For odd-indexed external models (external1,3,5,7) use corrupted labels.
        if (i % 2) == 0:
            # i=0 corresponds to external1, i=2 corresponds to external3, etc.
            subset = CorruptedDataset(subset)
        external_datasets.append(subset)
    
    # -------------------------------
    # TRAINING PHASE
    # -------------------------------
    # Train base model.
    print("Training Base model (base0)...")
    base_model = CNN().to(device)
    base_loader = DataLoader(base0_dataset, batch_size=64, shuffle=True, num_workers=2)
    train_model(base_model, base_loader, pretrain_epochs, device)
    torch.save(base_model.state_dict(), "./model_scaling_ablate/model/base.pt")
    
    # Train external models: external1 to external8.
    external_models = []  # To store trained external models.
    for i in range(8):
        print(f"Training External model external{i+1}...")
        model_ext = CNN().to(device)
        loader_ext = DataLoader(external_datasets[i], batch_size=64, shuffle=True, num_workers=2)
        train_model(model_ext, loader_ext, pretrain_epochs, device)
        torch.save(model_ext.state_dict(), f"./model_scaling_ablate/model/external{i+1}.pt")
        external_models.append(model_ext)
    
    # Train enhanced models: enhanced1 to enhanced8.
    enhanced_models = []
    for i in range(8):
        print(f"Training Enhanced model enhanced{i+1} with external{i+1}...")
        model_enh = EnhancedScalingCNN(num_external=1).to(device)
        enh_loader = DataLoader(base0_dataset, batch_size=64, shuffle=True, num_workers=2)
        # Train using the corresponding external model in a list.
        train_enhanced(model_enh, [external_models[i]], enh_loader, enhanced_epochs, device)
        torch.save(model_enh.state_dict(), f"./model_scaling_ablate/model/enhanced{i+1}.pt")
        enhanced_models.append(model_enh)
    
    # Train naive models: naive1 to naive8.
    # naive_i uses the concatenated features from external1 up to external_i.
    naive_models = []
    for i in range(8):
        print(f"Training Naive model naive{i+1} using external1 to external{i+1}...")
        # Use FeatureDatasetConcat over base0_dataset and a list of external models.
        concat_dataset = FeatureDatasetConcat(base0_dataset, external_models[:i+1], device)
        concat_loader = DataLoader(concat_dataset, batch_size=64, shuffle=True, num_workers=2)
        # Determine input dimension from a sample.
        sample_feat, _ = concat_dataset[0]
        input_dim = sample_feat.numel()
        model_naive = NaiveClassifier(input_dim=input_dim).to(device)
        train_naive(model_naive, concat_loader, enhanced_epochs, device)
        torch.save(model_naive.state_dict(), f"./model_scaling_ablate/model/naive{i+1}.pt")
        naive_models.append(model_naive)
    
    # -------------------------------
    # EVALUATION PHASE
    # -------------------------------
    test_copies = load_test_sets()  # 5 test subsets, each with 2000 samples.
    results = {}
    
    # Evaluate Base Model.
    print("Evaluating Base model...")
    base_acc_list, base_auc_list, base_f1_list = [], [], []
    for test_subset in test_copies:
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
        acc, auc, f1 = evaluate_classifier(base_model, test_loader, device)
        base_acc_list.append(acc)
        base_auc_list.append(auc)
        base_f1_list.append(f1)
    results["base_model"] = {
        "accuracy_mean": float(np.mean(base_acc_list)),
        "accuracy_std": float(np.std(base_acc_list)),
        "auc_mean": float(np.mean(base_auc_list)),
        "auc_std": float(np.std(base_auc_list)),
        "f1_mean": float(np.mean(base_f1_list)),
        "f1_std": float(np.std(base_f1_list))
    }
    
    # Evaluate External Models.
    results["external_models"] = {}
    for i, model_ext in enumerate(external_models):
        ext_acc_list, ext_auc_list, ext_f1_list = [], [], []
        for test_subset in test_copies:
            test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
            acc, auc, f1 = evaluate_classifier(model_ext, test_loader, device)
            ext_acc_list.append(acc)
            ext_auc_list.append(auc)
            ext_f1_list.append(f1)
        results["external_models"][f"external{i+1}"] = {
            "accuracy_mean": float(np.mean(ext_acc_list)),
            "accuracy_std": float(np.std(ext_acc_list)),
            "auc_mean": float(np.mean(ext_auc_list)),
            "auc_std": float(np.std(ext_auc_list)),
            "f1_mean": float(np.mean(ext_f1_list)),
            "f1_std": float(np.std(ext_f1_list))
        }
    
    # Evaluate Enhanced Models.
    results["enhanced_models"] = {}
    for i, model_enh in enumerate(enhanced_models):
        enh_acc_list, enh_auc_list, enh_f1_list = [], [], []
        for test_subset in test_copies:
            test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
            # Use evaluate_enhanced: pass the corresponding external model in a list.
            acc, auc, f1 = evaluate_enhanced(model_enh, [external_models[i]], test_loader, device)[0:3]
            enh_acc_list.append(acc)
            enh_auc_list.append(auc)
            enh_f1_list.append(f1)
        results["enhanced_models"][f"enhanced{i+1}"] = {
            "accuracy_mean": float(np.mean(enh_acc_list)),
            "accuracy_std": float(np.std(enh_acc_list)),
            "auc_mean": float(np.mean(enh_auc_list)),
            "auc_std": float(np.std(enh_auc_list)),
            "f1_mean": float(np.mean(enh_f1_list)),
            "f1_std": float(np.std(enh_f1_list))
        }
    
    # Evaluate Naive Models.
    results["naive_models"] = {}
    for i, model_naive in enumerate(naive_models):
        naive_acc_list, naive_auc_list, naive_f1_list = [], [], []
        for test_subset in test_copies:
            # Create a FeatureDatasetConcat for testing with external_models[:i+1]
            concat_test_dataset = FeatureDatasetConcat(test_subset, external_models[:i+1], device)
            test_loader = DataLoader(concat_test_dataset, batch_size=64, shuffle=False, num_workers=2)
            acc, auc, f1 = evaluate_naive(model_naive, test_loader, device)
            naive_acc_list.append(acc)
            naive_auc_list.append(auc)
            naive_f1_list.append(f1)
        results["naive_models"][f"naive{i+1}"] = {
            "accuracy_mean": float(np.mean(naive_acc_list)),
            "accuracy_std": float(np.std(naive_acc_list)),
            "auc_mean": float(np.mean(naive_auc_list)),
            "auc_std": float(np.std(naive_auc_list)),
            "f1_mean": float(np.mean(naive_f1_list)),
            "f1_std": float(np.std(naive_f1_list))
        }
    
    # Write out all results.
    with open("./results/scaling_ablate.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Experiment completed. Results saved to ./results/scaling_ablate.json")

if __name__ == "__main__":
    main()
