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
from model_def_test10.model_def import CNN

# Set seeds.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


###############################################################################
# DATASET FUNCTIONS
###############################################################################
def get_full_trainset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

def load_subset(dataset, start_idx, num_samples):
    indices = np.arange(start_idx, start_idx + num_samples)
    return Subset(dataset, indices)

def load_test_sets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    indices = np.arange(len(testset))
    # Divide testset into 5 chunks of 2000 images each.
    chunks = []
    for i in range(5):
        subset_indices = indices[i * 2000:(i + 1) * 2000]
        chunks.append(Subset(testset, subset_indices))
    return chunks


class CorruptedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # Define partner pairs as a dictionary.
        self.partner_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8}

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # With 50% probability, flip the label to its partner.
        if random.random() < 0.5:
            new_label = self.partner_map[label]
        else:
            new_label = label
        return img, new_label

    def __len__(self):
        return len(self.dataset)


###############################################################################
# TRAINING FUNCTIONS
###############################################################################
def train_model(model, dataloader, epochs, device):
    model.train()
    # Learning rate updated to 0.001.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # print(f"  Epoch {epoch+1}/{epochs}") # Optional: uncomment for epoch progress
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def train_enhanced(model, external_models, dataloader, epochs, device):
    model.train()
    # Learning rate updated to 0.001.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # Set external models to eval mode.
    for ext_model in external_models:
        ext_model.eval()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=len(external_models) if external_models else 1)
    for epoch in range(epochs):
        # print(f"  Epoch {epoch+1}/{epochs}") # Optional: uncomment for epoch progress
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Ensure external models are on the correct device for feature extraction
            futures = [executor.submit(lambda m, inp: m.get_features(inp), ext_model.to(device), inputs)
                       for ext_model in external_models]
            external_features = [future.result() for future in futures]
            outputs = model(inputs, external_features if external_features else None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    executor.shutdown()

class EnhancedScalingCNN(nn.Module):
    def __init__(self, num_external=0):
        super(EnhancedScalingCNN, self).__init__()
        self.num_external = num_external
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Assume CNN get_features outputs 512-dim vector (or whatever CNN().feature_dim is)
        feature_dim_cnn_conv = 64 * 4 * 4 # Correct dimension after conv layers
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim_cnn_conv, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Assumes external models' get_features also output a fixed dim vector
        # This is safer: get the feature dim from a dummy external model instance
        try:
            external_feature_dim = CNN().get_features(torch.randn(1, 3, 32, 32)).size(1)
        except Exception:
             # Fallback if get_features is not easily runnable or has different structure
             external_feature_dim = 512 # Default assumption

        final_input_dim = 512 + (self.num_external * external_feature_dim)
        self.final_layer = nn.Linear(final_input_dim, 10)

    # Need a get_features method for consistency if this model were to be used externally
    # For this script's purpose, it's not strictly necessary as it's only used as the main model.

    def forward(self, x, external_features):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        if external_features is not None and len(external_features) > 0:
            # Ensure external features are on the same device as 'features'
            external_features = [feat.to(features.device) for feat in external_features]
            ext_concat = torch.cat(external_features, dim=1)
            combined = torch.cat((features, ext_concat), dim=1)
        else:
            combined = features
        return self.final_layer(combined)


class FeatureDatasetConcat(torch.utils.data.Dataset):
    def __init__(self, base_dataset, external_models, device):
        self.base_dataset = base_dataset
        self.external_models = external_models
        self.device = device
        self.features = []
        self.labels = []
        self.precompute_features()

    def precompute_features(self):
        print("Precomputing features for Naive model...")
        # Ensure external models are on the correct device and in eval mode
        for model in self.external_models:
            model.eval()
            model.to(self.device)

        loader = DataLoader(self.base_dataset, batch_size=64, shuffle=False, num_workers=2)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                feats_list = [ext_model.get_features(inputs).cpu() for ext_model in self.external_models]
                concat_feats = torch.cat(feats_list, dim=1)
                self.features.append(concat_feats)
                self.labels.append(labels)
        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        print("Feature precomputation complete.")

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

class NaiveClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(NaiveClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def train_naive(model, dataloader, epochs, device):
    model.train()
    # Learning rate updated to 0.001.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # print(f"  Epoch {epoch+1}/{epochs}") # Optional: uncomment for epoch progress
        for feats, labels in dataloader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


###############################################################################
# EVALUATION FUNCTIONS
###############################################################################
def evaluate_classifier(model, dataloader, device):
    model.eval()
    total_samples = 0
    correct = 0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    # Handle the case where AUC cannot be computed (e.g., only one class present)
    if len(np.unique(all_labels)) < 2:
        auc = float('nan') # Not Applicable or Undefined
    else:
        auc = roc_auc_score(all_labels_binarized, all_probs, average='weighted', multi_class='ovr')
    return accuracy, auc, f1

def evaluate_naive(model, dataloader, device):
    model.eval()
    total_samples = 0
    correct = 0
    all_labels, all_preds, all_probs = [], [], []
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
            # Corrected: Ensure tensor is on CPU before converting to numpy
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    if len(np.unique(all_labels)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(all_labels_binarized, all_probs, average='weighted', multi_class='ovr')
    return accuracy, auc, f1

def evaluate_enhanced(model, external_models, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    all_labels, all_preds, all_probs = [], [], []
    # Ensure external models are on the correct device and in eval mode
    for ext_model in external_models:
        ext_model.eval()
        ext_model.to(device)

    with torch.no_grad():
        from concurrent.futures import ThreadPoolExecutor
        # Adjust max_workers based on the number of external models
        max_workers = len(external_models) if external_models else 1
        # Limit max_workers to avoid excessive threads if many models exist
        if max_workers > os.cpu_count() * 2: # Heuristic limit
             max_workers = os.cpu_count() * 2

        executor = ThreadPoolExecutor(max_workers=max_workers)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Get external features if provided.
            if external_models is not None and len(external_models) > 0:
                 # Pass inputs to the model methods directly
                futures = [executor.submit(lambda m, inp: m.get_features(inp), ext_model, inputs)
                           for ext_model in external_models]
                external_features = [future.result() for future in futures]
            else:
                external_features = None # Ensure external_features is None or empty list if no external models
            outputs = model(inputs, external_features)
            _, predicted = torch.max(outputs.data, 1)
            probs = F.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        executor.shutdown() # Shutdown the executor after the loop

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    if len(np.unique(all_labels)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(all_labels_binarized, all_probs, average='weighted', multi_class='ovr')
    return accuracy, auc, f1


def evaluate_ensemble(base_model, external_models, dataloader, device):
    """Evaluates an unweighted ensemble of the base model and specified external models."""
    ensemble_members = [base_model] + external_models # List of models in this ensemble

    # Ensure all models are on the correct device and in eval mode
    for model in ensemble_members:
        model.eval()
        model.to(device)

    total = 0
    correct = 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get outputs (logits) from all models in the ensemble
            outputs_list = [model(inputs) for model in ensemble_members]

            # Sum the logits
            sum_outputs = torch.sum(torch.stack(outputs_list, dim=0), dim=0)

            # Average the logits (unweighted)
            avg_outputs = sum_outputs / len(ensemble_members)

            # Get predicted class and probabilities from averaged outputs
            _, predicted = torch.max(avg_outputs.data, 1)
            probs = F.softmax(avg_outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    if len(np.unique(all_labels)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(all_labels_binarized, all_probs, average='weighted', multi_class='ovr')
    return accuracy, auc, f1


###############################################################################
# UTILITY FUNCTION FOR LOADING OR TRAINING MODELS
###############################################################################
def load_or_train(model, filepath, train_fn, *train_args):
    if os.path.exists(filepath):
        print(f"Found {filepath}. Loading model...")
        # Ensure model is on the correct device before loading state dict
        # The last arg in *train_args is assumed to be the device
        target_device = train_args[-1] if train_args else torch.device("cpu")
        model.to(target_device)
        model.load_state_dict(torch.load(filepath, map_location=target_device))
    else:
        print(f"{filepath} not found. Training model...")
        # Ensure model is on device for training
        target_device = train_args[-1] if train_args else torch.device("cpu")
        model.to(target_device)
        # Pass args excluding model and including device correctly
        train_fn(model, *train_args[1:-1], target_device)
        torch.save(model.state_dict(), filepath)
    return model

###############################################################################
# MAIN
###############################################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # New model directory.
    os.makedirs("./model_scaling_low_lr", exist_ok=True)
    os.makedirs("./results_ablate", exist_ok=True)

    train_size = 2000
    pretrain_epochs = 30
    enhanced_epochs = 30

    full_trainset = get_full_trainset()

    # Dataset splitting.
    # Base dataset for the main model training
    base0_dataset = load_subset(full_trainset, 0, train_size)
    # External datasets: create 8 subsets (note: external datasets are now not corrupted).
    external_datasets = []
    for i in range(8):
        # Start after the base dataset samples
        start_idx = train_size + i * train_size
        subset = load_subset(full_trainset, start_idx, train_size)
        # external datasets are NOT corrupted in this version
        # subset = CorruptedDataset(subset) # This line is intentionally commented out/removed
        external_datasets.append(subset)

    # Base model.
    base_model = CNN().to(device)
    base_model_file = "./model_scaling_low_lr/base.pt"
    base_loader = DataLoader(base0_dataset, batch_size=64, shuffle=True, num_workers=2)
    # load_or_train args: model, filepath, train_fn, *train_args (model, dataloader, epochs, device)
    base_model = load_or_train(base_model, base_model_file, train_model, base_model, base_loader, pretrain_epochs, device)

    # External models.
    external_models = []
    for i in range(8):
        model_ext = CNN().to(device)
        ext_file = f"./model_scaling_low_lr/external{i+1}.pt"
        loader_ext = DataLoader(external_datasets[i], batch_size=64, shuffle=True, num_workers=2)
        # load_or_train args: model, filepath, train_fn, *train_args (model, dataloader, epochs, device)
        model_ext = load_or_train(model_ext, ext_file, train_model, model_ext, loader_ext, pretrain_epochs, device)
        external_models.append(model_ext)

    # Enhanced models.
    enhanced_models = []
    for i in range(8):
        model_enh = EnhancedScalingCNN(num_external=i+1).to(device)
        enh_file = f"./model_scaling_low_lr/enhanced{i+1}.pt"
        enh_loader = DataLoader(base0_dataset, batch_size=64, shuffle=True, num_workers=2)
        # load_or_train args: model, filepath, train_fn, *train_args (model, external_models_list, dataloader, epochs, device)
        # Need to pass the correct subset of external models for training
        model_enh = load_or_train(model_enh, enh_file, train_enhanced, model_enh, external_models[:i+1], enh_loader, enhanced_epochs, device)
        enhanced_models.append(model_enh)

    # Naive models.
    naive_models = []
    for i in range(8):
        # Create a dataset of features concatenated from the first i+1 external models on the base dataset
        concat_dataset = FeatureDatasetConcat(base0_dataset, external_models[:i+1], device)
        concat_loader = DataLoader(concat_dataset, batch_size=64, shuffle=True, num_workers=2)
        # Determine input dimension from the precomputed features
        sample_feat, _ = concat_dataset[0]
        input_dim = sample_feat.numel()
        model_naive = NaiveClassifier(input_dim=input_dim).to(device)
        naive_file = f"./model_scaling_low_lr/naive{i+1}.pt"
        # load_or_train args: model, filepath, train_fn, *train_args (model, dataloader, epochs, device)
        model_naive = load_or_train(model_naive, naive_file, train_naive, model_naive, concat_loader, enhanced_epochs, device)
        naive_models.append(model_naive)


    # Evaluation on test data.
    test_chunks = load_test_sets()
    results = {}

    # Utility function to compute mean and standard deviation from a list of metric tuples.
    def aggregate_metrics(metrics_list):
        metrics_arr = np.array(metrics_list)
        mean_vals = np.nanmean(metrics_arr, axis=0) # Use nanmean to handle potential NaNs from AUC
        std_vals = np.nanstd(metrics_arr, axis=0)  # Use nanstd to handle potential NaNs from AUC
        return {
            "accuracy": {"mean": float(mean_vals[0]), "std": float(std_vals[0]) if not np.isnan(std_vals[0]) else None},
            "auc": {"mean": float(mean_vals[1]) if not np.isnan(mean_vals[1]) else None, "std": float(std_vals[1]) if not np.isnan(std_vals[1]) else None},
            "f1": {"mean": float(mean_vals[2]), "std": float(std_vals[2]) if not np.isnan(std_vals[2]) else None},
        }

    print("\n--- Starting Evaluation ---")

    # Evaluate Base model.
    print("Evaluating Base model...")
    base_metrics = []
    for j, test_subset in enumerate(test_chunks):
        print(f"  Test Chunk {j+1}/5")
        loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
        base_metrics.append(evaluate_classifier(base_model, loader, device))
    results["base_model"] = aggregate_metrics(base_metrics)
    print(f"Base model evaluation complete. Results: {results['base_model']}")

    # Evaluate External models individually.
    results["external_models"] = {}
    for i, model_ext in enumerate(external_models):
        print(f"\nEvaluating External model {i+1}...")
        ext_metrics = []
        for j, test_subset in enumerate(test_chunks):
            print(f"  Test Chunk {j+1}/5")
            loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
            ext_metrics.append(evaluate_classifier(model_ext, loader, device))
        results["external_models"][f"external{i+1}"] = aggregate_metrics(ext_metrics)
        print(f"External model {i+1} evaluation complete. Results: {results['external_models'][f'external{i+1}']}")


    # Evaluate Enhanced models (scaling by number of external models used).
    results["enhanced_models"] = {}
    for i, model_enh in enumerate(enhanced_models):
        print(f"\nEvaluating Enhanced model with {i+1} external models...")
        enh_metrics = []
        for j, test_subset in enumerate(test_chunks):
            print(f"  Test Chunk {j+1}/5")
            loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
            # Pass the relevant subset of external models used during training
            enh_metrics.append(evaluate_enhanced(model_enh, external_models[:i+1], loader, device))
        results["enhanced_models"][f"enhanced{i+1}"] = aggregate_metrics(enh_metrics)
        print(f"Enhanced model {i+1} evaluation complete. Results: {results['enhanced_models'][f'enhanced{i+1}']}")


    # Evaluate Naive models (scaling by number of external features concatenated).
    results["naive_models"] = {}
    for i, model_naive in enumerate(naive_models):
        print(f"\nEvaluating Naive model with {i+1} external features...")
        naive_metrics = []
        for j, test_subset in enumerate(test_chunks):
            print(f"  Test Chunk {j+1}/5")
            # Need to precompute features for the test subset using the relevant external models
            concat_test_dataset = FeatureDatasetConcat(test_subset, external_models[:i+1], device)
            test_loader = DataLoader(concat_test_dataset, batch_size=64, shuffle=False, num_workers=2)
            naive_metrics.append(evaluate_naive(model_naive, test_loader, device))
        results["naive_models"][f"naive{i+1}"] = aggregate_metrics(naive_metrics)
        print(f"Naive model {i+1} evaluation complete. Results: {results['naive_models'][f'naive{i+1}']}")


    # --- ADDITION START: Evaluate Ensemble Baselines ---
    results["ensemble_baselines"] = {}
    print("\n--- Evaluating Ensemble Baselines ---")
    # Iterate from 1 external model up to 8 external models
    for i in range(8):
        num_external_in_ensemble = i + 1
        print(f"\nEvaluating Ensemble: Base + External 1-{num_external_in_ensemble}...")
        ensemble_metrics = []
        # Evaluate this ensemble on each test chunk
        for j, test_subset in enumerate(test_chunks):
             print(f"  Test Chunk {j+1}/5")
             loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
             # The ensemble includes the base model and the first 'num_external_in_ensemble' external models
             ensemble_metrics.append(evaluate_ensemble(base_model, external_models[:num_external_in_ensemble], loader, device))
        # Aggregate results across test chunks for this specific ensemble size
        results["ensemble_baselines"][f"ensemble{num_external_in_ensemble}"] = aggregate_metrics(ensemble_metrics)
        print(f"Ensemble {num_external_in_ensemble} evaluation complete. Results: {results['ensemble_baselines'][f'ensemble{num_external_in_ensemble}']}")

    # --- ADDITION END ---


    # Save the results. (Filename updated to scaling_low_lr.json)
    print("\nSaving results...")
    with open("./results_ablate/scaling_low_lr_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Experiment completed. Results saved to ./results_ablate/scaling_low_lr.json")

if __name__ == "__main__":
    main()