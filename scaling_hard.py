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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # Set external models to eval mode.
    for ext_model in external_models:
        ext_model.eval()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=len(external_models) if external_models else 1)
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            futures = [executor.submit(lambda m, inp: m.get_features(inp.to(device)), ext_model, inputs)
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
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        final_input_dim = 512 + (self.num_external * 512)
        self.final_layer = nn.Linear(final_input_dim, 10)
    
    def forward(self, x, external_features):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        if external_features is not None and len(external_features) > 0:
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
        for model in self.external_models:
            model.eval()
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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    auc = roc_auc_score(all_labels_binarized, all_probs, average='weighted', multi_class='ovr')
    return accuracy, auc, f1

def evaluate_enhanced(model, external_models, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Get external features if provided.
            if external_models is not None and len(external_models) > 0:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(external_models)) as executor:
                    futures = [executor.submit(lambda m: m.get_features(inputs), ext_model)
                               for ext_model in external_models]
                    external_features = [future.result() for future in futures]
            else:
                external_features = None
            outputs = model(inputs, external_features)
            _, predicted = torch.max(outputs.data, 1)
            probs = F.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    auc = roc_auc_score(all_labels_binarized, all_probs, average='weighted', multi_class='ovr')
    return accuracy, auc, f1


###############################################################################
# UTILITY FUNCTION FOR LOADING OR TRAINING MODELS
###############################################################################
def load_or_train(model, filepath, train_fn, *train_args):
    if os.path.exists(filepath):
        print(f"Found {filepath}. Loading model...")
        model.load_state_dict(torch.load(filepath))
    else:
        print(f"{filepath} not found. Training model...")
        train_fn(*train_args)
        torch.save(model.state_dict(), filepath)
    return model

###############################################################################
# MAIN
###############################################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./model_scaling_hard", exist_ok=True)
    os.makedirs("./results_ablate", exist_ok=True)
    
    train_size = 2000
    pretrain_epochs = 30
    enhanced_epochs = 30
    
    full_trainset = get_full_trainset()
    
    # Dataset splitting.
    base0_dataset = load_subset(full_trainset, 0, train_size)
    # External datasets: create 8 subsets, and now all are corrupted.
    external_datasets = []
    for i in range(8):
        start_idx = train_size + i * train_size
        subset = load_subset(full_trainset, start_idx, train_size)
        # All external subsets are wrapped with the corrupted dataset.
        subset = CorruptedDataset(subset)
        external_datasets.append(subset)
    
    # Base model.
    base_model = CNN().to(device)
    base_model_file = "./model_scaling_hard/base.pt"
    base_loader = DataLoader(base0_dataset, batch_size=64, shuffle=True, num_workers=2)
    base_model = load_or_train(base_model, base_model_file, train_model, base_model, base_loader, pretrain_epochs, device)
    
    # External models.
    external_models = []
    for i in range(8):
        model_ext = CNN().to(device)
        ext_file = f"./model_scaling_hard/external{i+1}.pt"
        loader_ext = DataLoader(external_datasets[i], batch_size=64, shuffle=True, num_workers=2)
        model_ext = load_or_train(model_ext, ext_file, train_model, model_ext, loader_ext, pretrain_epochs, device)
        external_models.append(model_ext)
    
    # Enhanced models.
    enhanced_models = []
    for i in range(8):
        model_enh = EnhancedScalingCNN(num_external=i+1).to(device)
        enh_file = f"./model_scaling_hard/enhanced{i+1}.pt"
        enh_loader = DataLoader(base0_dataset, batch_size=64, shuffle=True, num_workers=2)
        model_enh = load_or_train(model_enh, enh_file, train_enhanced, model_enh, external_models[:i+1], enh_loader, enhanced_epochs, device)
        enhanced_models.append(model_enh)
    
    # Naive models.
    naive_models = []
    for i in range(8):
        concat_dataset = FeatureDatasetConcat(base0_dataset, external_models[:i+1], device)
        concat_loader = DataLoader(concat_dataset, batch_size=64, shuffle=True, num_workers=2)
        sample_feat, _ = concat_dataset[0]
        input_dim = sample_feat.numel()
        model_naive = NaiveClassifier(input_dim=input_dim).to(device)
        naive_file = f"./model_scaling_hard/naive{i+1}.pt"
        model_naive = load_or_train(model_naive, naive_file, train_naive, model_naive, concat_loader, enhanced_epochs, device)
        naive_models.append(model_naive)
    
    # Evaluation on test data.
    test_chunks = load_test_sets()
    results = {}

    # Utility function to compute mean and standard deviation from a list of metric tuples.
    def aggregate_metrics(metrics_list):
        metrics_arr = np.array(metrics_list)
        mean_vals = np.mean(metrics_arr, axis=0)
        std_vals = np.std(metrics_arr, axis=0)
        return {
            "accuracy": {"mean": float(mean_vals[0]), "std": float(std_vals[0])},
            "auc": {"mean": float(mean_vals[1]), "std": float(std_vals[1])},
            "f1": {"mean": float(mean_vals[2]), "std": float(std_vals[2])},
        }
    
    # Evaluate Base model.
    print("Evaluating Base model...")
    base_metrics = []
    for test_subset in test_chunks:
        loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
        base_metrics.append(evaluate_classifier(base_model, loader, device))
    results["base_model"] = aggregate_metrics(base_metrics)
    
    # Evaluate External models.
    results["external_models"] = {}
    for i, model_ext in enumerate(external_models):
        ext_metrics = []
        for test_subset in test_chunks:
            loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
            ext_metrics.append(evaluate_classifier(model_ext, loader, device))
        results["external_models"][f"external{i+1}"] = aggregate_metrics(ext_metrics)
    
    # Evaluate Enhanced models.
    results["enhanced_models"] = {}
    for i, model_enh in enumerate(enhanced_models):
        enh_metrics = []
        for test_subset in test_chunks:
            loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
            enh_metrics.append(evaluate_enhanced(model_enh, external_models[:i+1], loader, device))
        results["enhanced_models"][f"enhanced{i+1}"] = aggregate_metrics(enh_metrics)
    
    # Evaluate Naive models.
    results["naive_models"] = {}
    for i, model_naive in enumerate(naive_models):
        naive_metrics = []
        for test_subset in test_chunks:
            concat_test_dataset = FeatureDatasetConcat(test_subset, external_models[:i+1], device)
            test_loader = DataLoader(concat_test_dataset, batch_size=64, shuffle=False, num_workers=2)
            naive_metrics.append(evaluate_naive(model_naive, test_loader, device))
        results["naive_models"][f"naive{i+1}"] = aggregate_metrics(naive_metrics)
    
    # Save the results.
    with open("./results_ablate/scaling_hard.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Experiment completed. Results saved to ./results_ablate/scaling_hard.json")
    
if __name__ == "__main__":
    main()