# train_scaling_experiment.py
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import json

# Import the enhanced model and evaluation function.
from model_def_test10.model_def_scaling import EnhancedScalingCNN, evaluate_enhanced
# Import the simple CNN for external models.
from model_def_test10.model_def import CNN

# Set seeds for reproducibility.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def get_full_trainset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


def load_external_subset(trainset, start_idx, num_samples):
    indices = np.arange(start_idx, start_idx + num_samples)
    return Subset(trainset, indices)


def load_enhanced_subset(trainset, available_indices, seed, num_samples):
    rng = np.random.RandomState(seed)
    chosen = rng.choice(available_indices, size=num_samples, replace=False)
    return Subset(trainset, chosen)


def load_test_sets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    indices = np.arange(len(testset))
    copies = []
    for i in range(5):
        subset_indices = indices[i * 2000:(i + 1) * 2000]
        copies.append(Subset(testset, subset_indices))
    return copies


def train_model(model, dataloader, epochs, device):
    model.train()
    # Changed optimizer to SGD
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
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
    # Changed optimizer to SGD
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    for ext_model in external_models:
        ext_model.eval()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=len(external_models) if external_models else 1)
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            external_features_list = []
            if external_models:
                futures = [
                    executor.submit(lambda m, inp: m.get_features(inp.to(device)), ext_model, inputs)
                    for ext_model in external_models
                ]
                for future in futures:
                    external_features_list.append(future.result())
            outputs = model(inputs, external_features_list if external_features_list else None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    executor.shutdown()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Directory consistency: using "./model_scaling" for saving models
    os.makedirs("./model_scaling", exist_ok=True)
    os.makedirs("./results_test10", exist_ok=True)

    # ----- Parameters -----
    external_train_size = 2000
    enhanced_train_size = 2000
    pretrain_epochs = 30
    enhanced_epochs = 30
    num_external_models = 15
    # -----------------------

    full_trainset = get_full_trainset()
    total_train_samples = len(full_trainset)  # 50,000 for CIFAR-10

    external_total = external_train_size * num_external_models
    external_indices_all = np.arange(0, external_total)
    available_indices = np.setdiff1d(np.arange(total_train_samples), external_indices_all)

    # ---- Pretrain External Models ----
    external_models = []
    for idx in range(num_external_models):
        start_idx = idx * external_train_size
        print(f"Pretraining external model {idx+1} on samples {start_idx} to {start_idx + external_train_size - 1}")
        dataset = load_external_subset(full_trainset, start_idx, external_train_size)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
        model_ext = CNN().to(device)
        train_model(model_ext, loader, pretrain_epochs, device)
        model_path = f"./model_scaling/external{idx+1}.pt"
        torch.save(model_ext.state_dict(), model_path)
        external_models.append(model_ext)

    # ---- Train Enhanced Models ----
    enhanced_models = []
    for idx in range(num_external_models + 1):
        num_ext_to_use = idx  # 0, 1, ... 15
        print(f"Training enhanced model {idx} using {num_ext_to_use} external representations")
        dataset = load_enhanced_subset(full_trainset, available_indices, seed=1000 + idx, num_samples=enhanced_train_size)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
        model_enh = EnhancedScalingCNN(num_external=num_ext_to_use).to(device)
        used_external = external_models[:num_ext_to_use] if num_ext_to_use > 0 else []
        train_enhanced(model_enh, used_external, loader, enhanced_epochs, device)
        model_path = f"./model_scaling/enhanced{idx}.pt"
        torch.save(model_enh.state_dict(), model_path)
        enhanced_models.append((model_enh, used_external))

    # ---- Evaluation ----
    test_copies = load_test_sets()  # 5 copies of 2000 samples each.
    scaling_results = {}
    for idx, (model_enh, used_external) in enumerate(enhanced_models):
        print(f"Evaluating enhanced model {idx}")
        acc_list, time_list, auc_list, f1_list = [], [], [], []
        for test_subset in test_copies:
            test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
            acc, inf_time, auc, f1 = evaluate_enhanced(model_enh, used_external, test_loader, device)
            acc_list.append(acc)
            time_list.append(inf_time)
            auc_list.append(auc)
            f1_list.append(f1)
        scaling_results[f"enhanced{idx}"] = {
            "accuracy_mean": float(np.mean(acc_list)),
            "accuracy_std": float(np.std(acc_list)),
            "inference_time_mean": float(np.mean(time_list)),
            "inference_time_std": float(np.std(time_list)),
            "auc_mean": float(np.mean(auc_list)),
            "auc_std": float(np.std(auc_list)),
            "f1_mean": float(np.mean(f1_list)),
            "f1_std": float(np.std(f1_list))
        }

    with open("./results_test10/scaling.json", "w") as f:
        json.dump(scaling_results, f, indent=2)

    print("Scaling law experiment completed. Results saved to ./results_test10/scaling.json")


if __name__ == "__main__":
    main()
