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

from model_def import CNN, EnhancedCNN, BaselineAdapter, BigCNN

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def load_data_split(seed):
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
    
    return pretrain_subset, raw_set, testset

def train_model(model, train_loader, num_epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}")

def train_linear_prob(model, train_loader, num_epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Linear Prob Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}")

def train_enhanced_model(model, train_loader, external_model, num_epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    external_model.eval()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                additional_features = external_model.get_features(inputs)
            optimizer.zero_grad()
            outputs = model(inputs, additional_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Enhanced Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}")

def train_distillation(student_model, teacher_model, train_loader, num_epochs, device, temperature=2.0, alpha=0.5):
    """
    Trains the student model using vanilla Hinton-style knowledge distillation.
    The loss is a combination of the standard cross-entropy loss on the true labels and
    a KL divergence loss between the softened outputs of the teacher and student.
    """
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_kd = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)
    student_model.train()
    teacher_model.eval()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
                soft_labels = torch.softmax(teacher_outputs / temperature, dim=1)
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            loss_ce = criterion_ce(student_outputs, labels)
            loss_kd = criterion_kd(torch.log_softmax(student_outputs / temperature, dim=1), soft_labels)
            loss = alpha * loss_ce + (1 - alpha) * (temperature ** 2) * loss_kd
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Distillation Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, device, enhanced=False, external_model=None):
    model.eval()
    if external_model is not None:
        external_model.eval()
    correct, total = 0, 0
    all_preds, all_labels, all_probs = [], [], []
    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if enhanced and external_model is not None:
                additional_features = external_model.get_features(inputs)
                outputs = model(inputs, additional_features)
            else:
                outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_correct[label] += int(pred == label)
                class_total[label] += 1
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    class_accs = [
        100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        for c in range(num_classes)
    ]
    min_class_acc = min(class_accs)
    y_true = np.eye(num_classes)[all_labels]
    auc = roc_auc_score(y_true, all_probs, multi_class='ovr')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, auc, f1, min_class_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Control epochs for different phases
    pretrain_epochs = 10  # Pretrain external model epochs
    train_epochs = 20     # Training epochs for all other models
    num_runs = 5
    flip_ratio = 0  # retained here only if you wish to add noise later
    
    pretrain_dataset, raw_set, test_dataset = load_data_split(seed=42)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=2)
    raw_loader = DataLoader(raw_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    print("\n=== Pretraining External Model (BigCNN) on 10k Samples ===")
    external_model = BigCNN().to(device)
    train_model(external_model, pretrain_loader, pretrain_epochs, device)
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
        _, raw_set_run, _ = load_data_split(seed=run_seed)
        run_loader = DataLoader(raw_set_run, batch_size=64, shuffle=True, num_workers=2)
        
        print("Training baseline model (CNN on raw set)...")
        baseline_model = CNN().to(device)
        train_model(baseline_model, run_loader, train_epochs, device)
        acc_b, auc_b, f1_b, min_cacc_b = evaluate_model(baseline_model, test_loader, device)
        metrics["baseline"]["acc"].append(acc_b)
        metrics["baseline"]["auc"].append(auc_b)
        metrics["baseline"]["f1"].append(f1_b)
        metrics["baseline"]["min_cacc"].append(min_cacc_b)
        
        print("Training linear probe model (fine-tuning external model's last layer)...")
        linear_model = copy.deepcopy(external_model)
        for param in linear_model.parameters():
            param.requires_grad = False
        for param in linear_model.fc_layers[-1].parameters():
            param.requires_grad = True
        train_linear_prob(linear_model, run_loader, train_epochs, device)
        acc_lp, auc_lp, f1_lp, min_cacc_lp = evaluate_model(linear_model, test_loader, device)
        metrics["linear_prob"]["acc"].append(acc_lp)
        metrics["linear_prob"]["auc"].append(auc_lp)
        metrics["linear_prob"]["f1"].append(f1_lp)
        metrics["linear_prob"]["min_cacc"].append(min_cacc_lp)
        
        print("Training enhanced model (concatenation)...")
        enhanced_concat_model = EnhancedCNN().to(device)
        train_enhanced_model(enhanced_concat_model, run_loader, external_model, train_epochs, device)
        acc_ec, auc_ec, f1_ec, min_cacc_ec = evaluate_model(enhanced_concat_model, test_loader, device, enhanced=True, external_model=external_model)
        metrics["enhanced_concat"]["acc"].append(acc_ec)
        metrics["enhanced_concat"]["auc"].append(auc_ec)
        metrics["enhanced_concat"]["f1"].append(f1_ec)
        metrics["enhanced_concat"]["min_cacc"].append(min_cacc_ec)
        
        print("Training baseline adapter model (external frozen with adapter)...")
        baseline_adapter_model = BaselineAdapter(copy.deepcopy(external_model)).to(device)
        train_model(baseline_adapter_model, run_loader, train_epochs, device)
        acc_ba, auc_ba, f1_ba, min_cacc_ba = evaluate_model(baseline_adapter_model, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(acc_ba)
        metrics["baseline_adapter"]["auc"].append(auc_ba)
        metrics["baseline_adapter"]["f1"].append(f1_ba)
        metrics["baseline_adapter"]["min_cacc"].append(min_cacc_ba)
        
        print("Training knowledge distillation model (CNN student with teacher external)...")
        student_model = CNN().to(device)
        train_distillation(student_model, external_model, run_loader, train_epochs, device, temperature=2.0, alpha=0.5)
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
    
    save_path = "./results/results_pretrain_{}_train_{}.json".format(pretrain_epochs, train_epochs)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(final_results, fp, indent=2)
    
    print(f"\nAll done. Final mean/std results saved to: {save_path}")

if __name__ == "__main__":
    main()
