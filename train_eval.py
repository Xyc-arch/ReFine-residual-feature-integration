import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, Subset, Dataset
import random




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


def train_distillation(student_model, teacher_model, train_loader, num_epochs, device, temperature=2.0, alpha=0.5):
    """
    Vanilla Hinton knowledge distillation:
      - teacher_model generates soft labels using temperature scaling.
      - student_model is trained using a combination of cross-entropy loss on true labels
        and a KL divergence loss on the softened outputs.
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
