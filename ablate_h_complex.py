import os
import json
import copy
import random
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

class NoisyLabelSubset(Dataset):
    def __init__(self, subset: Subset, num_classes: int = 10, flip_ratio: float = 0.0, seed: int = 0):
        self.subset = subset
        self.num_classes = num_classes
        self.flip_ratio = float(flip_ratio)
        self.rng = np.random.RandomState(seed)
        self._precompute_labels()
    def _precompute_labels(self):
        n = len(self.subset)
        self.labels = []
        for i in range(n):
            _, true_y = self.subset[i]
            if self.rng.rand() < self.flip_ratio:
                wrong = self.rng.randint(0, self.num_classes - 1)
                if wrong >= true_y:
                    wrong += 1
                self.labels.append(int(wrong))
            else:
                self.labels.append(int(true_y))
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        x, _ = self.subset[idx]
        y = self.labels[idx]
        return x, y

def load_cifar_splits(seed: int) -> Tuple[Subset, Subset, torchvision.datasets.CIFAR10]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    idx = np.arange(len(trainset))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    pretrain_idx = idx[:10000]
    raw_idx = idx[10000:14000]
    return Subset(trainset, pretrain_idx), Subset(trainset, raw_idx), testset

class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 80, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(80, 160, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(160, 320, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(320, 640, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640, 640, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640, 768, 3, padding=1), nn.ReLU()
        )
        self.fc_layers = nn.Sequential(nn.Linear(768, 2560), nn.ReLU(), nn.Dropout(0.5), nn.Linear(2560, 10))
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        f = self.fc_layers[0](x)
        f = self.fc_layers[1](f)
        return f

class CNNh(nn.Module):
    def __init__(self, channels: List[int], fc_dim: int):
        super().__init__()
        convs = []
        in_c = 3
        for c in channels:
            convs += [nn.Conv2d(in_c, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)]
            in_c = c
        self.conv = nn.Sequential(*convs)
        assert sum(isinstance(m, nn.MaxPool2d) for m in self.conv) >= 3
        self.fc_feat = nn.Sequential(nn.Linear(in_c*4*4, fc_dim), nn.ReLU(), nn.Dropout(0.5))
        self.fc_head = nn.Linear(fc_dim, 10)
        self.feature_dim = fc_dim
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        f = self.fc_feat(x)
        return self.fc_head(f)
    def get_features(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_feat(x)

class EnhancedConcat(nn.Module):
    def __init__(self, base_cnn: CNNh, ext_dim: int = 2560):
        super().__init__()
        self.base = base_cnn
        self.final = nn.Linear(self.base.feature_dim + ext_dim, 10)
    def forward(self, x, ext_feats):
        f_local = self.base.get_features(x)
        fused = torch.cat([f_local, ext_feats], dim=1)
        return self.final(fused)

def train_supervised(model, loader, epochs, device, lr=0.01, momentum=0.9):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    for ep in range(epochs):
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"[Supervised Ep {ep+1}] loss={running/len(loader):.4f}")

def train_enhanced(model: EnhancedConcat, ext_model: BigCNN, loader, epochs, device, lr=0.01, momentum=0.9):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    ext_model.eval()
    for ep in range(epochs):
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                ext_f = ext_model.get_features(xb)
            optimizer.zero_grad()
            logits = model(xb, ext_f)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"[Enhanced Ep {ep+1}] loss={running/len(loader):.4f}")

@torch.no_grad()
def evaluate(model, loader, device, num_classes=10, enhanced=False, ext_model=None):
    model.eval()
    if ext_model is not None:
        ext_model.eval()
    total = 0
    correct = 0
    class_correct = [0]*num_classes
    class_total = [0]*num_classes
    all_probs = []
    all_labels = []
    all_preds = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if enhanced and ext_model is not None:
            ext_f = ext_model.get_features(xb)
            logits = model(xb, ext_f)
        else:
            logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
        for i in range(yb.size(0)):
            c = int(yb[i].item())
            class_total[c] += 1
            class_correct[c] += int(pred[i].item() == c)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(yb.cpu().numpy())
        all_preds.append(pred.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    acc = 100.0 * correct / total
    class_accs = [100.0*class_correct[c]/class_total[c] if class_total[c]>0 else 0.0 for c in range(num_classes)]
    min_cacc = float(min(class_accs))
    y_true = np.eye(num_classes)[all_labels]
    auc = float(roc_auc_score(y_true, all_probs, multi_class='ovr'))
    f1 = float(f1_score(all_labels, all_preds, average='weighted'))
    return acc, auc, f1, min_cacc

def get_h_grids() -> List[Dict[str, Any]]:
    return [
        dict(name="h0_tiny",   channels=[16, 32, 32],  fc=128),
        dict(name="h1_small",  channels=[32, 64, 64],  fc=256),
        dict(name="h2_base",   channels=[32, 64, 128], fc=512),
        dict(name="h3_large",  channels=[64, 128,128], fc=768),
        dict(name="h4_xlarge", channels=[64, 128,256], fc=1024),
    ]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./results_ablate", exist_ok=True)
    flip_ratios = [0.8, 0.0]
    pretrain_epochs = 60
    finetune_epochs = 30
    seeds = [GLOBAL_SEED + i for i in range(5)]
    results: Dict[str, Any] = {}
    pretrain_subset_fixed, _, _ = load_cifar_splits(seed=GLOBAL_SEED)
    _, _, testset = load_cifar_splits(seed=GLOBAL_SEED)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    for fr in flip_ratios:
        print(f"\n=== flip_ratio = {fr} ===")
        ckpt = f"./model_test10/noise_{fr}.pt"
        if os.path.exists(ckpt):
            external = torch.load(ckpt, map_location=device)
            print(f"Loaded external model from {ckpt}")
        else:
            noisy = NoisyLabelSubset(pretrain_subset_fixed, num_classes=10, flip_ratio=fr, seed=GLOBAL_SEED)
            pretrain_loader = DataLoader(noisy, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
            external = BigCNN().to(device)
            print("Pretraining external BigCNN...")
            train_supervised(external, pretrain_loader, pretrain_epochs, device)
            torch.save(external, ckpt)
            print(f"Saved external model to {ckpt}")
        ext_acc, ext_auc, ext_f1, ext_minc = evaluate(external, test_loader, device)
        print(f"External Model | Acc={ext_acc:.2f}% AUC={ext_auc:.4f} F1={ext_f1:.4f} MinCAcc={ext_minc:.2f}%")
        grids = get_h_grids()
        grid_results: Dict[str, Any] = {}
        for grid in grids:
            h_name = grid["name"]
            print(f"\n--- Grid: {h_name} ---")
            m_enh = {"acc": [], "auc": [], "f1": [], "min_cacc": []}
            m_scr = {"acc": [], "auc": [], "f1": [], "min_cacc": []}
            for run_seed in seeds:
                print(f"[seed={run_seed}]")
                _, raw_subset, _ = load_cifar_splits(seed=run_seed)
                raw_loader = DataLoader(raw_subset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
                base_h = CNNh(channels=grid["channels"], fc_dim=grid["fc"]).to(device)
                model_scr = copy.deepcopy(base_h)
                train_supervised(model_scr, raw_loader, finetune_epochs, device)
                acc, auc, f1, mc = evaluate(model_scr, test_loader, device)
                m_scr["acc"].append(acc); m_scr["auc"].append(auc); m_scr["f1"].append(f1); m_scr["min_cacc"].append(mc)
                model_enh = EnhancedConcat(base_cnn=copy.deepcopy(base_h)).to(device)
                for p in external.parameters():
                    p.requires_grad = False
                train_enhanced(model_enh, external, raw_loader, finetune_epochs, device)
                acc, auc, f1, mc = evaluate(model_enh, test_loader, device, enhanced=True, ext_model=external)
                m_enh["acc"].append(acc); m_enh["auc"].append(auc); m_enh["f1"].append(f1); m_enh["min_cacc"].append(mc)
            grid_results[h_name] = {
                "scratch": {
                    "acc_mean": float(np.mean(m_scr["acc"])), "acc_std": float(np.std(m_scr["acc"])),
                    "auc_mean": float(np.mean(m_scr["auc"])), "auc_std": float(np.std(m_scr["auc"])),
                    "f1_mean":  float(np.mean(m_scr["f1"])),  "f1_std":  float(np.std(m_scr["f1"])),
                    "min_cacc_mean": float(np.mean(m_scr["min_cacc"])), "min_cacc_std": float(np.std(m_scr["min_cacc"]))
                },
                "enhanced": {
                    "acc_mean": float(np.mean(m_enh["acc"])), "acc_std": float(np.std(m_enh["acc"])),
                    "auc_mean": float(np.mean(m_enh["auc"])), "auc_std": float(np.std(m_enh["auc"])),
                    "f1_mean":  float(np.mean(m_enh["f1"])),  "f1_std":  float(np.std(m_enh["f1"])),
                    "min_cacc_mean": float(np.mean(m_enh["min_cacc"])), "min_cacc_std": float(np.std(m_enh["min_cacc"]))
                }
            }
            s = grid_results[h_name]["scratch"]
            e = grid_results[h_name]["enhanced"]
            print(f"[{h_name}] scratch:  Acc={s['acc_mean']:.2f}±{s['acc_std']:.2f}  AUC={s['auc_mean']:.4f}±{s['auc_std']:.4f}  F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}  MinCAcc={s['min_cacc_mean']:.2f}±{s['min_cacc_std']:.2f}")
            print(f"[{h_name}] enhanced: Acc={e['acc_mean']:.2f}±{e['acc_std']:.2f}  AUC={e['auc_mean']:.4f}±{e['auc_std']:.4f}  F1={e['f1_mean']:.4f}±{e['f1_std']:.4f}  MinCAcc={e['min_cacc_mean']:.2f}±{e['min_cacc_std']:.2f}")
        results[str(fr)] = {
            "pretrained": {"acc": float(ext_acc), "auc": float(ext_auc), "f1": float(ext_f1), "min_cacc": float(ext_minc)},
            "grids": grid_results
        }
    save_path = "./results_ablate/h_complex.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll done. Results saved to: {save_path}")

if __name__ == "__main__":
    main()
