#!/usr/bin/env python3
import os
import json
import random
import copy
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from torchvision.datasets import USPS, MNIST

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ─── Model Definitions ─────────────────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64*4*4,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1,64*4*4)
        return self.fc_layers(x)
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(-1,64*4*4)
        return self.fc_layers[:-1](x)

class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64*4*4,512), nn.ReLU(), nn.Dropout(0.5)
        )
        self.final = nn.Linear(512 + 2560, 10)  # 2560 from BigCNN features
    def forward(self, x, ext):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        h = self.fc_layers(x)
        return self.final(torch.cat([h, ext], dim=1))

class BaselineAdapter(nn.Module):
    def __init__(self, teacher, bottleneck_dim=128):
        super().__init__()
        self.teacher = teacher
        for p in self.teacher.parameters(): p.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(2560, bottleneck_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(bottleneck_dim,2560), nn.ReLU(), nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(2560,10)
    def forward(self, x):
        f = self.teacher.get_features(x)
        return self.classifier(self.adapter(f))

class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,80,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(80,160,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(160,320,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(320,640,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640,640,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640,768,3,padding=1), nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(768,2560), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2560,10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        h = self.fc_layers[0](x)
        h = self.fc_layers[1](h)
        return h

class ResizeWrapper(nn.Module):
    def __init__(self, model, size=(32,32)):
        super().__init__()
        self.model = model
        self.size = size
    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
        return self.model(x)
    def get_features(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
        return self.model.get_features(x)


# ─── Training/Eval Routines ───────────────────────────────────────────────────

def train_model(model, loader, epochs, device, lr=0.01, name="model"):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    print(f"\n>> Starting training for {name} ({epochs} epochs)")
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"   → [{name}] Epoch {ep:02d}/{epochs}  Avg Loss: {avg:.4f}")
    print(f">> Finished training {name}\n")


def train_enhanced(model, loader, teacher, epochs, device, lr=0.01):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    print(f"\n>> Starting training for EnhancedCNN ({epochs} epochs)")
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                ext = teacher.get_features(x)
            logits = model(x, ext)
            loss = crit(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"   → [Enhanced] Epoch {ep:02d}/{epochs}  Avg Loss: {avg:.4f}")
    print(">> Finished training EnhancedCNN\n")


def evaluate(model, loader, device, enhanced=False, teacher=None):
    model.to(device).eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            if enhanced:
                ext = teacher.get_features(x)
                logits = model(x, ext)
            else:
                logits = model(x)
            p = F.softmax(logits, dim=1).cpu().numpy()
            cls = p.argmax(axis=1)
            ys.append(y.numpy())
            preds.append(cls)
            probs.append(p)
    ys    = np.concatenate(ys)
    preds = np.concatenate(preds)
    probs = np.vstack(probs)
    acc   = accuracy_score(ys, preds)*100
    f1w   = f1_score(ys, preds, average='weighted')*100
    y_onehot = np.eye(10)[ys]
    aucw  = roc_auc_score(y_onehot, probs, average='weighted', multi_class='ovr')*100
    minc  = min(accuracy_score(ys[ys==c], preds[ys==c]) for c in range(10)) * 100
    return acc, aucw, f1w, minc


# ─── Data Helpers ──────────────────────────────────────────────────────────────

DATA_ROOT    = "./data"
PRETRAIN_N   = 5000
FINETUNE_N   = 100
BATCH_SIZE   = 32

tf = transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

def ensure_dataset(name, cls, split):
    """Check if split exists; if not, download it."""
    print(f"--- Checking {name} [{split}] ---")
    try:
        cls(root=DATA_ROOT, train=(split=="train"), download=False, transform=tf)
        print(f"{name} [{split}] already present.")
    except Exception:
        print(f"{name} [{split}] not found. Downloading now…")
        try:
            cls(root=DATA_ROOT, train=(split=="train"), download=True, transform=tf)
            print(f"{name} [{split}] downloaded successfully.")
        except Exception:
            print(f"Error downloading {name} [{split}]:")
            traceback.print_exc()

def subset(ds, n, seed):
    if n is None or n >= len(ds): return ds
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return Subset(ds, idx[:n])


# ─── Main Experiment ──────────────────────────────────────────────────────────

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1) Download / verify datasets
    print("\n===== DATASET SETUP =====")
    ensure_dataset("USPS", USPS, "train")
    ensure_dataset("USPS", USPS, "test")
    ensure_dataset("MNIST", MNIST, "train")
    ensure_dataset("MNIST", MNIST, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Teacher pretraining
    teacher_path = "models/usps_teacher.pt"
    if os.path.exists(teacher_path):
        print(f"\nLoading pretrained teacher from {teacher_path}")
        teacher = torch.load(teacher_path, map_location=device)
    else:
        print("\n===== TEACHER PRETRAINING ON USPS =====")
        teacher = BigCNN().to(device)
        usps_tr = subset(USPS(DATA_ROOT, train=True, download=False, transform=tf), PRETRAIN_N, seed=42)
        loader  = DataLoader(usps_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        train_model(teacher, loader, epochs=60, device=device, lr=0.01, name="BigCNN-USPS")
        torch.save(teacher, teacher_path)
        print(f"Saved teacher model to {teacher_path}")
    teacher = ResizeWrapper(teacher, (32,32)).to(device)

    # 3) Prepare MNIST test
    print("\n===== MNIST TEST SETUP =====")
    mnist_test = MNIST(DATA_ROOT, train=False, download=False, transform=tf)
    print(f"MNIST test examples: {len(mnist_test)}")
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    metrics = {
        "baseline": {"acc":[], "auc":[], "f1":[], "minc":[]},
        "linear":   {"acc":[], "auc":[], "f1":[], "minc":[]},
        "enhanced":{"acc":[], "auc":[], "f1":[], "minc":[]},
        "adapter":  {"acc":[], "auc":[], "f1":[], "minc":[]},
    }

    # 4) Multiple runs
    for run in range(5):
        run_seed = 42 + run
        print(f"\n===== RUN {run+1}/5 (seed={run_seed}) =====")
        mnist_tr = subset(MNIST(DATA_ROOT, train=True, download=False, transform=tf), FINETUNE_N, seed=run_seed)
        print(f"MNIST train subset size: {len(mnist_tr)}")
        train_loader = DataLoader(mnist_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # a) baseline CNN
        m1 = CNN().to(device)
        train_model(m1, train_loader, 30, device, name="Baseline-CNN")
        res1 = evaluate(m1, test_loader, device)
        print(f"→ Baseline: Acc={res1[0]:.2f}%, AUC={res1[1]:.2f}%, F1={res1[2]:.2f}%, MinC={res1[3]:.2f}%")
        for k,v in zip(["acc","auc","f1","minc"], res1): metrics["baseline"][k].append(v)

        # b) linear probe
        base = copy.deepcopy(teacher.model if isinstance(teacher, ResizeWrapper) else teacher)
        for p in base.parameters(): p.requires_grad=False
        base.fc_layers[-1] = nn.Linear(2560,10).to(device)
        for p in base.fc_layers[-1].parameters(): p.requires_grad=True
        m2 = ResizeWrapper(base,(32,32)).to(device)
        train_model(m2, train_loader, 30, device, name="LinearProbe")
        res2 = evaluate(m2, test_loader, device)
        print(f"→ LinearProbe: Acc={res2[0]:.2f}%, AUC={res2[1]:.2f}%, F1={res2[2]:.2f}%, MinC={res2[3]:.2f}%")
        for k,v in zip(["acc","auc","f1","minc"], res2): metrics["linear"][k].append(v)

        # c) enhanced concat
        m3 = EnhancedCNN().to(device)
        train_enhanced(m3, train_loader, teacher, 30, device)
        res3 = evaluate(m3, test_loader, device, enhanced=True, teacher=teacher)
        print(f"→ Enhanced: Acc={res3[0]:.2f}%, AUC={res3[1]:.2f}%, F1={res3[2]:.2f}%, MinC={res3[3]:.2f}%")
        for k,v in zip(["acc","auc","f1","minc"], res3): metrics["enhanced"][k].append(v)

        # d) baseline adapter
        m4 = BaselineAdapter(copy.deepcopy(teacher)).to(device)
        train_model(m4, train_loader, 30, device, name="Adapter")
        res4 = evaluate(m4, test_loader, device)
        print(f"→ Adapter: Acc={res4[0]:.2f}%, AUC={res4[1]:.2f}%, F1={res4[2]:.2f}%, MinC={res4[3]:.2f}%")
        for k,v in zip(["acc","auc","f1","minc"], res4): metrics["adapter"][k].append(v)

    # 5) Aggregate & save
    print("\n===== AGGREGATING RESULTS =====")
    output = {}
    for name, vals in metrics.items():
        summary = {}
        for k, arr in vals.items():
            summary[f"{k}_mean"] = float(np.mean(arr))
            summary[f"{k}_std"]  = float(np.std(arr))
        output[name] = summary

    result_file = "results/usps_mnist_results.json"
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"All done! Final results written to {result_file}")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
