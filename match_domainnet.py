#!/usr/bin/env python3
import os
import random
import copy
import warnings
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock, resnet18
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ─── Results storage ───────────────────────────────────────────────────────────
results = {
    'ResNet10-Baseline': {'acc': [], 'auc': [], 'f1': [], 'minc': []},
    'ResNet18-LP':       {'acc': [], 'auc': [], 'f1': [], 'minc': []},
    'ResNet10-Refine':   {'acc': [], 'auc': [], 'f1': [], 'minc': []},
    'ResNet10-Adapter':  {'acc': [], 'auc': [], 'f1': [], 'minc': []},
}

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT        = "./data/domainnet"
SOURCE_DOMAIN    = "clipart"
TARGET_DOMAIN    = "sketch"
PRETRAIN_N       = 3000
FINETUNE_N       = 1000
BATCH_SIZE       = 32
NUM_RUNS         = 5
LR               = 0.01
SEED             = 42
PRETRAIN_EPOCHS  = 20
FINETUNE_EPOCHS  = 20
TEACHER_PATH     = f"models/{SOURCE_DOMAIN}_teacher.pth"

# ─── Pick 40 classes ────────────────────────────────────────────────────────────
src_txt = os.path.join(DATA_ROOT, f"{SOURCE_DOMAIN}_train.txt")
all_classes = sorted({line.split()[0].split('/')[1] for line in open(src_txt)})
if len(all_classes) < 40:
    raise RuntimeError(f"Only found {len(all_classes)} classes, need ≥40")
CLASSES = all_classes[:40]

# ─── Transforms ────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# ─── Dataset ───────────────────────────────────────────────────────────────────
class DomainNetDataset(Dataset):
    def __init__(self, root, split_txt, classes, transform=None):
        self.transform = transform
        cls2idx = {c:i for i,c in enumerate(classes)}
        self.samples = []
        with open(split_txt) as f:
            for line in f:
                path,_ = line.split()
                _,rel = path.split('/',1)
                cls,fn = rel.split('/',1)
                if cls in cls2idx:
                    self.samples.append((os.path.join(root,cls,fn), cls2idx[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,l = self.samples[i]
        im = Image.open(p).convert("RGB")
        if self.transform: im = self.transform(im)
        return im, l

def subset(ds, n, seed):
    if n is None or n>=len(ds): return ds
    idxs = list(range(len(ds))); random.Random(seed).shuffle(idxs)
    return Subset(ds, idxs[:n])

# ─── Models ────────────────────────────────────────────────────────────────────
def ResNet10(num_classes):
    return ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes)

class Refine10(nn.Module):
    def __init__(self, teacher, num_classes):
        super().__init__()
        self.teacher = teacher
        for p in teacher.parameters(): p.requires_grad=False
        self.teacher_feat = nn.Sequential(*list(teacher.children())[:-1], nn.Flatten())
        feat_t = teacher.fc.in_features
        student = ResNet10(num_classes=num_classes)
        feat_s = student.fc.in_features
        student.fc = nn.Identity()
        self.student = student
        self.classifier = nn.Linear(feat_s + feat_t, num_classes)

    def forward(self, x):
        t = self.teacher_feat(x)
        s = self.student(x)
        return self.classifier(torch.cat([s, t], dim=1))

class Adapter10(nn.Module):
    def __init__(self, teacher, num_classes, bottleneck=128):
        super().__init__()
        self.teacher = teacher
        for p in teacher.parameters(): p.requires_grad=False
        self.teacher_feat = nn.Sequential(*list(teacher.children())[:-1], nn.Flatten())
        feat_t = teacher.fc.in_features
        self.adapter = nn.Sequential(
            nn.Linear(feat_t, bottleneck), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(bottleneck, feat_t), nn.ReLU(), nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(feat_t, num_classes)

    def forward(self, x):
        t = self.teacher_feat(x)
        return self.classifier(self.adapter(t))

# ─── Training/Eval ──────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval().to(device)
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            logits = model(x)
            p = F.softmax(logits, dim=1).cpu().numpy()
            ps.append(p.argmax(1)); probs.append(p); ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.vstack(probs)
    acc  = accuracy_score(y_true, y_pred)*100
    f1w  = f1_score(y_true, y_pred, average='weighted')*100
    yoh  = np.eye(len(CLASSES))[y_true]
    aucw = roc_auc_score(yoh, y_prob, average='weighted', multi_class='ovr')*100
    minc = min(accuracy_score(y_true[y_true==c], y_pred[y_true==c])
               for c in np.unique(y_true))*100
    return acc, aucw, f1w, minc

def train_model(model, tloader, vloader, device, epochs, name, lr=LR):
    opt  = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    model.to(device)
    print(f"\n>> {name} ({epochs} epochs)")
    for e in range(1, epochs+1):
        model.train()
        tot = 0
        for x,y in tloader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        print(f"  [{name}] Epoch {e}/{epochs}  loss={tot/len(tloader.dataset):.4f}", end='  ')
        acc, auc, f1, minc = evaluate(model, vloader, device)
        print(f"acc={acc:.2f}% auc={auc:.2f}% f1={f1:.2f}% minc={minc:.2f}%")
    print(f">> Done {name}\n")

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cudnn.deterministic=True; cudnn.benchmark=False

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # count samples
    src = DomainNetDataset(f"{DATA_ROOT}/{SOURCE_DOMAIN}",
                           f"{DATA_ROOT}/{SOURCE_DOMAIN}_train.txt",
                           CLASSES, tf)
    tgt = DomainNetDataset(f"{DATA_ROOT}/{TARGET_DOMAIN}",
                           f"{DATA_ROOT}/{TARGET_DOMAIN}_train.txt",
                           CLASSES, tf)
    print(f"Clipart samples: {len(src)}  Sketch samples: {len(tgt)}\n")

    # 1) Pretrain or load teacher on Clipart
    if os.path.exists(TEACHER_PATH):
        teacher = torch.load(TEACHER_PATH, map_location=dev)
        print(f"Loaded pretrained teacher from {TEACHER_PATH}")
    else:
        teacher = resnet18(num_classes=len(CLASSES), pretrained=False)
        tloader = DataLoader(subset(src, PRETRAIN_N, SEED),
                             batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        train_model(teacher, tloader, tloader, dev, PRETRAIN_EPOCHS, "ResNet18-Teacher")
        torch.save(teacher, TEACHER_PATH)
        print(f"Saved teacher to {TEACHER_PATH}")

    # 2) Fine-tune on Sketch with ResNet-10 variants
    test_ds = DomainNetDataset(f"{DATA_ROOT}/{TARGET_DOMAIN}",
                               f"{DATA_ROOT}/{TARGET_DOMAIN}_test.txt",
                               CLASSES, tf)
    vloader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4)

    for run in range(NUM_RUNS):
        seed_run = SEED + run
        print(f"==== RUN {run+1}/{NUM_RUNS} (seed={seed_run}) ====")
        train_ds = subset(tgt, FINETUNE_N, seed_run)
        tloader  = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)

        baseline = ResNet10(len(CLASSES))
        train_model(baseline, tloader, vloader, dev, FINETUNE_EPOCHS, "ResNet10-Baseline")

        lp = copy.deepcopy(teacher)
        for p in lp.parameters(): p.requires_grad=False
        lp.fc = nn.Linear(lp.fc.in_features, len(CLASSES))
        train_model(lp, tloader, vloader, dev, FINETUNE_EPOCHS, "ResNet18-LP")

        refine = Refine10(teacher, len(CLASSES))
        train_model(refine, tloader, vloader, dev, FINETUNE_EPOCHS, "ResNet10-Refine")

        adapter = Adapter10(teacher, len(CLASSES))
        train_model(adapter, tloader, vloader, dev, FINETUNE_EPOCHS, "ResNet10-Adapter")

        # record final metrics for this run
        for name, model in [
            ("ResNet10-Baseline", baseline),
            ("ResNet18-LP",       lp),
            ("ResNet10-Refine",   refine),
            ("ResNet10-Adapter",  adapter),
        ]:
            acc, auc, f1, minc = evaluate(model, vloader, dev)
            results[name]['acc'].append(acc)
            results[name]['auc'].append(auc)
            results[name]['f1'].append(f1)
            results[name]['minc'].append(minc)

    print("All done.")

    # aggregate & save means/stds
    summary = {}
    for name, vals in results.items():
        summary[name] = {
            'acc_mean':  float(np.mean(vals['acc'])),  'acc_std':  float(np.std(vals['acc'])),
            'auc_mean':  float(np.mean(vals['auc'])),  'auc_std':  float(np.std(vals['auc'])),
            'f1_mean':   float(np.mean(vals['f1'])),   'f1_std':   float(np.std(vals['f1'])),
            'minc_mean': float(np.mean(vals['minc'])), 'minc_std': float(np.std(vals['minc'])),
        }
    with open("results/domainnet.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved aggregated results to ./results/domainnet.json")

if __name__=="__main__":
    main()
