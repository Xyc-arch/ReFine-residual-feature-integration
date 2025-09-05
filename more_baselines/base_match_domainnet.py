#!/usr/bin/env python3
import os
import json
import copy
import random
import warnings

import torch
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

# ──────────────────────────────────────────────────────────────────────────────
# Config (match your match_domainnet.py)
# ──────────────────────────────────────────────────────────────────────────────
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

# Save results alongside other *match* baselines
RESULTS_PATH     = "results_match_base/base_match_domainnet.json"

# ──────────────────────────────────────────────────────────────────────────────
# Class subset (40) as in match_domainnet.py
# ──────────────────────────────────────────────────────────────────────────────
src_txt = os.path.join(DATA_ROOT, f"{SOURCE_DOMAIN}_train.txt")
all_classes = sorted({line.split()[0].split('/')[1] for line in open(src_txt)})
if len(all_classes) < 40:
    raise RuntimeError(f"Only found {len(all_classes)} classes, need ≥40")
CLASSES = all_classes[:40]
NUM_CLASSES = len(CLASSES)

# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
def ResNet10(num_classes):
    return ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes)

# ──────────────────────────────────────────────────────────────────────────────
# LoRA + GRL building blocks (LoRA on teacher's final FC)
# ──────────────────────────────────────────────────────────────────────────────
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class LoRALinear(nn.Module):
    """
    Parallel LoRA on a Linear: y = (W0 x + b0) + B(Ax) * (alpha/r)
    Only A,B are trainable; W0,b0 are frozen copies of the original head.
    """
    def __init__(self, in_features, out_features, r=8, alpha=16.0):
        super().__init__()
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        # freeze base
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        # init LoRA path
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

class Discriminator(nn.Module):
    """
    Label-conditioned discriminator operating on LOGITS (dim = NUM_CLASSES).
    Modest capacity to avoid overpowering the LoRA head.
    """
    def __init__(self, feat_dim, num_classes, emb_dim=8, hidden=64, dropout=0.2):
        super().__init__()
        self.nil_idx = num_classes
        self.emb = nn.Embedding(num_classes + 1, emb_dim)  # +1 for NIL
        self.net = nn.Sequential(
            nn.Linear(feat_dim + emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat, label):
        e = self.emb(label)
        x = torch.cat([feat, e], dim=1)
        return self.net(x)

# ──────────────────────────────────────────────────────────────────────────────
# Eval (same protocol as match_domainnet.py)
# ──────────────────────────────────────────────────────────────────────────────
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
    yoh  = np.eye(NUM_CLASSES)[y_true]
    aucw = roc_auc_score(yoh, y_prob, average='weighted', multi_class='ovr')*100
    # per-class minimum accuracy among present classes
    minc = min(accuracy_score(y_true[y_true==c], y_pred[y_true==c])
               for c in np.unique(y_true))*100
    return acc, aucw, f1w, minc

# ──────────────────────────────────────────────────────────────────────────────
# Train helpers (teacher pretrain & generic SGD loop)
# ──────────────────────────────────────────────────────────────────────────────
def train_supervised(model, tloader, vloader, device, epochs, name, lr=LR):
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
    return model

# ──────────────────────────────────────────────────────────────────────────────
# LoRA head-only (backbone frozen) on teacher
# ──────────────────────────────────────────────────────────────────────────────
def train_lora_head(teacher_in, loader, device, epochs=FINETUNE_EPOCHS, r=8, alpha=16.0):
    model = copy.deepcopy(teacher_in).to(device)
    # freeze all
    for p in model.parameters():
        p.requires_grad = False
    # swap final FC to LoRA head
    last = model.fc
    in_dim, out_dim = last.in_features, last.out_features
    lora_head = LoRALinear(in_dim, out_dim, r=r, alpha=alpha).to(device)
    with torch.no_grad():
        lora_head.linear.weight.copy_(last.weight)
        if last.bias is not None:
            lora_head.linear.bias.copy_(last.bias)
    model.fc = lora_head

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    print(f"\n>> LoRA(head-only) for {epochs} epochs")
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0)
        print(f"   [LoRA] Ep {ep:02d}/{epochs}  Loss={total/len(loader.dataset):.4f}")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# DANN-Gate (LoRA on head; backbone frozen; discriminator over logits)
# Uses *paired* source/target mini-batches of equal size (zip)
# ──────────────────────────────────────────────────────────────────────────────
def train_dann_gate_head(teacher_in, src_loader, tgt_loader, device,
                         epochs=FINETUNE_EPOCHS, r=8, alpha=16.0, max_grl=1.0, mu=1.0):
    import math

    model = copy.deepcopy(teacher_in).to(device)
    # freeze all
    for p in model.parameters():
        p.requires_grad = False
    # LoRA on classifier head
    last = model.fc
    in_dim, out_dim = last.in_features, last.out_features
    lora_head = LoRALinear(in_dim, out_dim, r=r, alpha=alpha).to(device)
    with torch.no_grad():
        lora_head.linear.weight.copy_(last.weight)
        if last.bias is not None:
            lora_head.linear.bias.copy_(last.bias)
    model.fc = lora_head
    num_classes = out_dim  # equals NUM_CLASSES

    # discriminator on LOGITS
    D = Discriminator(feat_dim=num_classes, num_classes=num_classes,
                      emb_dim=8, hidden=64, dropout=0.2).to(device)

    opt_FC = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    opt_D  = torch.optim.Adam(D.parameters(), lr=5e-4, weight_decay=1e-4)

    ce_mean = nn.CrossEntropyLoss()
    ce  = nn.CrossEntropyLoss(reduction='none')
    bce = nn.BCELoss()
    lambda_gate = 1.0

    print(f"\n>> DANN-Gate (LoRA on head r={r} α={alpha}, backbone frozen) for {epochs} epochs")
    for ep in range(1, epochs+1):
        p_s = (ep-1) / max(1,(epochs-1))
        lam_grl = (2.0/(1.0+np.exp(-10*p_s)) - 1.0) * max_grl

        model.train(); D.train()
        total = 0.0

        for (xs, ys), (xt, yt) in zip(src_loader, tgt_loader):
            xs, ys = xs.to(device), ys.to(device)
            xt, yt = xt.to(device), yt.to(device)

            # forward to logits
            logits_s = model(xs)  # [B, C]
            logits_t = model(xt)  # [B, C]

            # gated source CE + target CE
            ls = ce(logits_s, ys)
            with torch.no_grad():
                d_sy = D(logits_s, ys).clamp_(1e-4, 1-1e-4)  # p(target | logits_s)
                ws   = d_sy / (1 - d_sy)
            loss_src = (lambda_gate * ws.view(-1) * ls).mean()
            loss_tgt = ce_mean(logits_t, yt)
            loss_cls = loss_src + loss_tgt

            # adversarial joint (known labels)
            ds_joint = D(grad_reverse(logits_s, lam_grl), ys)   # want 0 (source)
            dt_joint = D(grad_reverse(logits_t, lam_grl), yt)   # want 1 (target)
            loss_joint = 0.5*( bce(ds_joint, torch.zeros_like(ds_joint)) +
                               bce(dt_joint, torch.ones_like(dt_joint)) )

            # adversarial marginal (NIL)
            nil_s = torch.full((xs.size(0),), num_classes, device=device, dtype=torch.long)
            nil_t = torch.full((xt.size(0),), num_classes, device=device, dtype=torch.long)
            ds_marg = D(grad_reverse(logits_s, lam_grl), nil_s) # want 0
            dt_marg = D(grad_reverse(logits_t, lam_grl), nil_t) # want 1
            loss_marg = 0.5*( bce(ds_marg, torch.zeros_like(ds_marg)) +
                              bce(dt_marg, torch.ones_like(dt_marg)) )

            loss_adv = loss_joint + loss_marg

            # update LoRA head
            opt_FC.zero_grad()
            (loss_cls + mu*loss_adv).backward()
            opt_FC.step()

            # train discriminator on detached logits
            ls_det = logits_s.detach(); lt_det = logits_t.detach()
            ds_det   = D(ls_det, ys)
            dt_det   = D(lt_det, yt)
            ds_m_det = D(ls_det, nil_s)
            dt_m_det = D(lt_det, nil_t)
            loss_D = 0.25*( bce(ds_det, torch.zeros_like(ds_det)) +
                            bce(dt_det, torch.zeros_like(dt_det)) +
                            bce(ds_m_det, torch.zeros_like(ds_m_det)) +
                            bce(dt_m_det, torch.zeros_like(dt_m_det)) )
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            total += (loss_cls.item() + loss_adv.item() + loss_D.item())

        print(f"   [DANN-Gate-Head] Ep {ep:02d}/{epochs}  λ_grl={lam_grl:.3f}  Loss={total/max(1,len(src_loader)):.4f}")

    return model

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results_match_base", exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Count samples & build datasets/loaders
    src_ds_full = DomainNetDataset(f"{DATA_ROOT}/{SOURCE_DOMAIN}",
                                   f"{DATA_ROOT}/{SOURCE_DOMAIN}_train.txt",
                                   CLASSES, tf)
    tgt_ds_full = DomainNetDataset(f"{DATA_ROOT}/{TARGET_DOMAIN}",
                                   f"{DATA_ROOT}/{TARGET_DOMAIN}_train.txt",
                                   CLASSES, tf)
    print(f"{SOURCE_DOMAIN.capitalize()} samples: {len(src_ds_full)}  "
          f"{TARGET_DOMAIN.capitalize()} samples: {len(tgt_ds_full)}\n")

    # 1) Pretrain/load teacher on SOURCE domain
    if os.path.exists(TEACHER_PATH):
        teacher = torch.load(TEACHER_PATH, map_location=dev)
        print(f"Loaded pretrained teacher from {TEACHER_PATH}")
    else:
        teacher = resnet18(num_classes=NUM_CLASSES, pretrained=False)
        src_pre_ds = subset(src_ds_full, PRETRAIN_N, SEED)
        pre_loader = DataLoader(src_pre_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        # simple supervised pretraining on source
        teacher = train_supervised(teacher, pre_loader, pre_loader, dev, PRETRAIN_EPOCHS, "ResNet18-Teacher", lr=LR)
        torch.save(teacher, TEACHER_PATH)
        print(f"Saved teacher to {TEACHER_PATH}")

    # Constant TARGET test set (official test split)
    test_ds = DomainNetDataset(f"{DATA_ROOT}/{TARGET_DOMAIN}",
                               f"{DATA_ROOT}/{TARGET_DOMAIN}_test.txt",
                               CLASSES, tf)
    vloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    results = {
        "LoRA-Head": {"acc": [], "auc": [], "f1": [], "minc": []},
        "DANN-Gate": {"acc": [], "auc": [], "f1": [], "minc": []},
    }

    # 2) Runs with different target subsets; for DANN use matched-size source subset
    for run in range(NUM_RUNS):
        seed_run = SEED + run
        print(f"==== RUN {run+1}/{NUM_RUNS} (seed={seed_run}) ====")
        tgt_train_ds = subset(tgt_ds_full, FINETUNE_N, seed_run)
        src_train_ds = subset(src_ds_full, FINETUNE_N, seed_run)  # balanced for zip()

        tgt_loader = DataLoader(tgt_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        src_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        # LoRA head-only fine-tune on TARGET only
        lora_model = train_lora_head(teacher, tgt_loader, dev, epochs=FINETUNE_EPOCHS, r=8, alpha=16.0)
        acc, auc, f1, minc = evaluate(lora_model, vloader, dev)
        results["LoRA-Head"]["acc"].append(acc)
        results["LoRA-Head"]["auc"].append(auc)
        results["LoRA-Head"]["f1"].append(f1)
        results["LoRA-Head"]["minc"].append(minc)
        print(f"LoRA(head): Acc={acc:.2f}% AUC={auc:.2f}% F1={f1:.2f}% MinC={minc:.2f}%")

        # DANN-Gate: source+target with LoRA head; backbone frozen
        dann_model = train_dann_gate_head(teacher, src_loader, tgt_loader, dev,
                                          epochs=FINETUNE_EPOCHS, r=8, alpha=16.0, max_grl=1.0, mu=1.0)
        acc, auc, f1, minc = evaluate(dann_model, vloader, dev)
        results["DANN-Gate"]["acc"].append(acc)
        results["DANN-Gate"]["auc"].append(auc)
        results["DANN-Gate"]["f1"].append(f1)
        results["DANN-Gate"]["minc"].append(minc)
        print(f"DANN-Gate:  Acc={acc:.2f}% AUC={auc:.2f}% F1={f1:.2f}% MinC={minc:.2f}%")

    # 3) Aggregate & save
    summary = {}
    for name, vals in results.items():
        summary[name] = {
            'acc_mean':  float(np.mean(vals['acc'])),  'acc_std':  float(np.std(vals['acc'])),
            'auc_mean':  float(np.mean(vals['auc'])),  'auc_std':  float(np.std(vals['auc'])),
            'f1_mean':   float(np.mean(vals['f1'])),   'f1_std':   float(np.std(vals['f1'])),
            'minc_mean': float(np.mean(vals['minc'])), 'minc_std': float(np.std(vals['minc'])),
        }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved aggregated results to {RESULTS_PATH}")

if __name__=="__main__":
    main()
