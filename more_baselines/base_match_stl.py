#!/usr/bin/env python3
import os
import sys
import json
import copy
import random
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ──────────────────────────────────────────────────────────────────────────────
# Ensure repo root (parent of 'more_baselines') is on sys.path so imports match match_stl.py
# ──────────────────────────────────────────────────────────────────────────────
FILE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../refine/more_baselines
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))  # .../refine
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ──────────────────────────────────────────────────────────────────────────────
# External: teacher model + STL evaluator (same import head as match_stl.py)
# ──────────────────────────────────────────────────────────────────────────────
from model_def_test100.model_def10 import BigCNN
from train_eval_test100 import evaluate_model_stl  # (acc, auc, f1, min_cacc)

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility & Control
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

PRETRAIN_SIZE      = 10000   # CIFAR-10 subset for teacher pretraining
RAW_SIZE           = 2000    # STL-10 train subset per run
NUM_EPOCHS         = 30
NUM_EPOCH_TEACHER  = 60
NUM_RUNS           = 5
BATCH_SIZE         = 32

DATA_ROOT          = os.path.join(REPO_ROOT, "data")
TEACHER_PATH       = os.path.join(REPO_ROOT, "model_test100", "match.pt")
RESULTS_PATH = os.path.join(REPO_ROOT, "results_match_base", "base_match_stl.json")

# ──────────────────────────────────────────────────────────────────────────────
# Utility: resize wrapper so the teacher always sees 32×32 inputs
# ──────────────────────────────────────────────────────────────────────────────
class ResizeWrapper(nn.Module):
    def __init__(self, model, target_size=(32, 32)):
        super().__init__()
        self.model = model
        self.target_size = target_size
    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return self.model(x)
    def get_features(self, x):
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return self.model.get_features(x)

def _unwrap_resize(m):
    """Return (base_model, was_wrapped, target_size_if_wrapped)."""
    if isinstance(m, ResizeWrapper):
        return m.model, True, m.target_size
    return m, False, (32, 32)

# ──────────────────────────────────────────────────────────────────────────────
# LoRA + GRL building blocks
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
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        # freeze the base linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        # init LoRA path
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

# Label-conditioned discriminator that operates on *logits* (dimension = num_classes)
class Discriminator(nn.Module):
    def __init__(self, feat_dim, num_classes, emb_dim=8, hidden=64, dropout=0.2):
        super().__init__()
        self.nil_idx = num_classes  # NIL label index
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
        # feat: [B, C] (logits), label in [0..num_classes] with num_classes meaning NIL
        e = self.emb(label)
        x = torch.cat([feat, e], dim=1)
        return self.net(x)

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers (CIFAR-10 for teacher; STL-10 for adaptation)
# ──────────────────────────────────────────────────────────────────────────────
def download_stl():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    torchvision.datasets.STL10(root=DATA_ROOT, split="train", download=True, transform=tf)
    torchvision.datasets.STL10(root=DATA_ROOT, split="test",  download=True, transform=tf)

def load_teacher_data(pretrain_size=PRETRAIN_SIZE, seed=42):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=tf)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return Subset(ds, idx[:pretrain_size])

def load_stl(split="train", raw_size=None, seed=42):
    tf = transforms.Compose([
        transforms.Resize((32, 32)),  # downsample to the teacher’s input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds = torchvision.datasets.STL10(root=DATA_ROOT, split=split, download=False, transform=tf)
    if split == "train" and raw_size is not None and raw_size < len(ds):
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        ds = Subset(ds, idx[:raw_size])
    return ds

# ──────────────────────────────────────────────────────────────────────────────
# Training: LoRA (head-only, backbone frozen)
# ──────────────────────────────────────────────────────────────────────────────
def train_lora_head(model_in, loader, device, epochs=30, r=8, alpha=16.0):
    base, was_wrapped, tgt_size = _unwrap_resize(model_in)
    model = copy.deepcopy(base).to(device)

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # swap last FC with LoRA
    last = model.fc_layers[-1]
    in_dim, out_dim = last.in_features, last.out_features
    lora_head = LoRALinear(in_dim, out_dim, r=r, alpha=alpha).to(device)
    with torch.no_grad():
        lora_head.linear.weight.copy_(last.weight)
        if last.bias is not None:
            lora_head.linear.bias.copy_(last.bias)
    model.fc_layers[-1] = lora_head

    # rewrap if original was wrapped
    if was_wrapped:
        model = ResizeWrapper(model, target_size=tgt_size).to(device)

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
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
            total += loss.item()
        print(f"   [LoRA] Ep {ep:02d}/{epochs}  Loss={total/max(1,len(loader)):.4f}")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Training: DANN-Gate with LoRA on last FC; discriminator over logits
# ──────────────────────────────────────────────────────────────────────────────
def train_dann_gate_head(model_in, loader, device, epochs=30, r=8, alpha=16.0, max_grl=1.0, mu=1.0):
    import math

    base, was_wrapped, tgt_size = _unwrap_resize(model_in)
    model = copy.deepcopy(base).to(device)

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # LoRA on last FC
    last = model.fc_layers[-1]
    in_dim, out_dim = last.in_features, last.out_features
    lora_head = LoRALinear(in_dim, out_dim, r=r, alpha=alpha).to(device)
    with torch.no_grad():
        lora_head.linear.weight.copy_(last.weight)
        if last.bias is not None:
            lora_head.linear.bias.copy_(last.bias)
    model.fc_layers[-1] = lora_head
    num_classes = out_dim

    # rewrap so inputs get resized like the teacher did
    if was_wrapped:
        model = ResizeWrapper(model, target_size=tgt_size).to(device)

    # discriminator sees logits
    D = Discriminator(feat_dim=num_classes, num_classes=num_classes,
                      emb_dim=8, hidden=64, dropout=0.2).to(device)

    opt_FC = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    opt_D  = optim.Adam(D.parameters(), lr=5e-4, weight_decay=1e-4)

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

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            B = x.size(0)
            if B < 2: continue
            if B % 2 == 1:  # ensure even split
                x, y = x[:B-1], y[:B-1]
                B -= 1
            half = B // 2

            xs, xt = x[:half], x[half:]
            ys, yt = y[:half], y[half:]

            # logits (depend only on LoRA head)
            logits_s = model(xs)
            logits_t = model(xt)

            # gated CE
            ls = ce(logits_s, ys)
            with torch.no_grad():
                d_sy = D(logits_s, ys).clamp_(1e-4, 1-1e-4)  # p(target|logits_s)
                ws   = d_sy / (1 - d_sy)
            loss_src = (lambda_gate * ws.view(-1) * ls).mean()
            loss_tgt = ce_mean(logits_t, yt)
            loss_cls = loss_src + loss_tgt

            # adversarial joint (labels known)
            ds_joint = D(grad_reverse(logits_s, lam_grl), ys)
            dt_joint = D(grad_reverse(logits_t, lam_grl), yt)
            loss_joint = 0.5*( bce(ds_joint, torch.zeros_like(ds_joint)) +
                               bce(dt_joint, torch.ones_like(dt_joint)) )

            # adversarial marginal (NIL)
            nil_s = torch.full((half,), num_classes, device=device, dtype=torch.long)
            nil_t = torch.full((half,), num_classes, device=device, dtype=torch.long)
            ds_marg = D(grad_reverse(logits_s, lam_grl), nil_s)
            dt_marg = D(grad_reverse(logits_t, lam_grl), nil_t)
            loss_marg = 0.5*( bce(ds_marg, torch.zeros_like(ds_marg)) +
                              bce(dt_marg, torch.zeros_like(dt_marg)) )
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

        print(f"   [DANN-Gate-Head] Ep {ep:02d}/{epochs}  λ_grl={lam_grl:.3f}  Loss={total/max(1,len(loader)):.4f}")

    return model

# ──────────────────────────────────────────────────────────────────────────────
# Main experiment runner (STL-10)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEACHER_PATH), exist_ok=True)
    os.makedirs(DATA_ROOT, exist_ok=True)

    print("Checking STL-10 dataset…")
    download_stl()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher: pretrain (or load) on CIFAR-10
    if os.path.exists(TEACHER_PATH):
        teacher_model = torch.load(TEACHER_PATH, map_location=device).to(device)
        print(f"Loaded teacher from {TEACHER_PATH}")
    else:
        print("\n>> Pretraining teacher (BigCNN) on CIFAR-10 subset")
        teacher_train = load_teacher_data(pretrain_size=PRETRAIN_SIZE, seed=42)
        teacher_loader = DataLoader(teacher_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        teacher_model = BigCNN().to(device)
        # simple supervised pretraining
        opt = optim.SGD(teacher_model.parameters(), lr=0.01, momentum=0.9)
        ce = nn.CrossEntropyLoss()
        for ep in range(1, NUM_EPOCH_TEACHER+1):
            teacher_model.train()
            total = 0.0
            for x,y in teacher_loader:
                x,y = x.to(device), y.to(device)
                logits = teacher_model(x)
                loss = ce(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            print(f"   [Teacher] Ep {ep:02d}/{NUM_EPOCH_TEACHER}  Loss={total/max(1,len(teacher_loader)):.4f}")
        torch.save(teacher_model, TEACHER_PATH)
        print(f"Saved teacher to {TEACHER_PATH}")

    # Wrap teacher so it always sees 32×32 inputs
    teacher_stl = ResizeWrapper(teacher_model, target_size=(32,32)).to(device)

    # Constant STL test set
    stl_test = load_stl(split="test")
    test_loader = DataLoader(stl_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    results = {
        "lora_head":   {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        "dann_gate":   {"acc": [], "auc": [], "f1": [], "min_cacc": []},
    }

    for run in range(NUM_RUNS):
        run_seed = 42 + run
        print(f"\n===== RUN {run+1}/{NUM_RUNS} (seed={run_seed}) =====")
        stl_train = load_stl(split="train", raw_size=RAW_SIZE, seed=run_seed)
        train_loader = DataLoader(stl_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # LoRA (head-only)
        lora_model = train_lora_head(teacher_stl, train_loader, device, epochs=NUM_EPOCHS, r=8, alpha=16.0)
        acc, auc, f1, min_cacc = evaluate_model_stl(lora_model, test_loader, device)
        results["lora_head"]["acc"].append(acc)
        results["lora_head"]["auc"].append(auc)
        results["lora_head"]["f1"].append(f1)
        results["lora_head"]["min_cacc"].append(min_cacc)
        print(f"→ LoRA(head): Acc={acc:.2f} AUC={auc:.2f} F1={f1:.2f} MinC={min_cacc:.2f}")

        # DANN-Gate (LoRA on head, logits to D)
        dann_model = train_dann_gate_head(teacher_stl, train_loader, device,
                                          epochs=NUM_EPOCHS, r=8, alpha=16.0, max_grl=1.0, mu=1.0)
        acc, auc, f1, min_cacc = evaluate_model_stl(dann_model, test_loader, device)
        results["dann_gate"]["acc"].append(acc)
        results["dann_gate"]["auc"].append(auc)
        results["dann_gate"]["f1"].append(f1)
        results["dann_gate"]["min_cacc"].append(min_cacc)
        print(f"→ DANN-Gate:  Acc={acc:.2f} AUC={auc:.2f} F1={f1:.2f} MinC={min_cacc:.2f}")

    # Aggregate and save
    out = {}
    for name, m in results.items():
        arr = {k: np.array(v) for k, v in m.items()}
        out[name] = {
            "acc_mean": float(arr["acc"].mean()),   "acc_std": float(arr["acc"].std()),
            "auc_mean": float(arr["auc"].mean()),   "auc_std": float(arr["auc"].std()),
            "f1_mean":  float(arr["f1"].mean()),    "f1_std":  float(arr["f1"].std()),
            "min_cacc_mean": float(arr["min_cacc"].mean()),
            "min_cacc_std":  float(arr["min_cacc"].std()),
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
