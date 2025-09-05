#!/usr/bin/env python3
import os
import json
import copy
import random
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from torchvision.datasets import USPS, MNIST

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
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

class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,80,3,padding=1),  nn.ReLU(), nn.MaxPool2d(2,2),
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
        return h  # 2560-D features

class ResizeWrapper(nn.Module):
    """Ensures inputs are 32×32 and exposes get_features passthrough."""
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

# ──────────────────────────────────────────────────────────────────────────────
# LoRA / DANN-Gate building blocks
# ──────────────────────────────────────────────────────────────────────────────
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class LoRALinear(nn.Module):
    """
    Parallel LoRA head: out = W0 x + B(Ax) * (alpha/r) + b0
    W0,b0 are frozen; only A and B train.
    """
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

def loraize_classifier_head(base, r=8, alpha=16):
    """
    Freeze everything, then replace the LAST fc layer with a LoRA-wrapped Linear.
    Only LoRA A/B are trainable; the base linear is frozen and copied.
    """
    for p in base.parameters():
        p.requires_grad = False
    last = base.fc_layers[-1]
    assert isinstance(last, nn.Linear), "Expected base.fc_layers[-1] to be Linear"
    lora = LoRALinear(last.in_features, last.out_features, r=r, alpha=alpha)
    with torch.no_grad():
        lora.linear.weight.copy_(last.weight)
        if last.bias is not None:
            lora.linear.bias.copy_(last.bias)
    base.fc_layers[-1] = lora
    return base

# ──────────────────────────────────────────────────────────────────────────────
# Discriminator (label-conditioned MLP, tuned to be less identifying)
# ──────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    p(domain=target | [feat, label]) with a trainable MLP conditioned on label.
    Includes moderate regularization to reduce discriminative power.
    """
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
        """
        feat:  [B, feat_dim]  (here: logits when we adapt the head)
        label: LongTensor [B], values in [0..num_classes] where num_classes==NIL
        """
        e = self.emb(label)               # [B, emb_dim]
        x = torch.cat([feat, e], dim=1)   # [B, feat_dim+emb_dim]
        return self.net(x)                # [B, 1] in (0,1)

# ──────────────────────────────────────────────────────────────────────────────
# Training / Eval
# ──────────────────────────────────────────────────────────────────────────────
def train_supervised(model, loader, epochs, device, lr=0.01, name="model"):
    model.to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    print(f"\n>> {name}: {epochs} epochs")
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0)
        print(f"   [{name}] Epoch {ep:02d}/{epochs}  Loss={total/len(loader.dataset):.4f}")

def _unwrap_teacher(teacher):
    if isinstance(teacher, ResizeWrapper):
        return teacher.model, True
    return teacher, False

def train_lora(teacher, loader, test_loader, device, epochs=30):
    base, was_wrapped = _unwrap_teacher(teacher)
    model = copy.deepcopy(base).to(device)
    for p in model.parameters():
        p.requires_grad = False
    last = model.fc_layers[-1]
    in_dim, out_dim = last.in_features, last.out_features
    lora = LoRALinear(in_dim, out_dim, r=8, alpha=16).to(device)
    with torch.no_grad():
        lora.linear.weight.copy_(last.weight)
        if last.bias is not None:
            lora.linear.bias.copy_(last.bias)
    model.fc_layers[-1] = lora
    if was_wrapped:
        model = ResizeWrapper(model, (32, 32)).to(device)
    train_supervised(model, loader, epochs, device, lr=1e-3, name="LoRA")
    return evaluate(model, test_loader, device)

# ──────────────────────────────────────────────────────────────────────────────
# DANN-Gate with LoRA on LAST FC (classifier head); backbone frozen
# Discriminator observes LOGITS so GRL gradients hit only the LoRA head
# ──────────────────────────────────────────────────────────────────────────────
def train_dann_gate_joint(teacher, src_loader, tgt_loader, test_loader, device,
                          epochs=30, lora_r=8, lora_alpha=16, mu=1.0, max_grl=1.0):

    base, was_wrapped = _unwrap_teacher(teacher)
    model = copy.deepcopy(base).to(device)

    # LoRA on the classifier head (last FC); freeze everything else
    model = loraize_classifier_head(model, r=lora_r, alpha=lora_alpha)
    if was_wrapped:
        model = ResizeWrapper(model, (32,32)).to(device)

    # Discriminator consumes logits (dim = num_classes)
    # Keep it modest so it doesn't overpower the LoRA head
    num_class = base.fc_layers[-1].out_features
    D = Discriminator(feat_dim=num_class, num_classes=num_class,
                      emb_dim=8, hidden=64, dropout=0.2).to(device)

    # only LoRA params on the head require grad; everything else is frozen
    opt_FC = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                              lr=1e-3, weight_decay=1e-4)
    opt_D  = torch.optim.Adam(D.parameters(), lr=5e-4, weight_decay=1e-4)

    ce_mean = nn.CrossEntropyLoss()
    ce  = nn.CrossEntropyLoss(reduction='none')
    bce = nn.BCELoss()
    lambda_gate = 1.0

    print(f"\n>> DANN-Gate (LoRA on head r={lora_r} α={lora_alpha}, backbone frozen): {epochs} epochs")
    for ep in range(1, epochs+1):
        p_s = (ep-1) / max(1, (epochs-1))
        lam_grl = (2.0/(1.0+np.exp(-10*p_s)) - 1.0) * max_grl

        model.train(); D.train()
        total_loss = 0.0

        for (xs, ys), (xt, yt) in zip(src_loader, tgt_loader):
            xs, ys = xs.to(device), ys.to(device)
            xt, yt = xt.to(device), yt.to(device)

            # forward to logits (depend on LoRA head)
            logits_s = model(xs)  # [B, C]
            logits_t = model(xt)  # [B, C]

            # ----- gated source CE + target CE -----
            ls = ce(logits_s, ys)
            with torch.no_grad():
                d_sy = D(logits_s, ys).clamp_(1e-4, 1-1e-4)
                ws   = d_sy / (1 - d_sy)
            loss_src = (lambda_gate * ws.view(-1) * ls).mean()
            loss_tgt = ce_mean(logits_t, yt)
            loss_cls = loss_src + loss_tgt

            # ----- adversarial joint (GRL on logits) -----
            ds_joint = D(grad_reverse(logits_s, lam_grl), ys)
            dt_joint = D(grad_reverse(logits_t, lam_grl), yt)
            loss_joint = 0.5*( bce(ds_joint, torch.zeros_like(ds_joint)) +
                               bce(dt_joint, torch.ones_like(dt_joint)) )

            # ----- adversarial marginal (use NIL) -----
            nil_s = torch.full((xs.size(0),), num_class, device=device, dtype=torch.long)
            nil_t = torch.full((xt.size(0),), num_class, device=device, dtype=torch.long)
            ds_marg = D(grad_reverse(logits_s, lam_grl), nil_s)
            dt_marg = D(grad_reverse(logits_t, lam_grl), nil_t)
            loss_marg = 0.5*( bce(ds_marg, torch.zeros_like(ds_marg)) +
                              bce(dt_marg, torch.ones_like(dt_marg)) )
            loss_adv = loss_joint + loss_marg

            # ----- update LoRA head via combined loss -----
            opt_FC.zero_grad()
            (loss_cls + mu*loss_adv).backward()
            opt_FC.step()

            # ----- update discriminator on detached logits -----
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

            total_loss += (loss_cls.item() + loss_adv.item() + loss_D.item())

        print(f"   [DANN-Gate-Head] Epoch {ep:02d}/{epochs}  "
              f"λ_grl={lam_grl:.3f}  Loss={(total_loss/max(1,len(src_loader))):.4f}")

    return evaluate(model, test_loader, device)

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers (USPS → MNIST)
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT  = "./data"
BATCH_SIZE = 32
PRETRAIN_N = 5000
FINETUNE_N = 100

tf = transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

def ensure_dataset(name, cls, split):
    print(f"--- Checking {name} [{split}] ---")
    try:
        cls(root=DATA_ROOT, train=(split=="train"), download=False, transform=tf)
        print(f"{name} [{split}] already present.")
    except Exception:
        print(f"{name} [{split}] not found. Downloading …")
        try:
            cls(root=DATA_ROOT, train=(split=="train"), download=True, transform=tf)
            print(f"{name} [{split}] downloaded.")
        except Exception:
            print(f"Error downloading {name} [{split}]:")
            traceback.print_exc()

def subset(ds, n, seed):
    if n is None or n >= len(ds): return ds
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return Subset(ds, idx[:n])

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results_match_base", exist_ok=True)

    print("\n===== DATASET SETUP =====")
    ensure_dataset("USPS", USPS, "train")
    ensure_dataset("USPS", USPS, "test")
    ensure_dataset("MNIST", MNIST, "train")
    ensure_dataset("MNIST", MNIST, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretrain teacher on USPS
    teacher_path = "models/usps_teacher.pt"
    if os.path.exists(teacher_path):
        print(f"\nLoading pretrained teacher from {teacher_path}")
        teacher = torch.load(teacher_path, map_location=device)
    else:
        print("\n===== TEACHER PRETRAINING ON USPS =====")
        teacher = BigCNN().to(device)
        usps_tr_full = subset(USPS(DATA_ROOT, train=True, download=False, transform=tf), PRETRAIN_N, seed=42)
        pre_loader   = DataLoader(usps_tr_full, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        train_supervised(teacher, pre_loader, epochs=60, device=device, lr=0.01, name="BigCNN-USPS")
        torch.save(teacher, teacher_path)
        print(f"Saved teacher model to {teacher_path}")
    teacher = ResizeWrapper(teacher, (32,32)).to(device)

    # MNIST test loader
    mnist_test = MNIST(DATA_ROOT, train=False, download=False, transform=tf)
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    metrics = {
        "lora":      {"acc":[], "auc":[], "f1":[], "minc":[]},
        "dann_gate": {"acc":[], "auc":[], "f1":[], "minc":[]},
    }

    for run in range(5):
        run_seed = 42 + run
        print(f"\n===== RUN {run+1}/5 (seed={run_seed}) =====")

        # target tiny MNIST subset
        mnist_tr = subset(MNIST(DATA_ROOT, train=True, download=False, transform=tf), FINETUNE_N, seed=run_seed)
        tgt_loader = DataLoader(mnist_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # source tiny USPS subset (match size to target for stable zip)
        usps_tr  = subset(USPS(DATA_ROOT, train=True, download=False, transform=tf), FINETUNE_N, seed=run_seed)
        src_loader = DataLoader(usps_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # LoRA baseline (target-only fine-tune of head)
        acc, auc, f1, minc = train_lora(teacher, tgt_loader, test_loader, device, epochs=30)
        print(f"→ LoRA: Acc={acc:.2f}%, AUC={auc:.2f}%, F1={f1:.2f}%, MinC={minc:.2f}%")
        metrics["lora"]["acc"].append(acc);  metrics["lora"]["auc"].append(auc)
        metrics["lora"]["f1"].append(f1);    metrics["lora"]["minc"].append(minc)

        # DANN-Gate (source+target, LoRA on head; backbone frozen)
        acc, auc, f1, minc = train_dann_gate_joint(
            teacher, src_loader, tgt_loader, test_loader, device,
            epochs=30, lora_r=8, lora_alpha=16, mu=1.0, max_grl=1.0
        )
        print(f"→ DANN-Gate (LoRA on head r=8 α=16): Acc={acc:.2f}%, AUC={auc:.2f}%, "
              f"F1={f1:.2f}%, MinC={minc:.2f}%")
        metrics["dann_gate"]["acc"].append(acc);  metrics["dann_gate"]["auc"].append(auc)
        metrics["dann_gate"]["f1"].append(f1);    metrics["dann_gate"]["minc"].append(minc)

    # Aggregate and save
    out = {}
    for name, vals in metrics.items():
        arr = {k: np.array(v) for k,v in vals.items()}
        out[name] = {
            "acc_mean": float(arr["acc"].mean()),   "acc_std": float(arr["acc"].std()),
            "auc_mean": float(arr["auc"].mean()),   "auc_std": float(arr["auc"].std()),
            "f1_mean":  float(arr["f1"].mean()),    "f1_std":  float(arr["f1"].std()),
            "minc_mean":float(arr["minc"].mean()),  "minc_std": float(arr["minc"].std()),
        }

    save_path = "results_match_base/match_digit.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAll done. Results saved to {save_path}")

if __name__ == "__main__":
    main()