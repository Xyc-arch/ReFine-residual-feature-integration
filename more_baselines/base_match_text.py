#!/usr/bin/env python3
# base_match_text.py — LoRA(head-only) & DANN-Gate(head-only) for Books → Kitchen

import os
import sys
import random
import json
import copy
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")


FILE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../refine/more_baselines
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))  # .../refine
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# 0) USER‐CONFIGURABLE HYPERPARAMETERS (mirrors match_text.py style)
# ─────────────────────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS   = 20   # epochs to pretrain BigTransformer on Books
FINETUNE_EPOCHS   = 10   # epochs to finetune on Kitchen

# sample counts (same semantics as match_text.py)
BOOK_PRE_POS      = 1000
BOOK_PRE_NEG      = 1000
KITCH_FT_POS      = 200
KITCH_FT_NEG      = 200
KITCH_TEST_POS    = 500
KITCH_TEST_NEG    = 500

BATCH_FT          = 32
BATCH_TEST        = 64
SEEDS             = list(range(42, 47))  # five runs: 42..46

# results file (match other *match* baselines)
RESULTS_PATH      = "./results_match_base/base_match_text.json"
CKPT_PATH         = "./models/big_books_pre.pt"
DATA_DIR          = "./data/processed_acl"

# ─────────────────────────────────────────────────────────────────────────────
# 1) MODELS (import from your repo)
# ─────────────────────────────────────────────────────────────────────────────
from model_def_text import (
    TextTransformerClassifier,
    EnhancedTransformer,              # not used here, but kept for parity
    BaselineAdapterTransformer,       # not used here
    BigTransformer,                   # used (has .classifier as final Linear)
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) LoRA & GRL building blocks
# ─────────────────────────────────────────────────────────────────────────────
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
    Parallel LoRA for a Linear head:
      y = (W0 x + b0) + B(Ax) * (alpha/r)
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
        # init LoRA branch
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

class Discriminator(nn.Module):
    """
    Label-conditioned discriminator over LOGITS (dim = num_classes).
    Modest capacity to avoid overpowering the LoRA head.
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
        # feat: [B, C] (logits), label in [0..C] with C meaning NIL
        e = self.emb(label)
        x = torch.cat([feat, e], dim=1)
        return self.net(x)

# ─────────────────────────────────────────────────────────────────────────────
# 3) TRAIN / EVAL routines (inlined, compatible with your models)
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, loader, epochs, device, lr=0.01, momentum=0.9):
    criterion = torch.nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    for e in range(epochs):
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[Epoch {e+1}/{epochs}] Loss: {running/len(loader):.4f}")

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            probs = torch.softmax(out, dim=1)
            pred  = torch.argmax(probs, dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = 100.0 * correct / total

    # min-class accuracy
    num_c = len(all_probs[0])
    class_corr = [0]*num_c
    class_tot  = [0]*num_c
    for p, t in zip(all_preds, all_labels):
        class_tot[t]  += 1
        class_corr[t] += int(p == t)
    min_class_acc = min(
        (100.0*c/t if t>0 else 0.0) for c, t in zip(class_corr, class_tot)
    )

    # require both classes
    if len(set(all_labels)) < 2:
        raise RuntimeError(f"Only one class in y_true: {set(all_labels)}")

    # binary AUC on class-1 probability
    scores_pos = np.array(all_probs)[:,1]
    auc = roc_auc_score(all_labels, scores_pos)
    f1  = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, auc, f1, min_class_acc

# ─────────────────────────────────────────────────────────────────────────────
# 4) DATA loading — JHU ACL processed format (same as match_text.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_jhu(domain):
    texts, labels = [], []
    base = os.path.join(DATA_DIR, domain)
    for fname, lbl in [("positive.review",1),("negative.review",0)]:
        with open(os.path.join(base,fname), encoding="utf8") as f:
            for line in f:
                toks = [tc.split(":",1)[0] for tc in line.strip().split()[:-1]]
                texts.append(toks); labels.append(lbl)
    return texts, labels

books_texts, books_labels = load_jhu("books")
kitch_texts, kitch_labels = load_jhu("kitchen")

# build vocab
vocab = {"<pad>":0,"<unk>":1,"<cls>":2}
for seq in books_texts + kitch_texts:
    for t in seq:
        if t not in vocab:
            vocab[t] = len(vocab)

def seq2tensor(seq, max_len=256):
    toks = ["<cls>"] + seq
    toks = toks[:max_len] + ["<pad>"]*(max_len-len(toks))
    return torch.tensor([vocab.get(t,1) for t in toks], dtype=torch.long)

books_t = [seq2tensor(s) for s in books_texts]
kitch_t = [seq2tensor(s) for s in kitch_texts]

# ─────────────────────────────────────────────────────────────────────────────
# 5) SPLITS (same semantics as match_text.py)
# ─────────────────────────────────────────────────────────────────────────────
book_pos = [i for i,y in enumerate(books_labels) if y==1]
book_neg = [i for i,y in enumerate(books_labels) if y==0]
kit_pos  = [i for i,y in enumerate(kitch_labels) if y==1]
kit_neg  = [i for i,y in enumerate(kitch_labels) if y==0]

random.seed(42); np.random.seed(42)
random.shuffle(book_pos); random.shuffle(book_neg)
random.shuffle(kit_pos);  random.shuffle(kit_neg)

pre_idx  = book_pos[:BOOK_PRE_POS] + book_neg[:BOOK_PRE_NEG]
ft_idx   = kit_pos [:KITCH_FT_POS] + kit_neg [:KITCH_FT_NEG]
test_idx = kit_pos [KITCH_FT_POS:KITCH_FT_POS+KITCH_TEST_POS] \
         + kit_neg [KITCH_FT_NEG:KITCH_FT_NEG+KITCH_TEST_NEG]

# sanity checks like match_text.py
if len(kit_pos)<KITCH_FT_POS+KITCH_TEST_POS or len(kit_neg)<KITCH_FT_NEG+KITCH_TEST_NEG:
    print(f"Not enough kitchen data: pos {len(kit_pos)}, neg {len(kit_neg)}")
    sys.exit(1)

lbls = [kitch_labels[i] for i in test_idx]
cnts = {lbl: lbls.count(lbl) for lbl in set(lbls)}
if cnts.get(1,0)!=KITCH_TEST_POS or cnts.get(0,0)!=KITCH_TEST_NEG:
    print("Test split mismatch:", cnts); sys.exit(1)

random.shuffle(pre_idx); random.shuffle(ft_idx); random.shuffle(test_idx)

class TextDS(Dataset):
    def __init__(self,xs,ys): self.x, self.y = xs, ys
    def __len__(self):      return len(self.y)
    def __getitem__(self,i): return self.x[i], self.y[i]

pre_ds   = Subset(TextDS(books_t,  books_labels),  pre_idx)
ft_ds    = Subset(TextDS(kitch_t,  kitch_labels),  ft_idx)
test_ds  = Subset(TextDS(kitch_t,  kitch_labels),  test_idx)

# ─────────────────────────────────────────────────────────────────────────────
# 6) PRETRAIN external model on Books (as in match_text.py)
# ─────────────────────────────────────────────────────────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_loader = DataLoader(pre_ds, batch_size=BATCH_FT, shuffle=True)

if os.path.exists(CKPT_PATH):
    external = torch.load(CKPT_PATH, map_location=device)
else:
    external = BigTransformer(vocab_size=len(vocab)).to(device)
    train_model(external, pre_loader, PRETRAIN_EPOCHS, device)
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(external, CKPT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# 7) LoRA(head-only) and DANN-Gate(head-only)
# ─────────────────────────────────────────────────────────────────────────────
def loraize_classifier(model_in, r=8, alpha=16.0):
    """Clone model; freeze backbone; replace .classifier with LoRA head."""
    model = copy.deepcopy(model_in).to(device)
    for p in model.parameters():
        p.requires_grad = False
    # assume BigTransformer has .classifier = nn.Linear(in_dim, 2)
    base_head = model.classifier
    in_dim, out_dim = base_head.in_features, base_head.out_features
    lora_head = LoRALinear(in_dim, out_dim, r=r, alpha=alpha).to(device)
    with torch.no_grad():
        lora_head.linear.weight.copy_(base_head.weight)
        if base_head.bias is not None:
            lora_head.linear.bias.copy_(base_head.bias)
    model.classifier = lora_head
    return model, out_dim  # num_classes

def train_lora_head(ft_loader, epochs=FINETUNE_EPOCHS, r=8, alpha=16.0):
    model, _ = loraize_classifier(external, r=r, alpha=alpha)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ce  = nn.CrossEntropyLoss()
    print(f"\n>> LoRA(head-only) for {epochs} epochs")
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x, y in ft_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"   [LoRA] Ep {ep:02d}/{epochs}  Loss={running/len(ft_loader):.4f}")
    return model

def train_dann_gate_head(src_loader, tgt_loader, epochs=FINETUNE_EPOCHS, r=8, alpha=16.0, max_grl=1.0, mu=1.0):
    """
    DANN-Gate: discriminator on logits, GRL hits LoRA head only; backbone frozen.
    Trains with zipped (src, tgt) mini-batches of equal expected size.
    """
    import math

    model, num_classes = loraize_classifier(external, r=r, alpha=alpha)
    D = Discriminator(feat_dim=num_classes, num_classes=num_classes,
                      emb_dim=8, hidden=64, dropout=0.2).to(device)

    opt_FC = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    opt_D  = optim.Adam(D.parameters(), lr=5e-4, weight_decay=1e-4)

    ce_mean = nn.CrossEntropyLoss()
    ce  = nn.CrossEntropyLoss(reduction='none')
    bce = nn.BCELoss()
    lambda_gate = 1.0

    print(f"\n>> DANN-Gate(head LoRA r={r} α={alpha}) for {epochs} epochs")
    for ep in range(1, epochs+1):
        p_s = (ep-1) / max(1,(epochs-1))
        lam_grl = (2.0/(1.0+np.exp(-10*p_s)) - 1.0) * max_grl

        model.train(); D.train()
        total = 0.0

        for (xs, ys), (xt, yt) in zip(src_loader, tgt_loader):
            xs, ys = xs.to(device), ys.to(device)
            xt, yt = xt.to(device), yt.to(device)

            logits_s = model(xs)
            logits_t = model(xt)

            # gated source CE + target CE
            ls = ce(logits_s, ys)
            with torch.no_grad():
                d_sy = D(logits_s, ys).clamp_(1e-4, 1-1e-4)  # p(target | logits_s)
                ws   = d_sy / (1 - d_sy)
            loss_src = (lambda_gate * ws.view(-1) * ls).mean()
            loss_tgt = ce_mean(logits_t, yt)
            loss_cls = loss_src + loss_tgt

            # adversarial joint (with labels)
            ds_joint = D(grad_reverse(logits_s, lam_grl), ys)  # want 0
            dt_joint = D(grad_reverse(logits_t, lam_grl), yt)  # want 1
            loss_joint = 0.5*( bce(ds_joint, torch.zeros_like(ds_joint)) +
                               bce(dt_joint, torch.ones_like(dt_joint)) )

            # adversarial marginal (NIL labels)
            nil_s = torch.full((xs.size(0),), num_classes, device=device, dtype=torch.long)
            nil_t = torch.full((xt.size(0),), num_classes, device=device, dtype=torch.long)
            ds_marg = D(grad_reverse(logits_s, lam_grl), nil_s)  # want 0
            dt_marg = D(grad_reverse(logits_t, lam_grl), nil_t)  # want 1
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
                            bce(dt_det, torch.ones_like(dt_det)) +
                            bce(ds_m_det, torch.zeros_like(ds_m_det)) +
                            bce(dt_m_det, torch.zeros_like(dt_m_det)) )
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            total += (loss_cls.item() + loss_adv.item() + loss_D.item())

        print(f"   [DANN-Gate] Ep {ep:02d}/{epochs}  λ_grl={lam_grl:.3f}  Loss={total/max(1,len(src_loader)):.4f}")

    return model

# ─────────────────────────────────────────────────────────────────────────────
# 8) RUN: build loaders, train/eval, aggregate over seeds
# ─────────────────────────────────────────────────────────────────────────────
ft_loader   = DataLoader(ft_ds,   batch_size=BATCH_FT,   shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False)

# source loader for DANN-Gate: subset of Books pretrain set, matched size to ft_ds
src_size = len(ft_ds)
src_subset = Subset(pre_ds, list(range(min(src_size, len(pre_ds)))))
src_loader = DataLoader(src_subset, batch_size=BATCH_FT, shuffle=True)

results = {
    "lora_head":   [],
    "dann_gate":   [],
}

for seed in SEEDS:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"\n=== Seed {seed} ===")

    # a) LoRA (head-only) on Kitchen
    m_lora = train_lora_head(ft_loader, epochs=FINETUNE_EPOCHS, r=8, alpha=16.0)
    res_lora = evaluate_model(m_lora, test_loader, device=device)
    print(f"LoRA(head) → Acc={res_lora[0]:.2f}, AUC={res_lora[1]:.4f}, F1={res_lora[2]:.4f}, MinCAcc={res_lora[3]:.2f}")
    results["lora_head"].append(res_lora)

    # b) DANN-Gate (head-only GRL over logits) using Books (src) + Kitchen (tgt)
    m_dann = train_dann_gate_head(src_loader, ft_loader, epochs=FINETUNE_EPOCHS, r=8, alpha=16.0, max_grl=1.0, mu=1.0)
    res_dann = evaluate_model(m_dann, test_loader, device=device)
    print(f"DANN-Gate  → Acc={res_dann[0]:.2f}, AUC={res_dann[1]:.4f}, F1={res_dann[2]:.4f}, MinCAcc={res_dann[3]:.2f}")
    results["dann_gate"].append(res_dann)

# ─────────────────────────────────────────────────────────────────────────────
# 9) SAVE aggregated metrics (mean/std) to results_match_base
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

def _agg(arr_list):
    A = np.array(arr_list)  # shape (runs, 4)
    return {
        "accuracy":      {"mean": float(A[:,0].mean()), "std": float(A[:,0].std())},
        "auc":           {"mean": float(A[:,1].mean()), "std": float(A[:,1].std())},
        "f1":            {"mean": float(A[:,2].mean()), "std": float(A[:,2].std())},
        "min_class_acc": {"mean": float(A[:,3].mean()), "std": float(A[:,3].std())},
    }

summary = {k: _agg(v) for k, v in results.items()}

with open(RESULTS_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved aggregated results to {RESULTS_PATH}")
