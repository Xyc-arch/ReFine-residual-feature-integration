#!/usr/bin/env python3
# domain_transfer_dvd_to_electronics.py

import os
import sys
import random
import json

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# 0) USER‐CONFIGURABLE HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS   = 20   # epochs to pretrain external on DVD
FINETUNE_EPOCHS   = 10   # epochs to finetune on Electronics

# sample counts (adjust as needed)
DVD_PRE_POS       = 1000
DVD_PRE_NEG       = 1000
ELEC_FT_POS       = 200
ELEC_FT_NEG       = 200
ELEC_TEST_POS     = 500
ELEC_TEST_NEG     = 500

# ─────────────────────────────────────────────────────────────────────────────
# 1) TRAIN / EVAL ROUTINES
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

def train_linear_prob(model, loader, epochs, device, lr=0.01, momentum=0.9):
    criterion = torch.nn.CrossEntropyLoss()
    to_opt = [p for p in model.parameters() if p.requires_grad]
    opt = optim.SGD(to_opt, lr=lr, momentum=momentum)
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
        print(f"[LinearProb {e+1}/{epochs}] Loss: {running/len(loader):.4f}")

def train_enhanced_model(model, loader, external, epochs, device, lr=0.01, momentum=0.9):
    criterion = torch.nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    external.eval()
    for e in range(epochs):
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                ext_feat = external.get_features(x)
            opt.zero_grad()
            out = model(x, ext_feat)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[Enhanced {e+1}/{epochs}] Loss: {running/len(loader):.4f}")

def evaluate_model(model, loader, device, enhanced=False, external=None):
    model.eval()
    if external is not None:
        external.eval()

    all_preds, all_labels, all_probs = [], [], []
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if enhanced and external is not None:
                feat = external.get_features(x)
                out  = model(x, feat)
            else:
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

    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, auc, f1, min_class_acc

# ─────────────────────────────────────────────────────────────────────────────
# 2) IMPORT MODELS & LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
from model_def_text import (
    TextTransformerClassifier,
    EnhancedTransformer,
    BaselineAdapterTransformer,
    BigTransformer,
)

DATA_DIR = "./data/processed_acl"
def load_jhu(domain):
    texts, labels = [], []
    base = os.path.join(DATA_DIR, domain)
    for fname, lbl in [("positive.review",1),("negative.review",0)]:
        with open(os.path.join(base, fname), encoding="utf8") as f:
            for line in f:
                toks = [tc.split(":",1)[0] for tc in line.strip().split()[:-1]]
                texts.append(toks); labels.append(lbl)
    return texts, labels

# switch domains: DVD → electronics
dvd_texts, dvd_labels = load_jhu("dvd")
elec_texts, elec_labels = load_jhu("electronics")

# build vocab
vocab = {"<pad>":0, "<unk>":1, "<cls>":2}
for seq in dvd_texts + elec_texts:
    for t in seq:
        if t not in vocab:
            vocab[t] = len(vocab)

def seq2tensor(seq, max_len=256):
    toks = ["<cls>"] + seq
    toks = toks[:max_len] + ["<pad>"]*(max_len-len(toks))
    return torch.tensor([vocab.get(t,1) for t in toks], dtype=torch.long)

dvd_t  = [seq2tensor(s) for s in dvd_texts]
elec_t = [seq2tensor(s) for s in elec_texts]

# ─────────────────────────────────────────────────────────────────────────────
# 3) SPLITS & SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
dvd_pos = [i for i,y in enumerate(dvd_labels) if y==1]
dvd_neg = [i for i,y in enumerate(dvd_labels) if y==0]
elec_pos = [i for i,y in enumerate(elec_labels) if y==1]
elec_neg = [i for i,y in enumerate(elec_labels) if y==0]

random.seed(42); np.random.seed(42)
random.shuffle(dvd_pos); random.shuffle(dvd_neg)
random.shuffle(elec_pos); random.shuffle(elec_neg)

pre_idx  = dvd_pos[:DVD_PRE_POS] + dvd_neg[:DVD_PRE_NEG]
ft_idx   = elec_pos[:ELEC_FT_POS] + elec_neg[:ELEC_FT_NEG]
test_idx = elec_pos[ELEC_FT_POS:ELEC_FT_POS+ELEC_TEST_POS] \
         + elec_neg[ELEC_FT_NEG:ELEC_FT_NEG+ELEC_TEST_NEG]

if len(elec_pos)<ELEC_FT_POS+ELEC_TEST_POS or len(elec_neg)<ELEC_FT_NEG+ELEC_TEST_NEG:
    print(f"Not enough electronics data: pos {len(elec_pos)}, neg {len(elec_neg)}")
    sys.exit(1)

lbls = [elec_labels[i] for i in test_idx]
cnts = {lbl: lbls.count(lbl) for lbl in set(lbls)}
if cnts.get(1,0)!=ELEC_TEST_POS or cnts.get(0,0)!=ELEC_TEST_NEG:
    print("Test split mismatch:", cnts)
    sys.exit(1)

random.shuffle(pre_idx)
random.shuffle(ft_idx)
random.shuffle(test_idx)

class TextDS(Dataset):
    def __init__(self, xs, ys): self.x, self.y = xs, ys
    def __len__(self):       return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

pre_ds  = Subset(TextDS(dvd_t,  dvd_labels),  pre_idx)
ft_ds   = Subset(TextDS(elec_t, elec_labels), ft_idx)
test_ds = Subset(TextDS(elec_t, elec_labels), test_idx)

# ─────────────────────────────────────────────────────────────────────────────
# 4) PRETRAIN external model on DVD
# ─────────────────────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_loader = DataLoader(pre_ds, batch_size=32, shuffle=True)

ckpt = "./models/big_dvd_pre.pt"
if os.path.exists(ckpt):
    external = torch.load(ckpt, map_location=device)
else:
    external = BigTransformer(vocab_size=len(vocab)).to(device)
    train_model(external, pre_loader, PRETRAIN_EPOCHS, device)
    torch.save(external, ckpt)

# ─────────────────────────────────────────────────────────────────────────────
# 5) FINETUNE, EVALUATE & SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
ft_loader   = DataLoader(ft_ds,   batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

baselines = ["notrans", "linearprob", "refine", "adapter"]
results = {b: [] for b in baselines}

for seed in range(42, 47):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"\n=== Seed {seed} ===")

    # a) NoTrans
    m0 = TextTransformerClassifier(len(vocab), 2).to(device)
    train_model(m0, ft_loader, FINETUNE_EPOCHS, device)
    res0 = evaluate_model(m0, test_loader, device=device)
    print(f"NoTrans    → Acc={res0[0]:.2f}, AUC={res0[1]:.4f}, F1={res0[2]:.4f}, MinCAcc={res0[3]:.2f}")
    results["notrans"].append(res0)

    # b) LinearProb
    m1 = BigTransformer(len(vocab)).to(device)
    m1.load_state_dict(external.state_dict())
    for p in m1.parameters(): p.requires_grad = False
    for p in m1.classifier.parameters(): p.requires_grad = True
    train_linear_prob(m1, ft_loader, FINETUNE_EPOCHS, device)
    res1 = evaluate_model(m1, test_loader, device=device)
    print(f"LinearProb → Acc={res1[0]:.2f}, AUC={res1[1]:.4f}, F1={res1[2]:.4f}, MinCAcc={res1[3]:.2f}")
    results["linearprob"].append(res1)

    # c) Refine
    m2 = EnhancedTransformer(len(vocab), 2).to(device)
    train_enhanced_model(m2, ft_loader, external, FINETUNE_EPOCHS, device)
    res2 = evaluate_model(m2, test_loader, device=device, enhanced=True, external=external)
    print(f"Refine      → Acc={res2[0]:.2f}, AUC={res2[1]:.4f}, F1={res2[2]:.4f}, MinCAcc={res2[3]:.2f}")
    results["refine"].append(res2)

    # d) Adapter
    m3 = BaselineAdapterTransformer(external).to(device)
    train_model(m3, ft_loader, FINETUNE_EPOCHS, device)
    res3 = evaluate_model(m3, test_loader, device=device)
    print(f"Adapter     → Acc={res3[0]:.2f}, AUC={res3[1]:.4f}, F1={res3[2]:.4f}, MinCAcc={res3[3]:.2f}")
    results["adapter"].append(res3)

# compute mean & std
summary = {}
for b in baselines:
    arr = np.array(results[b])  # shape (n_seeds, 4)
    summary[b] = {
        "accuracy":      {"mean": float(arr[:,0].mean()), "std": float(arr[:,0].std())},
        "auc":           {"mean": float(arr[:,1].mean()), "std": float(arr[:,1].std())},
        "f1":            {"mean": float(arr[:,2].mean()), "std": float(arr[:,2].std())},
        "min_class_acc": {"mean": float(arr[:,3].mean()), "std": float(arr[:,3].std())},
    }

with open("./results/text_dvd2elec.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved aggregated results to text_dvd2elec.json")
