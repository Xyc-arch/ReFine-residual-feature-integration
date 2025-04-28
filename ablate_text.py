#!/usr/bin/env python3
# ablate_text.py

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
PRETRAIN_EPOCHS   = 20
FINETUNE_EPOCHS   = 10

DVD_PRE_POS       = 1000
DVD_PRE_NEG       = 1000
ELEC_FT_POS       = 200
ELEC_FT_NEG       = 200
ELEC_TEST_POS     = 500
ELEC_TEST_NEG     = 500

BASE_ADAPTER_DIM     = 128
ADAPTER_MULTIPLIERS  = [1, 25, 50, 100, 500]
SEEDS                = [42, 43, 44]

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

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out   = model(x)
            probs = torch.softmax(out, dim=1)
            pred  = torch.argmax(probs, dim=1)

            correct += (pred == y).sum().item()
            total   += y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100.0 * correct / total

    # compute min-class accuracy
    n_classes = probs.shape[1]
    class_corr = [0]*n_classes
    class_tot  = [0]*n_classes
    for p, t in zip(all_preds, all_labels):
        class_tot[t]  += 1
        class_corr[t] += int(p == t)
    min_class_acc = min((100.0*c/t if t>0 else 0.0)
                        for c, t in zip(class_corr, class_tot))

    # binary AUC on class-1 prob
    scores_pos = np.array(all_probs)[:,1]
    auc = roc_auc_score(all_labels, scores_pos)
    f1  = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, auc, f1, min_class_acc

# ─────────────────────────────────────────────────────────────────────────────
# 2) IMPORT MODELS & LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
from model_def_text import BigTransformer, BaselineAdapterTransformer

DATA_DIR = "./data/processed_acl"
def load_jhu(domain):
    texts, labels = [], []
    base = os.path.join(DATA_DIR, domain)
    for fname, lbl in [("positive.review",1), ("negative.review",0)]:
        with open(os.path.join(base, fname), encoding="utf8") as f:
            for line in f:
                toks = [tc.split(":",1)[0] for tc in line.strip().split()[:-1]]
                texts.append(toks)
                labels.append(lbl)
    return texts, labels

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
# 3) SPLITS & DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────
# Create positive & negative index lists
dvd_pos  = [i for i,y in enumerate(dvd_labels) if y==1]
dvd_neg  = [i for i,y in enumerate(dvd_labels) if y==0]
elec_pos = [i for i,y in enumerate(elec_labels) if y==1]
elec_neg = [i for i,y in enumerate(elec_labels) if y==0]

# shuffle
random.seed(42); np.random.seed(42)
random.shuffle(dvd_pos); random.shuffle(dvd_neg)
random.shuffle(elec_pos); random.shuffle(elec_neg)

# define train / ft / test splits
pre_idx  = dvd_pos[:DVD_PRE_POS] + dvd_neg[:DVD_PRE_NEG]
ft_idx   = elec_pos[:ELEC_FT_POS] + elec_neg[:ELEC_FT_NEG]
test_idx = elec_pos[ELEC_FT_POS:ELEC_FT_POS+ELEC_TEST_POS] \
         + elec_neg[ELEC_FT_NEG:ELEC_FT_NEG+ELEC_TEST_NEG]

if len(elec_pos) < ELEC_FT_POS + ELEC_TEST_POS or len(elec_neg) < ELEC_FT_NEG + ELEC_TEST_NEG:
    print(f"Not enough electronics data: pos {len(elec_pos)}, neg {len(elec_neg)}")
    sys.exit(1)

# sanity check test counts
lbls = [elec_labels[i] for i in test_idx]
cnts = {lbl: lbls.count(lbl) for lbl in set(lbls)}
if cnts.get(1,0)!=ELEC_TEST_POS or cnts.get(0,0)!=ELEC_TEST_NEG:
    print("Test split mismatch:", cnts)
    sys.exit(1)

random.shuffle(pre_idx)
random.shuffle(ft_idx)
random.shuffle(test_idx)

class TextDS(Dataset):
    def __init__(self, xs, ys):
        self.x, self.y = xs, ys
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

pre_ds  = Subset(TextDS(dvd_t,  dvd_labels),  pre_idx)
ft_ds   = Subset(TextDS(elec_t, elec_labels), ft_idx)
test_ds = Subset(TextDS(elec_t, elec_labels), test_idx)

# ─────────────────────────────────────────────────────────────────────────────
# 4) PRETRAIN external model on DVD
# ─────────────────────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_loader = DataLoader(pre_ds, batch_size=32, shuffle=True)

ckpt = "./big_dvd_pre.pt"
if os.path.exists(ckpt):
    external = torch.load(ckpt, map_location=device)
else:
    external = BigTransformer(vocab_size=len(vocab)).to(device)
    train_model(external, pre_loader, PRETRAIN_EPOCHS, device)
    torch.save(external, ckpt)

# ─────────────────────────────────────────────────────────────────────────────
# 5) FINETUNE & EVALUATE ADAPTERS
# ─────────────────────────────────────────────────────────────────────────────
ft_loader   = DataLoader(ft_ds,   batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

results = {f"adapter_{m}x": [] for m in ADAPTER_MULTIPLIERS}

for m in ADAPTER_MULTIPLIERS:
    adapter_dim = BASE_ADAPTER_DIM * m
    print(f"\n=== Adapter bottleneck {adapter_dim} ({m}×) ===")
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        model = BaselineAdapterTransformer(
            external,
            bottleneck_dim=adapter_dim,
            num_classes=2
        ).to(device)

        train_model(model, ft_loader, FINETUNE_EPOCHS, device)
        res = evaluate_model(model, test_loader, device)
        print(f"[Seed {seed}] Acc={res[0]:.2f}, AUC={res[1]:.4f}, F1={res[2]:.4f}, MinCAcc={res[3]:.2f}")

        results[f"adapter_{m}x"].append(res)

# ─────────────────────────────────────────────────────────────────────────────
# 6) AGGREGATE & SAVE
# ─────────────────────────────────────────────────────────────────────────────
summary = {}
for k, runs in results.items():
    arr = np.array(runs)  # shape (3,4)
    summary[k] = {
        "accuracy":      {"mean": float(arr[:,0].mean()), "std": float(arr[:,0].std())},
        "auc":           {"mean": float(arr[:,1].mean()), "std": float(arr[:,1].std())},
        "f1":            {"mean": float(arr[:,2].mean()), "std": float(arr[:,2].std())},
        "min_class_acc": {"mean": float(arr[:,3].mean()), "std": float(arr[:,3].std())},
    }

with open("ablate_adapters.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved results to ablate_adapters.json")
