import math
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ------------------------------
# Reproducibility
# ------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ------------------------------
# Device (GPU if available)
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Gradient Reversal Layer
# ------------------------------
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

# ------------------------------
# CIFAR10 subsets
# ------------------------------
class CIFAR10SubsetRandomLabel(Dataset):
    def __init__(self, dataset, indices, num_classes=10):
        self.dataset = dataset
        self.indices = indices
        self.targets = np.random.randint(0, num_classes, size=len(indices)).tolist()
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        img, _ = self.dataset[self.indices[idx]]
        return img, self.targets[idx]

class CIFAR10Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.dataset[self.indices[idx]]

# ------------------------------
# Models
# ------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(8*8*128, 256),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

class LabelClassifier(nn.Module):
    def __init__(self, feat_dim=256, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes)
    def forward(self, x): return self.classifier(x)

class JointDiscriminator(nn.Module):
    def __init__(self, feat_dim=256, num_classes=10):
        super().__init__()
        self.ncls = num_classes + 1
        self.net = nn.Sequential(
            nn.Linear(feat_dim + self.ncls, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, feat, label):
        B = feat.size(0)
        onehot = torch.zeros(B, self.ncls, device=feat.device)
        onehot.scatter_(1, label.unsqueeze(1), 1)
        return self.net(torch.cat([feat, onehot], dim=1))

# ------------------------------
# LoRA wrapper
# ------------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

# ------------------------------
# Residual encoder for ReFine
# ------------------------------
class ResidualEncoder(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(8*8*64, 128),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

class ReFineClassifier(nn.Module):
    def __init__(self, feat_dim=256, resid_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + resid_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, f, h):
        return self.net(torch.cat([f, h], dim=1))

# ------------------------------
# Data loaders (1000 src, 1000 tgt)
# ------------------------------
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
full_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_set   = datasets.CIFAR10('./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ]))

perm      = np.random.permutation(len(full_train))
src_idx   = perm[:1000]
tgt_idx   = perm[1000:2000]
source_ds = CIFAR10SubsetRandomLabel(full_train, src_idx)
target_ds = CIFAR10Subset(full_train, tgt_idx)

bs = 64
src_loader = DataLoader(source_ds, bs, shuffle=True,  drop_last=True)
tgt_loader = DataLoader(target_ds, bs, shuffle=True,  drop_last=True)
test_loader= DataLoader(test_set,  bs, shuffle=False)

num_epochs = 10

# ------------------------------
# Helpers
# ------------------------------
def eval_acc(F, C, loader):
    F.eval(); C.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = C(F(x)).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total

# Minimal CORAL loss helper (Deep CORAL)
def coral_loss(fs, ft):
    b, d = fs.size(0), fs.size(1)
    fs_c = fs - fs.mean(dim=0, keepdim=True)
    ft_c = ft - ft.mean(dim=0, keepdim=True)
    cs = (fs_c.t() @ fs_c) / (b - 1)
    ct = (ft_c.t() @ ft_c) / (b - 1)
    return ((cs - ct).pow(2).sum()) / (4.0 * d * d)

# ============================================
# 1) SOURCE‐ONLY baseline
# ============================================
ce_mean = nn.CrossEntropyLoss()
F_s = FeatureExtractor().to(device)
C_s = LabelClassifier().to(device)
opt_s = optim.Adam(list(F_s.parameters()) + list(C_s.parameters()), lr=1e-3)
for ep in range(num_epochs):
    F_s.train(); C_s.train()
    total_loss = 0.0
    for x_s, y_s in src_loader:
        x_s, y_s = x_s.to(device), y_s.to(device)
        loss = ce_mean(C_s(F_s(x_s)), y_s)
        opt_s.zero_grad(); loss.backward(); opt_s.step()
        total_loss += loss.item()
    print(f"[Source] Ep{ep+1}/{num_epochs} loss={total_loss/len(src_loader):.4f}")
src_acc = eval_acc(F_s, C_s, test_loader)
print(f"Source‐Only Test Acc: {src_acc*100:.2f}%\n")

# ============================================
# 2) DANN‐GATE adaptation (Corrected)
# ============================================
F = F_s; C = C_s
D = JointDiscriminator(num_classes=10).to(device)
opt_FC = optim.Adam(list(F.parameters()) + list(C.parameters()), lr=1e-3)
opt_D  = optim.Adam(D.parameters(), lr=1e-3)

ce  = nn.CrossEntropyLoss(reduction='none')
bce = nn.BCELoss()
lambda_gate = 1.0
mu = 1.0

for ep in range(num_epochs):
    p = ep/(num_epochs-1)
    lam_grl = 2.0/(1.0+np.exp(-10*p)) - 1.0

    F.train(); C.train(); D.train()
    for (xs, ys), (xt, yt) in zip(src_loader, tgt_loader):
        xs, ys = xs.to(device), ys.to(device)
        xt, yt = xt.to(device), yt.to(device)
        fs, ft = F(xs), F(xt)

        # classification with gating
        ls = ce(C(fs), ys)
        lt = ce(C(ft), yt)
        with torch.no_grad():
            d_sy = D(fs, ys).clamp_(1e-4, 1-1e-4)
            ws = d_sy / (1 - d_sy)
        loss_cls = (lambda_gate * ws.view(-1) * ls).mean() + lt.mean()

        # adversarial joint
        ds_joint = D(grad_reverse(fs, lam_grl), ys)
        dt_joint = D(grad_reverse(ft, lam_grl), yt)
        loss_joint = 0.5*(bce(ds_joint, torch.zeros_like(ds_joint)) +
                          bce(dt_joint, torch.ones_like(dt_joint)))

        # adversarial marginal
        nil_s = torch.full((xs.size(0),), 10, device=device, dtype=torch.long)
        nil_t = torch.full((xt.size(0),), 10, device=device, dtype=torch.long)
        ds_marg = D(grad_reverse(fs, lam_grl), nil_s)
        dt_marg = D(grad_reverse(ft, lam_grl), nil_t)
        loss_marg = 0.5*(bce(ds_marg, torch.zeros_like(ds_marg)) +
                         bce(dt_marg, torch.ones_like(dt_marg)))

        loss_adv = loss_joint + loss_marg

        # update F,C
        opt_FC.zero_grad()
        (loss_cls + mu*loss_adv).backward()
        opt_FC.step()

        # update D
        ds_det   = D(fs.detach(), ys)
        dt_det   = D(ft.detach(), yt)
        ds_m_det = D(fs.detach(), nil_s)
        dt_m_det = D(ft.detach(), nil_t)
        loss_D = 0.25*( bce(ds_det, torch.zeros_like(ds_det)) +
                        bce(dt_det, torch.ones_like(dt_det)) +
                        bce(ds_m_det, torch.zeros_like(ds_m_det)) +
                        bce(dt_m_det, torch.ones_like(dt_m_det)) )
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    print(f"[Adapt] Ep{ep+1}/{num_epochs} λ_grl={lam_grl:.3f}")

adapt_acc = eval_acc(F, C, test_loader)
print(f"DANN‐Gate Test Acc: {adapt_acc*100:.2f}%\n")

# ============================================
# 2b) DCORAL‐GATE baseline (non-adversarial)
# ============================================
F_cg = FeatureExtractor().to(device); F_cg.load_state_dict(F_s.state_dict())
C_cg = LabelClassifier().to(device);  C_cg.load_state_dict(C_s.state_dict())
D_g  = JointDiscriminator(num_classes=10).to(device)   # gate-only discriminator

opt_FC_cg = optim.Adam(list(F_cg.parameters()) + list(C_cg.parameters()), lr=1e-3)
opt_D_g   = optim.Adam(D_g.parameters(), lr=1e-3)

bce = nn.BCELoss()
lambda_gate = 1.0
coral_w = 1.0

for ep in range(num_epochs):
    F_cg.train(); C_cg.train(); D_g.train()
    total_cls = total_coral = total_ld = 0.0

    for (xs, ys), (xt, yt) in zip(src_loader, tgt_loader):
        xs, ys = xs.to(device), ys.to(device)
        xt, yt = xt.to(device), yt.to(device)

        fs = F_cg(xs); ft = F_cg(xt)

        # gate weights (stop-grad)
        with torch.no_grad():
            d_sy = D_g(fs, ys).clamp_(1e-4, 1-1e-4)
            ws   = d_sy / (1 - d_sy)

        # gated source CE + target CE + CORAL alignment
        ls = ce(C_cg(fs), ys)                 # per-sample CE from earlier (reduction='none')
        lt = ce_mean(C_cg(ft), yt)            # mean CE
        gate_src = (lambda_gate * ws.view(-1) * ls).mean()
        align    = coral_loss(fs, ft) * coral_w
        loss_FC  = gate_src + lt + align

        opt_FC_cg.zero_grad()
        loss_FC.backward()
        opt_FC_cg.step()

        # train gate discriminator (no GRL)
        nil_s = torch.full((xs.size(0),), 10, device=device, dtype=torch.long)
        nil_t = torch.full((xt.size(0),), 10, device=device, dtype=torch.long)

        ds_joint = D_g(fs.detach(), ys)
        dt_joint = D_g(ft.detach(), yt)
        ds_marg  = D_g(fs.detach(), nil_s)
        dt_marg  = D_g(ft.detach(), nil_t)

        loss_Dg = 0.25 * (
            bce(ds_joint, torch.zeros_like(ds_joint)) +
            bce(dt_joint, torch.ones_like(dt_joint))  +
            bce(ds_marg,  torch.zeros_like(ds_marg))  +
            bce(dt_marg,  torch.ones_like(dt_marg))
        )
        opt_D_g.zero_grad(); loss_Dg.backward(); opt_D_g.step()

        total_cls   += (gate_src + lt).item()
        total_coral += align.item()
        total_ld    += loss_Dg.item()

    print(f"[DCORAL-GATE] Ep{ep+1}/{num_epochs} "
          f"CE={total_cls/len(src_loader):.4f} "
          f"CORAL={total_coral/len(src_loader):.4f} "
          f"D_loss={total_ld/len(src_loader):.4f}")

dcoralgate_acc = eval_acc(F_cg, C_cg, test_loader)
print(f"DCORAL-Gate Test Acc: {dcoralgate_acc*100:.2f}%\n")

# ============================================
# 3) LoRA adaptation baseline
# ============================================
F_l = FeatureExtractor().to(device); F_l.load_state_dict(F_s.state_dict())
for p in F_l.parameters(): p.requires_grad = False
C_l = LabelClassifier().to(device); C_l.classifier = LoRALinear(256,10,r=8,alpha=16).to(device)
with torch.no_grad():
    C_l.classifier.linear.weight.copy_(C_s.classifier.weight)
    C_l.classifier.linear.bias.copy_(C_s.classifier.bias)
opt_l = optim.Adam([p for p in C_l.parameters() if p.requires_grad], lr=1e-3)
for ep in range(num_epochs):
    C_l.train(); total_loss=0.0
    for x_t,y_t in tgt_loader:
        x_t,y_t = x_t.to(device), y_t.to(device)
        logits = C_l(F_l(x_t)); loss=ce_mean(logits,y_t)
        opt_l.zero_grad(); loss.backward(); opt_l.step()
        total_loss+=loss.item()
    print(f"[LoRA ] Ep{ep+1}/{num_epochs} loss={total_loss/len(tgt_loader):.4f}")
lora_acc = eval_acc(F_l, C_l, test_loader)
print(f"LoRA-Adapt Test Acc: {lora_acc*100:.2f}%\n")

# ============================================
# 4) ReFine baseline
# ============================================
F_ref = FeatureExtractor().to(device); F_ref.load_state_dict(F_s.state_dict())
for p in F_ref.parameters(): p.requires_grad = False
h = ResidualEncoder().to(device)
w = ReFineClassifier(feat_dim=256, resid_dim=128, num_classes=10).to(device)
opt_ref = optim.Adam(list(h.parameters()) + list(w.parameters()), lr=1e-3)
for ep in range(num_epochs):
    h.train(); w.train(); total_loss=0.0
    for x_t, y_t in tgt_loader:
        x_t, y_t = x_t.to(device), y_t.to(device)
        feat = F_ref(x_t); resid = h(x_t)
        logits = w(feat, resid)
        loss = ce_mean(logits, y_t)
        opt_ref.zero_grad(); loss.backward(); opt_ref.step()
        total_loss += loss.item()
    print(f"[ReFine] Ep{ep+1}/{num_epochs} loss={total_loss/len(tgt_loader):.4f}")

def eval_refine(F_ref, h, w, loader):
    F_ref.eval(); h.eval(); w.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feat, resid = F_ref(x), h(x)
            pred = w(feat, resid).argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
    return correct / total

refine_acc = eval_refine(F_ref, h, w, test_loader)
print(f"ReFine-Adapt Test Acc: {refine_acc*100:.2f}%\n")

# ============================================
# 5) TARGET‐ONLY baseline
# ============================================
F_t = FeatureExtractor().to(device); C_t = LabelClassifier().to(device)
opt_t = optim.Adam(list(F_t.parameters()) + list(C_t.parameters()), lr=1e-3)
for ep in range(num_epochs):
    F_t.train(); C_t.train(); total_loss=0.0
    for x_t,y_t in tgt_loader:
        x_t,y_t = x_t.to(device), y_t.to(device)
        loss = ce_mean(C_t(F_t(x_t)), y_t)
        opt_t.zero_grad(); loss.backward(); opt_t.step()
        total_loss+=loss.item()
    print(f"[Target] Ep{ep+1}/{num_epochs} loss={total_loss/len(tgt_loader):.4f}")
tgt_acc = eval_acc(F_t, C_t, test_loader)
print(f"Target-Only Test Acc: {tgt_acc*100:.2f}%\n")

# ------------------------------
# Save all results
# ------------------------------
results = {
    "source_only":  src_acc,
    "dann_gate":    adapt_acc,
    "dcoral_gate":  dcoralgate_acc,
    "lora_adapt":   lora_acc,
    "refine_adapt": refine_acc,
    "target_only":  tgt_acc
}
with open("all_baseline_accs.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved baseline_accs.json")
