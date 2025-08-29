import torch
import torch.nn as nn
import torch.optim as optim
import copy

from train_eval import evaluate_model
from more_baselines.base_model_def_test10 import LoRALinear, JointDiscriminator, grad_reverse

# ------------------------------
# LoRA adaptation (head-only)
# ------------------------------
def train_lora(external_model, raw_loader, test_loader, device, epochs=30):
    model = copy.deepcopy(external_model).to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Insert LoRA at the LAST FC layer of BigCNN
    last_fc = model.fc_layers[-1]
    in_dim, out_dim = last_fc.in_features, last_fc.out_features
    lora_layer = LoRALinear(in_dim, out_dim, r=8, alpha=16).to(device)
    with torch.no_grad():
        lora_layer.linear.weight.copy_(last_fc.weight)
        lora_layer.linear.bias.copy_(last_fc.bias)
    model.fc_layers[-1] = lora_layer

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in raw_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # BigCNN forward already calls get_features -> fc_layers
            loss = ce(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[LoRA] Ep{ep+1}/{epochs} loss={total_loss/len(raw_loader):.4f}")

    return evaluate_model(model, test_loader, device)

# ------------------------------
# Wrapper: base.get_features + separate head
# ------------------------------
class FeatHead(nn.Module):
    def __init__(self, base_model, head_module):
        super().__init__()
        self.base = base_model
        self.head = head_module
    def forward(self, x):
        f = self.base.get_features(x)
        return self.head(f)

# ------------------------------
# DANN-Gate (uses get_features)
# ------------------------------
def train_dann_gate(external_model, raw_loader, test_loader, device, epochs=30):
    base = copy.deepcopy(external_model).to(device)
    last_fc = copy.deepcopy(external_model.fc_layers[-1]).to(device)
    feat_dim   = last_fc.in_features
    num_classes = last_fc.out_features

    model = FeatHead(base, last_fc).to(device)
    D = JointDiscriminator(feat_dim=feat_dim, num_classes=num_classes).to(device)

    opt_FC = optim.Adam(model.parameters(), lr=1e-3)  # base + head
    opt_D  = optim.Adam(D.parameters(), lr=1e-3)

    ce  = nn.CrossEntropyLoss(reduction='none')
    bce = nn.BCELoss()
    lambda_gate, mu = 1.0, 1.0

    for ep in range(epochs):
        model.train(); D.train()
        total_loss = 0.0
        for x, y in raw_loader:
            x, y = x.to(device), y.to(device)

            f = base.get_features(x)   # [B, feat_dim]
            logits = last_fc(f)        # [B, num_classes]

            # gated CE
            ls = ce(logits, y)
            with torch.no_grad():
                d_sy = D(f, y).clamp_(1e-4, 1 - 1e-4)
                ws   = d_sy / (1 - d_sy)
            loss_cls = (lambda_gate * ws.view(-1) * ls).mean()

            # adversarial marginal (nil label)
            nil = torch.full((x.size(0),), num_classes, device=device, dtype=torch.long)
            d_marg = D(grad_reverse(f, 1.0), nil)
            loss_adv = bce(d_marg, torch.ones_like(d_marg))

            # update base + head
            opt_FC.zero_grad()
            (loss_cls + mu * loss_adv).backward()
            opt_FC.step()

            # update discriminator
            with torch.no_grad():
                f_det = base.get_features(x)
            d_det   = D(f_det, y)
            d_m_det = D(f_det, nil)
            loss_D = 0.5 * (
                bce(d_det, torch.zeros_like(d_det)) +
                bce(d_m_det, torch.zeros_like(d_m_det))
            )
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            total_loss += (loss_cls.item() + loss_adv.item() + loss_D.item())
        print(f"[DANN-Gate] Ep{ep+1}/{epochs} loss={total_loss/len(raw_loader):.4f}")

    return evaluate_model(model, test_loader, device)
