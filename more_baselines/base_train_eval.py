import torch
import torch.nn as nn
import torch.optim as optim
import copy

from train_eval import evaluate_model
from more_baselines.base_model_def_test10.base_model_def import LoRALinear, JointDiscriminator, grad_reverse

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
# ------------------------------
# DANN-Gate (uses get_features)
# ------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from train_eval import evaluate_model
from more_baselines.base_model_def_test10.base_model_def import LoRALinear, JointDiscriminator, grad_reverse

# ------------------------------
# LoRA adaptation (head-only)
# ------------------------------
def train_lora(external_model, raw_loader, test_loader, device, epochs=30):
    model = copy.deepcopy(external_model).to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Insert LoRA at the LAST FC layer
    last_fc = model.fc_layers[-1]
    in_dim, out_dim = last_fc.in_features, last_fc.out_features
    lora_layer = LoRALinear(in_dim, out_dim, r=8, alpha=16).to(device)
    with torch.no_grad():
        lora_layer.linear.weight.copy_(last_fc.weight)
        if last_fc.bias is not None:
            lora_layer.linear.bias.copy_(last_fc.bias)
    model.fc_layers[-1] = lora_layer  # only LoRA A/B train

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in raw_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # conv -> flatten -> fc_layers (LoRA head)
            loss = ce(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[LoRA] Ep{ep+1}/{epochs} loss={total_loss/len(raw_loader):.4f}")

    return evaluate_model(model, test_loader, device)

# ------------------------------
# (Kept for parity; not used below)
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
# DANN-Gate with LoRA on LAST FC; backbone frozen
# Discriminator observes LOGITS so GRL gradients hit only the LoRA head
# ------------------------------
def train_dann_gate(external_model, raw_loader, test_loader, device, epochs=30):
    import math

    # Label-conditioned discriminator over logits (modest capacity)
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
                    nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, feat, label):
            e = self.emb(label)               # [B, emb_dim]
            x = torch.cat([feat, e], dim=1)   # [B, feat_dim+emb_dim]
            return self.net(x)                # [B,1] in (0,1)

    # 1) Clone model and freeze everything
    model = copy.deepcopy(external_model).to(device)
    for p in model.parameters():
        p.requires_grad = False

    # 2) LoRA-ize the LAST FC (classifier head)
    last_fc = model.fc_layers[-1]
    in_dim, out_dim = last_fc.in_features, last_fc.out_features
    lora_head = LoRALinear(in_dim, out_dim, r=8, alpha=16.0).to(device)
    with torch.no_grad():
        lora_head.linear.weight.copy_(last_fc.weight)
        if last_fc.bias is not None:
            lora_head.linear.bias.copy_(last_fc.bias)
    model.fc_layers[-1] = lora_head  # only LoRA A/B are trainable
    num_classes = out_dim

    # 3) Discriminator acts on LOGITS (dimension = num_classes)
    D = Discriminator(feat_dim=num_classes, num_classes=num_classes,
                      emb_dim=8, hidden=64, dropout=0.2).to(device)

    # 4) Optimizers: LoRA head only; and discriminator
    opt_FC = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    opt_D  = optim.Adam(D.parameters(), lr=5e-4, weight_decay=1e-4)

    ce_mean = nn.CrossEntropyLoss()
    ce  = nn.CrossEntropyLoss(reduction='none')
    bce = nn.BCELoss()
    lambda_gate, mu = 1.0, 1.0

    for ep in range(epochs):
        # Standard GRL schedule
        p_s = ep / max(1, (epochs - 1))
        lam_grl = 2.0 / (1.0 + math.exp(-10 * p_s)) - 1.0

        model.train(); D.train()
        total_loss = 0.0

        for x, y in raw_loader:
            x, y = x.to(device), y.to(device)

            B = x.size(0)
            if B < 2:
                continue
            if B % 2 == 1:  # ensure even batch to split into src/tgt
                x, y = x[:B-1], y[:B-1]
                B -= 1

            half = B // 2
            xs, xt = x[:half], x[half:]
            ys, yt = y[:half], y[half:]

            # Forward to LOGITS (depend only on LoRA head)
            logits_s = model(xs)  # [half, C]
            logits_t = model(xt)  # [half, C]

            # ----- gated source CE + target CE (gate uses D on logits+label) -----
            ls = ce(logits_s, ys)
            with torch.no_grad():
                d_sy = D(logits_s, ys).clamp_(1e-4, 1 - 1e-4)  # p(target | logits_s)
                ws   = d_sy / (1 - d_sy)                       # gate
            loss_src = (lambda_gate * ws.view(-1) * ls).mean()
            loss_tgt = ce_mean(logits_t, yt)
            loss_cls = loss_src + loss_tgt

            # ----- adversarial joint (GRL on logits) -----
            ds_joint = D(grad_reverse(logits_s, lam_grl), ys)  # want 0 (source)
            dt_joint = D(grad_reverse(logits_t, lam_grl), yt)  # want 1 (target)
            loss_joint = 0.5 * (
                bce(ds_joint, torch.zeros_like(ds_joint)) +
                bce(dt_joint, torch.ones_like(dt_joint))
            )

            # ----- adversarial marginal (use NIL labels on both) -----
            nil_s = torch.full((half,), num_classes, device=device, dtype=torch.long)
            nil_t = torch.full((half,), num_classes, device=device, dtype=torch.long)
            ds_marg = D(grad_reverse(logits_s, lam_grl), nil_s)  # want 0
            dt_marg = D(grad_reverse(logits_t, lam_grl), nil_t)  # want 1
            loss_marg = 0.5 * (
                bce(ds_marg, torch.zeros_like(ds_marg)) +
                bce(dt_marg, torch.ones_like(dt_marg))
            )
            loss_adv = loss_joint + loss_marg

            # ----- update LoRA head via combined loss -----
            opt_FC.zero_grad()
            (loss_cls + mu * loss_adv).backward()
            opt_FC.step()

            # ----- update discriminator on DETACHED logits -----
            ls_det = logits_s.detach(); lt_det = logits_t.detach()
            ds_det   = D(ls_det, ys)
            dt_det   = D(lt_det, yt)
            ds_m_det = D(ls_det, nil_s)
            dt_m_det = D(lt_det, nil_t)
            loss_D = 0.25 * (
                bce(ds_det, torch.zeros_like(ds_det)) +
                bce(dt_det, torch.ones_like(dt_det)) +
                bce(ds_m_det, torch.zeros_like(ds_m_det)) +
                bce(dt_m_det, torch.zeros_like(dt_m_det))
            )
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            total_loss += (loss_cls.item() + loss_adv.item() + loss_D.item())

        print(f"[DANN-Gate-Head] Ep{ep+1}/{epochs} λ_grl={lam_grl:.3f} "
              f"loss={total_loss/max(1, len(raw_loader)):.4f}")

    return evaluate_model(model, test_loader, device)


def _loraize_transformer_classifier(model: nn.Module, r=8, alpha=16.0):
    """
    Replace model.classifier (nn.Linear) with LoRALinear while freezing backbone.
    Returns (model, num_classes). Raises AttributeError if classifier missing.
    """
    if not (hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear)):
        raise AttributeError("Expected a Transformer model with a .classifier nn.Linear head.")

    # freeze all first
    for p in model.parameters():
        p.requires_grad = False

    last_fc = model.classifier
    in_dim, out_dim = last_fc.in_features, last_fc.out_features

    lora_head = LoRALinear(in_dim, out_dim, r=r, alpha=alpha).to(next(model.parameters()).device)
    with torch.no_grad():
        lora_head.linear.weight.copy_(last_fc.weight)
        if last_fc.bias is not None and lora_head.linear.bias is not None:
            lora_head.linear.bias.copy_(last_fc.bias)

    model.classifier = lora_head
    return model, out_dim


def train_lora_tf(external_model, raw_loader, test_loader, device, epochs=30):
    """
    LoRA on Transformer .classifier only; backbone frozen.
    """
    model = copy.deepcopy(external_model).to(device)
    model, _ = _loraize_transformer_classifier(model, r=8, alpha=16.0)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in raw_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[LoRA-TF] Ep{ep+1}/{epochs} loss={total_loss/max(1,len(raw_loader)):.4f}")

    return evaluate_model(model, test_loader, device)


def train_dann_gate_tf(external_model, raw_loader, test_loader, device, epochs=30):
    """
    DANN-Gate where discriminator sees logits; GRL sends signal only to LoRA head.
    Backbone frozen; only LoRA parameters train. Label-conditioned weak D.
    """
    import math

    class Discriminator(nn.Module):
        def __init__(self, feat_dim, num_classes, emb_dim=8, hidden=64, dropout=0.2):
            super().__init__()
            self.nil_idx = num_classes
            self.emb = nn.Embedding(num_classes + 1, emb_dim)  # +1 for NIL
            self.net = nn.Sequential(
                nn.Linear(feat_dim + emb_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, feat, label):
            e = self.emb(label)           # [B, emb_dim]
            x = torch.cat([feat, e], 1)   # [B, feat_dim+emb_dim]
            return self.net(x)            # [B,1]

    # 1) clone; LoRA-ize classifier; freeze backbone
    model = copy.deepcopy(external_model).to(device)
    model, num_classes = _loraize_transformer_classifier(model, r=8, alpha=16.0)

    # 2) discriminator over logits
    D = Discriminator(feat_dim=num_classes, num_classes=num_classes,
                      emb_dim=8, hidden=64, dropout=0.2).to(device)

    # 3) optimizers
    opt_FC = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    opt_D  = optim.Adam(D.parameters(), lr=5e-4, weight_decay=1e-4)

    ce_mean = nn.CrossEntropyLoss()
    ce  = nn.CrossEntropyLoss(reduction='none')
    bce = nn.BCELoss()
    lambda_gate, mu = 1.0, 1.0

    for ep in range(epochs):
        # GRL schedule
        p_s = ep / max(1, (epochs - 1))
        lam_grl = 2.0 / (1.0 + math.exp(-10 * p_s)) - 1.0

        model.train(); D.train()
        total_loss = 0.0

        for x, y in raw_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            if B < 2:
                continue
            if B % 2 == 1:
                x, y = x[:B-1], y[:B-1]
                B -= 1
            half = B // 2
            xs, xt = x[:half], x[half:]
            ys, yt = y[:half], y[half:]

            # logits from Transformer (only LoRA head is trainable)
            logits_s = model(xs)
            logits_t = model(xt)

            # gated source CE + target CE
            ls = ce(logits_s, ys)
            with torch.no_grad():
                d_sy = D(logits_s, ys).clamp_(1e-4, 1 - 1e-4)  # p(target | logits_s)
                ws   = d_sy / (1 - d_sy)
            loss_src = (lambda_gate * ws.view(-1) * ls).mean()
            loss_tgt = ce_mean(logits_t, yt)
            loss_cls = loss_src + loss_tgt

            # adversarial joint (labels)
            ds_joint = D(grad_reverse(logits_s, lam_grl), ys)   # want 0
            dt_joint = D(grad_reverse(logits_t, lam_grl), yt)   # want 1
            loss_joint = 0.5 * (bce(ds_joint, torch.zeros_like(ds_joint)) +
                                 bce(dt_joint, torch.ones_like(dt_joint)))

            # adversarial marginal (NIL labels)
            nil_s = torch.full((half,), num_classes, device=device, dtype=torch.long)
            nil_t = torch.full((half,), num_classes, device=device, dtype=torch.long)
            ds_marg = D(grad_reverse(logits_s, lam_grl), nil_s)  # want 0
            dt_marg = D(grad_reverse(logits_t, lam_grl), nil_t)  # want 1
            loss_marg = 0.5 * (bce(ds_marg, torch.zeros_like(ds_marg)) +
                               bce(dt_marg, torch.ones_like(dt_marg)))
            loss_adv = loss_joint + loss_marg

            # update LoRA head
            opt_FC.zero_grad()
            (loss_cls + mu * loss_adv).backward()
            opt_FC.step()

            # update discriminator (detach logits)
            ls_det = logits_s.detach(); lt_det = logits_t.detach()
            ds_det   = D(ls_det, ys)
            dt_det   = D(lt_det, yt)
            ds_m_det = D(ls_det, nil_s)
            dt_m_det = D(lt_det, nil_t)
            loss_D = 0.25 * (bce(ds_det, torch.zeros_like(ds_det)) +
                             bce(dt_det, torch.ones_like(dt_det)) +
                             bce(ds_m_det, torch.zeros_like(ds_m_det)) +
                             bce(dt_m_det, torch.ones_like(dt_m_det)))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            total_loss += (loss_cls.item() + loss_adv.item() + loss_D.item())

        print(f"[DANN-Gate-TF] Ep{ep+1}/{epochs} λ_grl={lam_grl:.3f} "
              f"loss={total_loss/max(1,len(raw_loader)):.4f}")

    return evaluate_model(model, test_loader, device)