import torch
import torch.nn as nn
import math

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
# LoRA Linear Layer
# ------------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

        # freeze main linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

# ------------------------------
# Joint Discriminator (for DANN-Gate)
# ------------------------------
class JointDiscriminator(nn.Module):
    def __init__(self, feat_dim=256, num_classes=10):
        super().__init__()
        self.ncls = num_classes + 1   # +1 for "nil" label
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
