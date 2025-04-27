# model_def_text.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1) BigTransformer: frozen “external” backbone
# ─────────────────────────────────────────────────────────────────────────────
class BigTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.d_model     = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        self.dropout     = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # head projects to a large “feature” vector
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 5),  # 512→2560
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(d_model * 5, 2)

        # init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def get_features(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Returns a 2560-dim feature vector (post-fc) for each input.
        """
        B, L = x.size()
        tok_emb = self.token_embed(x) * math.sqrt(self.d_model)  # (B, L, D)
        cls     = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        x       = torch.cat([cls, tok_emb], dim=1)               # (B, L+1, D)

        # ─── FIX: slice exactly L+1 positions ───
        x = x + self.pos_embed[:, : L + 1, :]                    # (B, L+1, D)

        x = self.dropout(x)
        x = self.encoder(x)                                       # (B, L+1, D)
        feat = x[:, 0]                                            # (B, D)
        return self.fc(feat)                                      # (B, 2560)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        feat = self.get_features(x)      # (B, 2560)
        return self.classifier(feat)     # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Scratch classifier baseline
# ─────────────────────────────────────────────────────────────────────────────
class TextTransformerClassifier(BigTransformer):
    def __init__(self, vocab_size: int, num_classes: int = 2, **kwargs):
        super().__init__(vocab_size, **kwargs)
        # override head
        self.fc         = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(512, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Refine / enhanced baseline
# ─────────────────────────────────────────────────────────────────────────────
class EnhancedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        # adapter_dim is no longer used for sizing the fusion layer
        adapter_dim: int = 64,
        **big_kw,
    ):
        super().__init__()
        # 1) your own small classifier (outputs 512‐dim features)
        self.main = TextTransformerClassifier(
            vocab_size,
            num_classes=adapter_dim,    # only affects its internal head
            **big_kw
        )
        # Grab its feature‐dim (the output size of main.fc)
        self.own_feat_dim = self.main.fc[0].out_features  # 512

        # 2) external features dim = BigTransformer.d_model * 5
        external_dim = big_kw.get("d_model", 512) * 5     # 2560

        # 3) fusion layer: must take (512 + 2560) = 3072 inputs
        self.final = nn.Linear(self.own_feat_dim + external_dim, num_classes)

    def forward(self, x: torch.LongTensor, ext_feat: torch.FloatTensor):
        # own 512‐dim features
        own_feat = self.main.get_features(x)              # (B, 512)
        # concat with external 2560‐dim
        combined = torch.cat([own_feat, ext_feat], dim=1) # (B, 3072)
        return self.final(combined)                       # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Adapter baseline
# ─────────────────────────────────────────────────────────────────────────────
class BaselineAdapterTransformer(nn.Module):
    def __init__(
        self,
        external: BigTransformer,
        bottleneck_dim: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        self.external = external
        for p in self.external.parameters():
            p.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(self.external.d_model * 5, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, self.external.d_model * 5),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(self.external.d_model * 5, num_classes)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        feat    = self.external.get_features(x)  # (B, 2560)
        adapted = self.adapter(feat)             # (B, 2560)
        return self.classifier(adapted)          # (B, num_classes)
