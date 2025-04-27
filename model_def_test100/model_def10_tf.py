# model_def_test100/model_def10_tf.py

import torch
import torch.nn as nn

# -------------------------------------------------------------------
# Student‐side transformers for CIFAR-100
# -------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """Baseline vision transformer for CIFAR-100."""
    def __init__(self, patch_size=4, emb_dim=128, num_layers=2, num_classes=100):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (32 // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=8, dim_feedforward=256,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(), nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(512, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                   # (B, emb_dim, H, W)
        x = x.flatten(2).transpose(1, 2)          # (B, num_patches, emb_dim)

        cls = self.cls_token.expand(B, -1, -1)    # (B,1,emb_dim)
        x = torch.cat((cls, x), dim=1) + self.pos_embed

        x = self.encoder(x)
        feat = self.mlp_head(x[:, 0])
        return self.classifier(feat)

    def get_features(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.mlp_head(x[:, 0])


class EnhancedTransformer(nn.Module):
    """Concatenate own features with teacher features for CIFAR-100."""
    def __init__(self, patch_size=4, emb_dim=128, num_layers=2,
                 num_classes=100, external_dim=2560):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (32 // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=8, dim_feedforward=256,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(), nn.Dropout(0.5)
        )
        self.final = nn.Linear(512 + external_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, external_feats):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.encoder(x)

        feat = self.mlp_head(x[:, 0])
        combined = torch.cat((feat, external_feats), dim=1)
        return self.final(combined)


class BaselineAdapterTransformer(nn.Module):
    """Frozen teacher + small adapter + new head for CIFAR-100."""
    def __init__(self, teacher_model, bottleneck_dim=128, num_classes=100):
        super().__init__()
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(2560, bottleneck_dim),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, 2560),
            nn.ReLU(), nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(2560, num_classes)

    def forward(self, x):
        feat = self.teacher.get_features(x)
        adapted = self.adapter(feat)
        return self.classifier(adapted)


# -------------------------------------------------------------------
# Teacher-side transformer for CIFAR-10
# -------------------------------------------------------------------

class BigTransformer(nn.Module):
    """Larger vision transformer for CIFAR-10 pretraining."""
    def __init__(self, patch_size=2, emb_dim=512, num_layers=6, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (32 // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=8, dim_feedforward=1024,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 2560),
            nn.ReLU(), nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(2560, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.encoder(x)
        out = self.fc(x[:, 0])
        return self.classifier(out)

    def get_features(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.fc(x[:, 0])
