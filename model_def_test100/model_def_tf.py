import torch
import torch.nn as nn

# Baseline transformer classifier for CIFAR-100
class TransformerClassifier(nn.Module):
    def __init__(self, patch_size=4, emb_dim=128, num_layers=2, num_classes=100):
        super(TransformerClassifier, self).__init__()
        self.patch_size = patch_size
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (32 // patch_size) ** 2

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head for features
        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Final classifier
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)          # (B, emb_dim, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, emb_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        cls_out = x[:, 0]
        features = self.mlp_head(cls_out)
        return self.classifier(features)

    def get_features(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)


# Enhanced transformer that concatenates its own features with external ones
class EnhancedTransformer(nn.Module):
    def __init__(self, patch_size=4, emb_dim=128, num_layers=2, num_classes=100, external_dim=2560):
        super(EnhancedTransformer, self).__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (32 // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Final classification layer takes concatenated features
        self.final_layer = nn.Linear(512 + external_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, external_features):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        cls_out = x[:, 0]
        features = self.mlp_head(cls_out)
        combined = torch.cat((features, external_features), dim=1)
        return self.final_layer(combined)


# Baseline adapter model with a frozen pretrained transformer
class BaselineAdapterTransformer(nn.Module):
    def __init__(self, pretrain_model, bottleneck_dim=128, num_classes=100):
        super(BaselineAdapterTransformer, self).__init__()
        self.pretrained = pretrain_model
        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(2560, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, 2560),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(2560, num_classes)

    def forward(self, x):
        features = self.pretrained.get_features(x)
        adapted = self.adapter(features)
        return self.classifier(adapted)


# Larger transformer for pretraining on CIFAR-100
class BigTransformer(nn.Module):
    def __init__(self, patch_size=2, emb_dim=512, num_layers=6, num_classes=100):
        super(BigTransformer, self).__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (32 // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected mapping to external feature dim
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 2560),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(2560, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        cls_out = x[:, 0]
        out = self.fc(cls_out)
        return self.classifier(out)

    def get_features(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        cls_out = x[:, 0]
        return self.fc(cls_out)
