import torch
import torch.nn as nn
import math

# Baseline transformer classifier (replacing the CNN baseline)
class TransformerClassifier(nn.Module):
    def __init__(self, patch_size=4, emb_dim=128, num_layers=2, num_classes=10):
        super(TransformerClassifier, self).__init__()
        self.patch_size = patch_size
        # Create patch embeddings using a convolution.
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        # For CIFAR10 (32x32), patch_size=4 gives 8x8=64 patches.
        self.num_patches = (32 // patch_size) ** 2
        
        # Learnable class token and positional embeddings.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))
        
        # Use built-in TransformerEncoderLayer with batch_first=True.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=8, 
            dim_feedforward=256, 
            dropout=0.1, 
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP head mapping transformer output to a 512-dim feature space.
        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Final classification layer.
        self.classifier = nn.Linear(512, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.patch_embed(x)  # (B, emb_dim, H, W) where H=W=32/patch_size
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (B, num_patches, emb_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Append the class token.
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed  # (B, num_patches+1, emb_dim)
        
        # Transformer encoder expects input shape (B, seq_len, emb_dim)
        x = self.transformer_encoder(x)
        cls_out = x[:, 0]  # Use the class token.
        features = self.mlp_head(cls_out)
        return self.classifier(features)
    
    def get_features(self, x):
        # Return 512-dim features from the MLP head (before classification).
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)

# Enhanced transformer model that concatenates its own features with external ones.
class EnhancedTransformer(nn.Module):
    def __init__(self, patch_size=4, emb_dim=128, num_layers=2, num_classes=10, external_dim=2560):
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
        # Final classification layer takes concatenated features.
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

# Baseline adapter model that uses a frozen pretrained external transformer.
class BaselineAdapterTransformer(nn.Module):
    def __init__(self, pretrain_model, bottleneck_dim=128):
        super(BaselineAdapterTransformer, self).__init__()
        self.pretrained = pretrain_model
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(2560, bottleneck_dim),  # down
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, 2560),  # up
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(2560, 10)
    
    def forward(self, x):
        features = self.pretrained.get_features(x)  # get features
        adapted = self.adapter(features)  # apply bottleneck adapter
        return self.classifier(adapted)

# A larger transformer model (replacing BigTransformer) for pretraining.
# Configured to have at least 20x more parameters than EnhancedTransformer.
class BigTransformer(nn.Module):
    def __init__(self, patch_size=2, emb_dim=512, num_layers=6, num_classes=10):
        """
        For 32x32 images with patch_size=2, there are 16x16=256 patches.
        The larger embedding dimension (512), more layers, and larger feedforward dimension (1024)
        ensure a high parameter count.
        """
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
        
        # Fully connected mapping to 2560 dimensions.
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
