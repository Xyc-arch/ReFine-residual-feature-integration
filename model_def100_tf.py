import torch
import torch.nn as nn
import torch.nn.functional as F

# Baseline transformer classifier (for CIFAR-10) similar to TransformerClassifier in model_def_tf.py.
class TransformerClassifier(nn.Module):
    def __init__(self, patch_size=4, emb_dim=128, num_layers=2, num_classes=10):
        super(TransformerClassifier, self).__init__()
        self.patch_size = patch_size
        # Patch embedding using a convolution.
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        # CIFAR-10 (32x32) -> 8x8 patches.
        self.num_patches = (32 // patch_size) ** 2
        
        # Learnable class token and positional embeddings.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))
        
        # Transformer encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=8, 
            dim_feedforward=256, 
            dropout=0.1, 
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP head to produce 512-dim features.
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
        x = self.patch_embed(x)  # (B, emb_dim, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, emb_dim)
        
        # Append class token.
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer_encoder(x)
        cls_out = x[:, 0]
        features = self.mlp_head(cls_out)
        return self.classifier(features)
    
    def get_features(self, x):
        # Return 512-dim features (before the final classifier)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)

# Enhanced transformer model that concatenates its own features with external (teacher) features.
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
        # Final layer takes concatenated features: (512 + external_dim) -> 10 classes.
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

# BigTransformer teacher model for CIFAR-100.
# This model is similar to BigTransformer from model_def_tf.py but with num_classes=100.
class BigTransformer(nn.Module):
    def __init__(self, patch_size=2, emb_dim=512, num_layers=6, num_classes=100):
        """
        For 32x32 images with patch_size=2, there are 16x16=256 patches.
        The larger embedding dimension (512), multiple layers, and a larger feedforward network
        yield a high parameter count.
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
        
        # Map to 2560 dimensions.
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 2560),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Teacher classification head: outputs 100 classes.
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

# Baseline adapter model for transformer (using teacher features).
class BaselineAdapter100(nn.Module):
    def __init__(self, teacher_model):
        super(BaselineAdapter100, self).__init__()
        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(2560, 5120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(5120, 2560),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(2560, 10)
    
    def forward(self, x):
        features = self.teacher.get_features(x)
        adapted = self.adapter(features)
        return self.classifier(adapted)
