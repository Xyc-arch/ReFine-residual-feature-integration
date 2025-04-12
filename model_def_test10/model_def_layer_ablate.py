import torch
import torch.nn as nn

class EnhancedCNN(nn.Module):
    """
    Enhanced CNN with fixed overall FC complexity but flexible concatenation point.
    
    The network consists of:
    - A convolutional part that outputs a flattened 1024-dim vector.
    - A total of T FC layers (here T = 5) that process this representation.
      The external features (of dimension 2560) are injected after a chosen number of FC layers.
      Let 'injection_point' be the number of FC layers applied before fusion.
      Then, the same total of T layers is always applied:
          pre_layers: injection_point layers (each maps from 512 to 512,
                     with the very first mapping 1024 -> 512)
          post_layers: (T - injection_point) layers that operate on the concatenated vector.
                     The first post layer maps from (512+2560)=3072 -> 512,
                     and subsequent layers keep 512-dim.
    - A final classification layer maps from 512 to 10 classes.
    
    This setup guarantees that all variants have the same total number of FC layers.
    For example, if T=5:
      - L1 variant (inject later) could have injection_point = 4  (pre: 4 layers, post: 1 layer)
      - L2 variant could have injection_point = 3  (pre: 3 layers, post: 2 layers)
      - L3 variant could have injection_point = 2  (pre: 2 layers, post: 3 layers)
    """
    def __init__(self, injection_point=4, total_layers=5):
        """
        injection_point: number of FC layers applied before concatenation.
                         Must be at least 1 and at most total_layers-1.
        total_layers: total number of FC layers in the enhanced branch.
        """
        super(EnhancedCNN, self).__init__()
        assert 1 <= injection_point < total_layers, "injection_point must be between 1 and total_layers-1"
        self.injection_point = injection_point
        self.total_layers = total_layers
        # Convolutional part; output flattened dimension = 64*4*4 = 1024.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Build pre-fusion FC layers (they all output 512-dim)
        self.pre_fusion = nn.ModuleList()
        # First layer: maps input 1024 -> 512.
        self.pre_fusion.append(nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        ))
        # The remaining pre layers (if injection_point > 1):
        for _ in range(1, injection_point):
            self.pre_fusion.append(nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5)
            ))
        
        # Build post-fusion FC layers.
        post_layers_count = total_layers - injection_point
        self.post_fusion = nn.ModuleList()
        if post_layers_count > 0:
            # First post-fusion layer: input dimension = 512 (from pre_fusion) + 2560 (external)
            self.post_fusion.append(nn.Sequential(
                nn.Linear(3072, 512),
                nn.ReLU(),
                nn.Dropout(0.5)
            ))
            # Remaining post-fusion layers.
            for _ in range(1, post_layers_count):
                self.post_fusion.append(nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                ))
        # Final classification layer (maps the final 512-dim features to 10 classes)
        self.classifier = nn.Linear(512, 10)
    
    def forward(self, x, external_features):
        # Convolution and flattening.
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # shape: [batch, 1024]
        
        # Pre-fusion processing.
        for layer in self.pre_fusion:
            x = layer(x)  # Each layer outputs 512-dim.
        # x is now a 512-dim feature vector from the pre_fusion block.
        
        # Concatenate with external features (assumed shape: [batch, 2560]).
        x = torch.cat((x, external_features), dim=1)  # Now size: [batch, 3072].
        
        # Post-fusion processing.
        for layer in self.post_fusion:
            x = layer(x)  # Remains 512-dim throughout.
        
        # Final classification.
        logits = self.classifier(x)
        return logits

# Convenience wrapper classes for the ablation study variants.
# We set total_layers=5 for all, and only change the injection point.
# Here, we map:
#  - EnhancedCNN_L1: injection_point = 4 (fusion occurs after 4 layers, then 1 layer processing of fused representation).
#  - EnhancedCNN_L2: injection_point = 3 (3 pre-fusion layers, 2 post-fusion layers).
#  - EnhancedCNN_L3: injection_point = 2 (2 pre-fusion layers, 3 post-fusion layers).
        
class EnhancedCNN_L1(EnhancedCNN):
    def __init__(self):
        super(EnhancedCNN_L1, self).__init__(injection_point=4, total_layers=5)
        
class EnhancedCNN_L2(EnhancedCNN):
    def __init__(self):
        super(EnhancedCNN_L2, self).__init__(injection_point=3, total_layers=5)
        
class EnhancedCNN_L3(EnhancedCNN):
    def __init__(self):
        super(EnhancedCNN_L3, self).__init__(injection_point=2, total_layers=5)
