import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# =============================================================================
# Model definitions (minimal versions to count parameters)
# =============================================================================
class EnhancedScalingCNN(nn.Module):
    def __init__(self, num_external=0):
        super(EnhancedScalingCNN, self).__init__()
        self.num_external = num_external
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # The final layer takes 512 features from the base network plus 512 per external model.
        final_input_dim = 512 + (self.num_external * 512)
        self.final_layer = nn.Linear(final_input_dim, 10)
    
    def forward(self, x, external_features=None):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        if external_features is not None and len(external_features) > 0:
            # In a real-case scenario, external_features would be a list of tensors.
            ext_concat = torch.cat(external_features, dim=1)
            combined = torch.cat((features, ext_concat), dim=1)
        else:
            combined = features
        return self.final_layer(combined)

class NaiveClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(NaiveClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# =============================================================================
# Utility function to count total trainable parameters.
# =============================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# Compute parameter counts for models 1-8
# =============================================================================
model_nums = list(range(1, 9))
enhanced_params = []
naive_params = []

# For each model number, we assume:
# - For the Enhanced model: num_external = model_num.
# - For the Naive model: the input dimension is the concatenation of features from 
#   'model_num' external models. (Assumed to be 512 each.)
for num in model_nums:
    # Create enhanced model instance.
    model_enh = EnhancedScalingCNN(num_external=num)
    enhanced_count = count_parameters(model_enh)
    enhanced_params.append(enhanced_count)

    # For the naive classifier, if external features are concatenated from "num" models,
    # the input dimension is assumed to be (num * 512).
    input_dim = num * 512
    model_naive = NaiveClassifier(input_dim=input_dim)
    naive_count = count_parameters(model_naive)
    naive_params.append(naive_count)

# =============================================================================
# Plotting
# =============================================================================
plt.figure(figsize=(8, 6))
plt.plot(model_nums, enhanced_params, marker='o', label='Enhanced Model')
plt.plot(model_nums, naive_params, marker='s', label='Naive Model')
plt.xlabel("Model Number (# of External Models)")
plt.ylabel("Total Trainable Parameters")
plt.title("Parameter Count: Enhanced vs Naive Models")
plt.xticks(model_nums)
plt.legend()
plt.grid(True)

# Make sure the destination directory exists.
save_dir = "plots/scaling_ablate"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "scaling_para.png")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
plt.close()
