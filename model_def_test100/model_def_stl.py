import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A baseline CNN for training on STL-10 (student).
    Adjusted for STL-10 input (32x32) and 10 output classes.
    """
    def __init__(self):
        super(CNN, self).__init__()
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
        # For 32x32 input, 3 pooling layers reduce spatial dims to 32/8 = 4.
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),  # 64*4*4 = 1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)  # STL-10 has 10 classes
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
    
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        # Exclude the final classification layer for feature extraction.
        return self.fc_layers[:-1](x)

    
    

class EnhancedCNN(nn.Module):
    """
    Enhanced CNN model for STL-10 that concatenates internal features with teacher features.
    Adapted for 32x32 inputs.
    """
    def __init__(self):
        super(EnhancedCNN, self).__init__()
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
        # Now, after conv layers for 32x32 inputs, feature map is 4x4.
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),  # Adjusted: 64*4*4 = 1024
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # External teacher features are assumed to remain 2560-dim (from BigCNN pretraining).
        # Therefore, the combined features will be 512 + 2560 = 3072.
        self.final_layer = nn.Linear(3072, 10)
    
    def forward(self, x, teacher_features):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        combined = torch.cat((features, teacher_features), dim=1)
        return self.final_layer(combined)



class BaselineAdapter(nn.Module):
    def __init__(self, teacher_model, bottleneck_dim=128):
        super(BaselineAdapter, self).__init__()
        self.teacher = teacher_model
        for param in self.teacher.parameters():
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
        features = self.teacher.get_features(x)  # get features
        adapted = self.adapter(features)  # apply adapter
        return self.classifier(adapted)
