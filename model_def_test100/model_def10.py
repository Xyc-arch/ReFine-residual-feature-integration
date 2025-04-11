import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A baseline CNN for training on CIFAR-100 (student).
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        # Final output is 100 classes
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 100)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 4 * 4)
        return self.fc_layers(x)
    
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 4 * 4)
        # Return features from before the final classification layer.
        return self.fc_layers[:-1](x)

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Features from student conv & fc layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Teacher features are assumed to have 2560 dimensions.
        # Concatenation dimension = 512 + 2560 = 3072.
        self.final_layer = nn.Linear(3072, 100)  # 100 classes for CIFAR-100
     
    def forward(self, x, teacher_features):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        combined = torch.cat((features, teacher_features), dim=1)
        return self.final_layer(combined)

class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()
        # This network is for the teacher, pretrained on CIFAR-10.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 80, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(80, 160, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(160, 320, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(320, 640, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640, 640, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640, 768, kernel_size=3, padding=1), nn.ReLU()
        )
        # For teacher on CIFAR-10, final classification layer outputs 10 classes.
        self.fc_layers = nn.Sequential(
            nn.Linear(768, 2560), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2560, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
    
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        # Return intermediate features (2560-dim) used for student guidance.
        features = self.fc_layers[0](x)
        features = self.fc_layers[1](features)
        return features



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
        self.classifier = nn.Linear(2560, 100)
    
    def forward(self, x):
        features = self.teacher.get_features(x)  # get features
        adapted = self.adapter(features)  # apply adapter
        return self.classifier(adapted)

