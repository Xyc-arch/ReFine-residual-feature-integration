import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 4 * 4)
        return self.fc_layers(x)
    
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 4 * 4)
        return self.fc_layers[:-1](x)

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5)
        )
        self.final_layer = nn.Linear(512 + 2560, 10)
    
    def forward(self, x, teacher_features):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 4 * 4)
        features = self.fc_layers(x)
        combined = torch.cat((features, teacher_features), dim=1)
        return self.final_layer(combined)

class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 80, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(80, 160, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(160, 320, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(320, 640, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640, 640, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640, 768, 3, padding=1), nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(768, 2560), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2560, 100)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
    
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers[0](x)
        features = self.fc_layers[1](features)
        return features

class BaselineAdapter100(nn.Module):
    def __init__(self, teacher_model, bottleneck_dim=128):
        super(BaselineAdapter100, self).__init__()
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
