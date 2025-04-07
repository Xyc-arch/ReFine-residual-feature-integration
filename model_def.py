import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    A baseline CNN for training on clean (raw) data.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64*4*4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64*4*4)
        return self.fc_layers(x)
    
    def get_features(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64*4*4)
        # Return features before final linear layer
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
        self.fc_layers = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Now, external features are 2560-dim from the modified BigCNN.
        # Concatenation dimension = 512 + 2560 = 3072.
        self.final_layer = nn.Linear(3072, 10)
    
    def forward(self, x, external_features):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        combined = torch.cat((features, external_features), dim=1)
        return self.final_layer(combined)


class BaselineAdapter(nn.Module):
    def __init__(self, pretrain_model):
        super(BaselineAdapter, self).__init__()
        self.pretrained = pretrain_model
        for param in self.pretrained.parameters():
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
        features = self.pretrained.get_features(x)
        adapted = self.adapter(features)
        return self.classifier(adapted)



class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 80, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(80, 160, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(160, 320, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(320, 640, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640, 640, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(640, 768, kernel_size=3, padding=1), nn.ReLU()
        )
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
        features = self.fc_layers[0](x)
        features = self.fc_layers[1](features)
        return features
