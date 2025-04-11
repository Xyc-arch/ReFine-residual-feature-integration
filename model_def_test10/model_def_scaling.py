import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

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
        final_input_dim = 512 + (self.num_external * 512)
        self.final_layer = nn.Linear(final_input_dim, 10)
    
    def forward(self, x, external_features):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        if external_features is not None and len(external_features) > 0:
            ext_concat = torch.cat(external_features, dim=1)
            combined = torch.cat((features, ext_concat), dim=1)
        else:
            combined = features
        return self.final_layer(combined)

def evaluate_enhanced(model, external_models, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_time = 0.0
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            import time
            start_time = time.time()
            external_features = []
            if external_models is not None and len(external_models) > 0:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(external_models)) as executor:
                    futures = [executor.submit(lambda m: m.get_features(inputs.to(device)), ext_model)
                               for ext_model in external_models]
                    external_features = [future.result() for future in futures]
            outputs = model(inputs, external_features if external_features else None)
            total_time += time.time() - start_time
            _, predicted = torch.max(outputs.data, 1)
            probs = F.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    all_labels_binarized = label_binarize(all_labels, classes=list(range(10)))
    auc = roc_auc_score(all_labels_binarized, all_probs, average='macro', multi_class='ovr')
    avg_time = total_time / len(dataloader)
    return accuracy, avg_time, auc, f1
