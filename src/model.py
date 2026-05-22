import torch
import torch.nn as nn
from torchvision import models
from config import DEVICE

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=True)
        
        # Remove final classification layer
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        return x

def load_model():
    model = ResNetFeatureExtractor()
    model.eval()
    model.to(DEVICE)
    return model