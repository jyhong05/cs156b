import torch.nn as nn
from torchvision import models


class CheXpertResNet18(nn.Module):
    def __init__(self, num_classes: int = 9, pretrained: bool = True) -> None:
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.model = models.resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
