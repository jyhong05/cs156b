import torch.nn as nn
from torchvision import models


class CheXpertBaseline(nn.Module):
	def __init__(self, num_classes: int = 9) -> None:
		super().__init__()
		self.model = models.resnet18(weights="DEFAULT")
		in_features = self.model.fc.in_features
		self.model.fc = nn.Linear(in_features, num_classes)

	def forward(self, x):
		return self.model(x)