import torch.nn as nn
from torchvision.models.densenet import DenseNet


class CheXpertDenseNet264(nn.Module):
    def __init__(self, num_classes: int = 9, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained:
            raise ValueError(
                "TorchVision does not provide pretrained DenseNet-264 weights. "
                "Set pretrained=false for densenet264."
            )

        self.model = DenseNet(
            growth_rate=32,
            block_config=(6, 12, 64, 48),
            num_init_features=64,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
