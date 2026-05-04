import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CheXpertCustomCNN(nn.Module):
    def __init__(self, num_classes: int = 9, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained:
            raise ValueError(
                "CheXpertCustomCNN does not provide pretrained weights. "
                "Set pretrained=false for customcnn."
            )

        self.features = nn.Sequential(
            _conv_block(3, 32),
            nn.MaxPool2d(kernel_size=2),
            _conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            _conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2),
            _conv_block(128, 256),
            nn.MaxPool2d(kernel_size=2),
            _conv_block(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
