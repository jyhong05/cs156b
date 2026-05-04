from typing import Optional

import torch.nn as nn

from .custom_cnn import CheXpertCustomCNN
from .densenet121 import CheXpertDenseNet121
from .densenet264 import CheXpertDenseNet264
from .resnet18 import CheXpertResNet18
from .resnet152 import CheXpertResNet152


MODEL_REGISTRY = {
    "customcnn": CheXpertCustomCNN,
    "resnet18": CheXpertResNet18,
    "resnet152": CheXpertResNet152,
    "densenet121": CheXpertDenseNet121,
    "densenet264": CheXpertDenseNet264,
}

MODEL_ALIASES = {
    "customcnn": "customcnn",
    "resnet18": "resnet18",
    "resnet152": "resnet152",
    "densenet121": "densenet121",
    "densenet264": "densenet264",
}


def normalize_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "").replace("_", "")


def available_models() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY.keys())


def get_model(
    model_name: str,
    num_classes: int = 9,
    pretrained: Optional[bool] = None,
) -> nn.Module:
    key = MODEL_ALIASES.get(normalize_model_name(model_name))
    if key is None:
        supported = ", ".join(available_models())
        raise ValueError(f"Unsupported model_name='{model_name}'. Use one of: {supported}.")

    kwargs = {"num_classes": num_classes}
    if pretrained is not None:
        kwargs["pretrained"] = pretrained

    return MODEL_REGISTRY[key](**kwargs)
