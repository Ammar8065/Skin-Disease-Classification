"""Transfer-learning classifiers built on torchvision backbones.

Each backbone's final classification layer is replaced with a dropout +
linear head sized to ``num_classes``. Pretrained ImageNet weights are loaded
by default.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def build_model(
    backbone: str = "efficientnet_b0",
    num_classes: int = 22,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Construct a classifier with a fresh head.

    Args:
        backbone: One of resnet18, resnet50, efficientnet_b0, efficientnet_b3,
            mobilenet_v3_large.
        num_classes: Number of output classes.
        pretrained: Load ImageNet-pretrained weights.
        dropout: Dropout probability before the final linear layer.
        freeze_backbone: If True, only the new head is trainable.
    """
    backbone = backbone.lower()
    weights = "DEFAULT" if pretrained else None

    if backbone in {"resnet18", "resnet50"}:
        model = getattr(models, backbone)(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

    elif backbone in {"efficientnet_b0", "efficientnet_b3"}:
        model = getattr(models, backbone)(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(
            f"Unsupported backbone {backbone!r}. Choose from: resnet18, "
            "resnet50, efficientnet_b0, efficientnet_b3, mobilenet_v3_large."
        )

    return model
