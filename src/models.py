"""Construção dos modelos do torchvision com a última camada adaptada."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

# nome -> (função construtora, pesos pré-treinados no ImageNet)
_REGISTRY = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "swin_t": (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}

PRETTY = {
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-T",
    "swin_t": "Swin-T",
    "resnet18": "ResNet-18",
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Cria o modelo e troca o classificador para ``num_classes`` saídas.

    Transfer learning: todo o modelo é ajustado (fine-tuning completo),
    partindo dos pesos do ImageNet quando ``pretrained=True``.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Modelo desconhecido: {name}. Opções: {list(_REGISTRY)}")
    fn, weights = _REGISTRY[name]
    model = fn(weights=weights if pretrained else None)

    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("vit"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name.startswith("efficientnet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name.startswith("convnext"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name.startswith("swin"):
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
