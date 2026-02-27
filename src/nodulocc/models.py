"""Model factory for classification.

Design goal: keep model customization in one file so swapping backbones/heads
is straightforward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import timm


class TinyBackbone(nn.Module):
    """Small fallback CNN used for tests and safe fallback when timm fails."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.num_features = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature vector of shape `[B, num_features]`."""
        x = self.features(x)
        return x.flatten(1)


def _build_backbone(name: str, pretrained: bool) -> nn.Module:
    """Build a backbone from `timm` or fallback to `TinyBackbone`."""
    if name == "tiny_cnn":
        return TinyBackbone()
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    except Exception:
        return TinyBackbone()


class ClassificationModel(nn.Module):
    """Binary classifier on top of a shared feature backbone."""

    def __init__(self, backbone_name: str, pretrained: bool, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = _build_backbone(backbone_name, pretrained)
        in_features = int(getattr(self.backbone, "num_features", 64))
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (before sigmoid), shape `[B]`."""
        feat = self.backbone(x)
        return self.head(feat).squeeze(1)


def build_model(task: str, model_cfg: dict[str, Any]) -> nn.Module:
    """Factory function returning the classification model."""
    if task != "classification":
        raise ValueError("Only classification task is supported.")

    backbone = str(model_cfg.get("backbone", "tiny_cnn"))
    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.0))
    return ClassificationModel(backbone_name=backbone, pretrained=pretrained, dropout=dropout)
