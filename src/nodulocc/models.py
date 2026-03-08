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
    """Small CNN used for tests and as explicit `tiny_cnn` backbone."""

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
    """Build a backbone from `timm` or explicit `tiny_cnn`."""
    if name == "tiny_cnn":
        return TinyBackbone()
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load backbone '{name}' with timm. "
            "Use a valid timm model name or set backbone='tiny_cnn'."
        ) from exc


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


class MilPatchClassificationModel(nn.Module):
    """Attention-based MIL classifier over a bag of image patches."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        dropout: float = 0.0,
        attention_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = _build_backbone(backbone_name, pretrained)
        in_features = int(getattr(self.backbone, "num_features", 64))
        self.attention = nn.Sequential(
            nn.Linear(in_features, int(attention_hidden)),
            nn.Tanh(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(attention_hidden), 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for MIL input of shape `[B, N, C, H, W]`."""
        if x.ndim != 5:
            raise ValueError(f"MIL model expects input shape [B,N,C,H,W], got {tuple(x.shape)}")

        b, n, c, h, w = x.shape
        feat = self.backbone(x.reshape(b * n, c, h, w))
        if feat.ndim != 2:
            feat = feat.flatten(1)
        feat = feat.reshape(b, n, feat.shape[-1])

        attn_logits = self.attention(feat).squeeze(-1)  # [B, N]
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(feat * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        return self.head(pooled).squeeze(1)


def build_model(task: str, model_cfg: dict[str, Any]) -> nn.Module:
    """Factory function returning the classification model."""
    if task != "classification":
        raise ValueError("Only classification task is supported.")

    model_type = str(model_cfg.get("type", "global")).lower()
    backbone = str(model_cfg.get("backbone", "tiny_cnn"))
    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.0))

    if model_type == "global":
        print(
            f"Building model type=global with backbone={backbone}, "
            f"pretrained={pretrained}, dropout={dropout}"
        )
        return ClassificationModel(backbone_name=backbone, pretrained=pretrained, dropout=dropout)

    if model_type == "mil_patch":
        mil_cfg = model_cfg.get("mil", {})
        attention_hidden = int(mil_cfg.get("attention_hidden", 256))
        print(
            f"Building model type=mil_patch with backbone={backbone}, "
            f"pretrained={pretrained}, dropout={dropout}, attention_hidden={attention_hidden}"
        )
        return MilPatchClassificationModel(
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
            attention_hidden=attention_hidden,
        )

    raise ValueError("model.type must be one of: global, mil_patch")
