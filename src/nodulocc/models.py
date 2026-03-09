"""Model factory for classification.

Design goal: keep model customization in one file so swapping backbones/heads
is straightforward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import timm

try:
    import torchxrayvision as _xrv_module
    _XRV_AVAILABLE = True
except ImportError:
    _xrv_module = None
    _XRV_AVAILABLE = False


# ---------------------------------------------------------------------------
# CBAM : Channel + Spatial Attention (Woo et al., 2018)
# Appliqué sur les feature maps avant le pooling global.
# ---------------------------------------------------------------------------

class _ChannelAttention(nn.Module):
    """Attention canal via avg-pool + max-pool → MLP partagé."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg = self.fc(x.mean(dim=(2, 3)))
        mx  = self.fc(x.amax(dim=(2, 3)))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class _SpatialAttention(nn.Module):
    """Attention spatiale via concaténation avg/max sur les canaux."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel puis spatial)."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.ca = _ChannelAttention(in_channels, reduction)
        self.sa = _SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


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


class XRVBackbone(nn.Module):
    """Wrapper autour d'un DenseNet torchxrayvision pré-entraîné sur RX thorax.

    Convertit l'entrée 3 canaux [0,1] → 1 canal [-1024,1024] attendu par XRV,
    puis retourne les feature maps spatiales (pour CBAM) ou le vecteur poolé.

    Noms de poids supportés (préfixe 'xrv:' dans la config) :
      xrv:densenet121-res224-nih   → NIH ChestX-ray14 (même dataset que le nôtre)
      xrv:densenet121-res224-all   → NIH + CheXpert + MIMIC + PadChest
      xrv:densenet121-res224-chex  → CheXpert uniquement
    """

    def __init__(self, weights: str, pretrained: bool = True, use_cbam: bool = False) -> None:
        super().__init__()
        if not _XRV_AVAILABLE:
            raise RuntimeError(
                "torchxrayvision n'est pas installé. "
                "Exécutez : pip install torchxrayvision"
            )
        model = _xrv_module.models.DenseNet(weights=weights if pretrained else None)
        self.feature_extractor = model.features  # [B, 1024, H', W']
        self.num_features: int = 1024
        self._use_cbam = use_cbam
        if not use_cbam:
            self._pool = nn.AdaptiveAvgPool2d(1)

    def _normalize_xrv(self, x: torch.Tensor) -> torch.Tensor:
        """3 canaux [0,1] → 1 canal [-1024, 1024] attendu par XRV."""
        x = x.mean(dim=1, keepdim=True)   # grayscale  [B, 1, H, W]
        x = x * 2048.0 - 1024.0           # rescale vers range XRV
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(self._normalize_xrv(x))
        if self._use_cbam:
            return feats  # [B, 1024, H', W'] pour CBAM
        return self._pool(feats).flatten(1)  # [B, 1024]


def _build_backbone(name: str, pretrained: bool, use_cbam: bool = False) -> nn.Module:
    """Build a backbone from `timm`, `xrv:` (torchxrayvision) ou `tiny_cnn`.

    Quand CBAM est activé, le backbone retourne des feature maps spatiales.
    """
    if name == "tiny_cnn":
        if use_cbam:
            print("[model] CBAM n'est pas compatible avec tiny_cnn, CBAM sera ignoré.")
        return TinyBackbone()
    if name.startswith("xrv:"):
        weights = name[4:]  # retire le préfixe "xrv:"
        return XRVBackbone(weights=weights, pretrained=pretrained, use_cbam=use_cbam)
    try:
        global_pool = "" if use_cbam else "avg"
        return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool=global_pool)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load backbone '{name}' with timm. "
            "Use a valid timm model name or set backbone='tiny_cnn'."
        ) from exc


class ClassificationModel(nn.Module):
    """Binary classifier avec backbone timm, CBAM optionnel, et tête linéaire."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        dropout: float = 0.0,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        # CBAM est désactivé pour tiny_cnn (pas de feature maps spatiales)
        effective_cbam = use_cbam and backbone_name != "tiny_cnn"
        self.backbone = _build_backbone(backbone_name, pretrained, use_cbam=effective_cbam)
        in_features = int(getattr(self.backbone, "num_features", 64))

        if effective_cbam:
            self.cbam: nn.Module = CBAM(in_features)
            self.pool: nn.Module = nn.AdaptiveAvgPool2d(1)
        else:
            self.cbam = nn.Identity()
            self.pool = nn.Identity()

        self._use_cbam = effective_cbam
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout) * 0.75),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les logits bruts (avant sigmoid), forme `[B]`."""
        feat = self.backbone(x)
        if self._use_cbam:
            feat = self.cbam(feat)
            feat = self.pool(feat).flatten(1)
        return self.head(feat).squeeze(1)


def build_model(task: str, model_cfg: dict[str, Any]) -> nn.Module:
    """Factory qui retourne le modèle de classification."""
    if task != "classification":
        raise ValueError("Only classification task is supported.")

    backbone = str(model_cfg.get("backbone", "tiny_cnn"))
    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.0))
    use_cbam = bool(model_cfg.get("use_cbam", False))
    print(f"Building model: backbone={backbone}, pretrained={pretrained}, dropout={dropout}, cbam={use_cbam}")
    return ClassificationModel(
        backbone_name=backbone,
        pretrained=pretrained,
        dropout=dropout,
        use_cbam=use_cbam,
    )
