"""Preprocessing profiles for classification datasets.

This module isolates heavier image preprocessing pipelines so `data.py` stays
focused on dataset/split/dataloader logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency, handled at runtime.
    cv2 = None


@dataclass(frozen=True)
class TeamV2Settings:
    """Configuration values for the EDA-inspired preprocessing pipeline."""

    clip_low: float = 1.0
    clip_high: float = 99.0
    use_thorax_mask: bool = True
    thorax_margin: float = 0.97
    thorax_blur: int = 51
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    use_gaussian_blur: bool = True
    gaussian_kernel_size: int = 3
    gaussian_sigma: float = 0.5
    resize_interpolation: str = "area"  # area | linear

    @staticmethod
    def from_cfg(cfg: dict[str, Any]) -> "TeamV2Settings":
        """Build validated settings from `data.preprocessing.team_v2` block."""
        p_cfg = cfg.get("data", {}).get("preprocessing", {}).get("team_v2", {})
        percentiles = p_cfg.get("clip_percentiles", [1.0, 99.0])
        if not isinstance(percentiles, list) or len(percentiles) != 2:
            raise ValueError("data.preprocessing.team_v2.clip_percentiles must be a 2-element list")
        clip_low = float(percentiles[0])
        clip_high = float(percentiles[1])
        if clip_low >= clip_high:
            raise ValueError("data.preprocessing.team_v2.clip_percentiles must satisfy low < high")

        thorax_blur = int(p_cfg.get("thorax_blur", 51))
        thorax_blur = thorax_blur if thorax_blur % 2 == 1 else thorax_blur + 1
        thorax_blur = max(3, thorax_blur)

        gaussian_kernel_size = int(p_cfg.get("gaussian_kernel_size", 3))
        gaussian_kernel_size = gaussian_kernel_size if gaussian_kernel_size % 2 == 1 else gaussian_kernel_size + 1
        gaussian_kernel_size = max(1, gaussian_kernel_size)

        resize_interpolation = str(p_cfg.get("resize_interpolation", "area")).lower()
        if resize_interpolation not in {"area", "linear"}:
            raise ValueError("data.preprocessing.team_v2.resize_interpolation must be one of: area, linear")

        return TeamV2Settings(
            clip_low=clip_low,
            clip_high=clip_high,
            use_thorax_mask=bool(p_cfg.get("use_thorax_mask", True)),
            thorax_margin=float(p_cfg.get("thorax_margin", 0.97)),
            thorax_blur=thorax_blur,
            use_clahe=bool(p_cfg.get("use_clahe", True)),
            clahe_clip_limit=float(p_cfg.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=int(p_cfg.get("clahe_tile_grid_size", 8)),
            use_gaussian_blur=bool(p_cfg.get("use_gaussian_blur", True)),
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=float(p_cfg.get("gaussian_sigma", 0.5)),
            resize_interpolation=resize_interpolation,
        )


class TeamV2Preprocessor:
    """EDA-inspired robust preprocessing pipeline with optional CLAHE.

    Pipeline:
    1) read `IMREAD_UNCHANGED`
    2) robust percentile normalization
    3) optional thorax mask
    4) uint8 conversion
    5) optional CLAHE
    6) optional light Gaussian blur
    7) resize with aspect-ratio preserving reflection padding
    8) map to tensor in [0, 1], grayscale replicated to 3 channels

    Note:
    - This preprocessor intentionally does not apply train-time augmentation.
    - It also does not normalize outputs; normalization is applied downstream
      so augmentation order can stay: preprocess -> augment -> normalize.
    """

    def __init__(
        self,
        img_size: int,
        settings: TeamV2Settings,
    ) -> None:
        self.img_size = int(img_size)
        self.settings = settings

    @staticmethod
    def _require_cv2() -> None:
        if cv2 is None:
            raise RuntimeError(
                "team_v2 preprocessing requires OpenCV but `cv2` is not available. "
                "Install/load OpenCV first."
            )

    def _thorax_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Build smooth elliptical thorax mask."""
        h, w = shape
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        ry = (h / 2.0) * float(self.settings.thorax_margin)
        rx = (w / 2.0) * float(self.settings.thorax_margin)
        mask = (((y - cy) ** 2) / (ry**2 + 1e-6) + ((x - cx) ** 2) / (rx**2 + 1e-6) <= 1.0).astype(np.float32)
        return cv2.GaussianBlur(mask, (self.settings.thorax_blur, self.settings.thorax_blur), 0)

    def _resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        """Resize without geometric distortion and pad with reflection."""
        h, w = img.shape[:2]
        th, tw = self.img_size, self.img_size
        scale = min(float(tw) / max(1, w), float(th) / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_AREA if self.settings.resize_interpolation == "area" else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        pad_top = (th - new_h) // 2
        pad_bottom = th - new_h - pad_top
        pad_left = (tw - new_w) // 2
        pad_right = tw - new_w - pad_left

        return cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

    def __call__(self, image_path: Path) -> torch.Tensor:
        """Preprocess one image from disk and return `torch.FloatTensor[C,H,W]`."""
        self._require_cv2()
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_f = img.astype(np.float32)
        p_low, p_high = np.percentile(img_f, [self.settings.clip_low, self.settings.clip_high])
        if p_high <= p_low:
            p_high = p_low + 1e-6
        img_f = np.clip(img_f, p_low, p_high)
        img_f = (img_f - p_low) / (p_high - p_low + 1e-6)

        if self.settings.use_thorax_mask:
            mask = self._thorax_mask(img_f.shape)
            background = float(np.percentile(img_f, 5))
            img_f = img_f * mask + background * (1.0 - mask)

        img_u8 = np.clip(img_f * 255.0, 0.0, 255.0).astype(np.uint8)

        if self.settings.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.settings.clahe_clip_limit,
                tileGridSize=(self.settings.clahe_tile_grid_size, self.settings.clahe_tile_grid_size),
            )
            img_u8 = clahe.apply(img_u8)

        if self.settings.use_gaussian_blur and self.settings.gaussian_kernel_size > 1:
            img_u8 = cv2.GaussianBlur(
                img_u8,
                (self.settings.gaussian_kernel_size, self.settings.gaussian_kernel_size),
                sigmaX=self.settings.gaussian_sigma,
            )

        img_u8 = self._resize_with_padding(img_u8)
        tensor = torch.from_numpy(img_u8.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)

        return tensor
