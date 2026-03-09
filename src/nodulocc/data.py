"""Data loading and dataset construction for classification.

This module handles CSV parsing, source routing, split persistence, dataset
construction, and dataloader creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .preprocess_profiles import TeamV2Preprocessor, TeamV2Settings

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency, handled at runtime.
    cv2 = None


LABEL_MAP = {"No Finding": 0, "Nodule": 1}


def _resolve(path_root: Path, value: str) -> Path:
    """Resolve a relative path against `path_root`; keep absolute paths unchanged."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (path_root / candidate)


def _load_image(path: Path) -> Image.Image:
    """Load an image as grayscale (`L`) to keep preprocessing consistent."""
    with Image.open(path) as img:
        if img.mode != "L":
            return img.convert("L")
        return img.copy()


def _classification_dirs(cfg: dict[str, Any], root: Path) -> dict[str, Path]:
    """Return source-to-directory mapping for classification images.

    Preferred config:
    data:
      classification:
        nih_images_subdir: nih_filtered_images
        lidc_images_subdir: lidc_png_16_bit

    Legacy fallback:
    data:
      classification:
        images_subdir: nih_filtered_images
    """
    cls_cfg = cfg.get("data", {}).get("classification", {})
    nih_subdir = cls_cfg.get("nih_images_subdir")
    lidc_subdir = cls_cfg.get("lidc_images_subdir")
    if isinstance(nih_subdir, str) and isinstance(lidc_subdir, str):
        return {
            "NIH": _resolve(root, nih_subdir),
            "LIDC": _resolve(root, lidc_subdir),
        }

    legacy_subdir = cls_cfg.get("images_subdir")
    if not isinstance(legacy_subdir, str):
        raise ValueError(
            "Missing classification image directory config. "
            "Set `data.classification.images_subdir` or both "
            "`data.classification.nih_images_subdir` and "
            "`data.classification.lidc_images_subdir`."
        )
    legacy_dir = _resolve(root, legacy_subdir)
    return {"NIH": legacy_dir, "LIDC": legacy_dir}


def _resolve_image_paths_by_source(df: pd.DataFrame, source_dirs: dict[str, Path]) -> pd.Series:
    """Resolve image paths using source-aware routing with recursive fallback.

    The fast path assumes flat directories (`dir / file_name`). If files are
    nested, we lazily build a recursive filename index for missing rows.
    """
    image_path = pd.Series(index=df.index, dtype=object)
    file_names = df["file_name"].astype(str)
    sources = df["source"].astype(str)

    for source_name, source_dir in source_dirs.items():
        mask = sources == source_name
        if not bool(mask.any()):
            continue

        default_paths = file_names[mask].map(lambda x: str(source_dir / x))
        exists_default = default_paths.map(lambda x: Path(x).is_file())
        resolved = default_paths.copy()

        if bool((~exists_default).any()):
            # Build index only when needed to keep startup fast.
            recursive_index: dict[str, str] = {}
            for p in source_dir.rglob("*.png"):
                recursive_index.setdefault(p.name, str(p))
            fallback = file_names[mask].map(recursive_index.get)
            resolved = resolved.where(exists_default, fallback)

        image_path.loc[mask] = resolved

    return image_path


def _normalization_stats(cfg: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    """Return normalization stats from config or None when disabled.

    Expected config block:
    data:
      normalization:
        enabled: true
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    """
    norm_cfg = cfg.get("data", {}).get("normalization", {})
    if not bool(norm_cfg.get("enabled", True)):
        return None

    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
        raise ValueError("data.normalization.mean/std must be 3-element lists")

    return [float(x) for x in mean], [float(x) for x in std]


def _model_type(cfg: dict[str, Any]) -> str:
    """Return model type (`global` or `mil_patch`)."""
    model_type = str(cfg.get("model", {}).get("type", "global")).lower()
    if model_type not in {"global", "mil_patch"}:
        raise ValueError("model.type must be one of: global, mil_patch")
    return model_type


def _mil_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return validated MIL settings from `model.mil`."""
    mil_cfg = cfg.get("model", {}).get("mil", {})
    num_patches = int(mil_cfg.get("num_patches", 16))
    patch_size = int(mil_cfg.get("patch_size", 160))
    min_scale = float(mil_cfg.get("min_scale", 0.9))
    max_scale = float(mil_cfg.get("max_scale", 1.8))
    use_loc = bool(mil_cfg.get("use_localization_priors", True))
    pos_patch_prob = float(mil_cfg.get("positive_patch_prob", 0.7))
    loc_jitter = float(mil_cfg.get("localization_jitter", 0.05))
    sampling_mode = str(mil_cfg.get("sampling_mode", "guided")).lower()
    candidate_grid = int(mil_cfg.get("candidate_grid", 9))
    topk_fraction = float(mil_cfg.get("topk_fraction", 0.35))
    train_explore_prob = float(mil_cfg.get("train_explore_prob", 0.2))
    guided_jitter = float(mil_cfg.get("guided_jitter", 0.02))
    score_kernel_size = int(mil_cfg.get("score_kernel_size", 21))

    if num_patches < 1:
        raise ValueError("model.mil.num_patches must be >= 1")
    if patch_size < 32:
        raise ValueError("model.mil.patch_size must be >= 32")
    if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
        raise ValueError("model.mil.min_scale/max_scale must satisfy 0 < min_scale <= max_scale")
    if pos_patch_prob < 0.0 or pos_patch_prob > 1.0:
        raise ValueError("model.mil.positive_patch_prob must be in [0, 1]")
    if loc_jitter < 0.0:
        raise ValueError("model.mil.localization_jitter must be >= 0")
    if sampling_mode not in {"guided", "random"}:
        raise ValueError("model.mil.sampling_mode must be one of: guided, random")
    if candidate_grid < 3:
        raise ValueError("model.mil.candidate_grid must be >= 3")
    if topk_fraction <= 0.0 or topk_fraction > 1.0:
        raise ValueError("model.mil.topk_fraction must be in (0, 1]")
    if train_explore_prob < 0.0 or train_explore_prob > 1.0:
        raise ValueError("model.mil.train_explore_prob must be in [0, 1]")
    if guided_jitter < 0.0:
        raise ValueError("model.mil.guided_jitter must be >= 0")
    if score_kernel_size < 3:
        raise ValueError("model.mil.score_kernel_size must be >= 3")
    if score_kernel_size % 2 == 0:
        score_kernel_size += 1

    return {
        "num_patches": num_patches,
        "patch_size": patch_size,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "use_localization_priors": use_loc,
        "positive_patch_prob": pos_patch_prob,
        "localization_jitter": loc_jitter,
        "sampling_mode": sampling_mode,
        "candidate_grid": candidate_grid,
        "topk_fraction": topk_fraction,
        "train_explore_prob": train_explore_prob,
        "guided_jitter": guided_jitter,
        "score_kernel_size": score_kernel_size,
    }


def _load_localization_points(cfg: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """Load localization CSV and return raw `(x, y)` points grouped by file name."""
    cls_cfg = cfg.get("data", {}).get("classification", {})
    loc_csv = str(cls_cfg.get("localization_csv", "localization_labels.csv"))
    root = Path(cfg["data"]["dataset_root"]).resolve()
    loc_path = _resolve(root, loc_csv)
    if not loc_path.is_file():
        return {}

    loc_df = pd.read_csv(loc_path)
    required = {"file_name", "x", "y"}
    if not required.issubset(loc_df.columns):
        return {}

    loc_df = loc_df[["file_name", "x", "y"]].copy()
    loc_df["x"] = pd.to_numeric(loc_df["x"], errors="coerce")
    loc_df["y"] = pd.to_numeric(loc_df["y"], errors="coerce")
    loc_df = loc_df.dropna(subset=["file_name", "x", "y"])
    if len(loc_df) == 0:
        return {}

    points_by_file: dict[str, list[tuple[float, float]]] = {}
    for file_name, part in loc_df.groupby("file_name"):
        xs = part["x"].to_numpy(dtype=np.float32)
        ys = part["y"].to_numpy(dtype=np.float32)
        points_by_file[str(file_name)] = list(zip(xs.tolist(), ys.tolist()))
    return points_by_file


def _split_stratify_mode(cfg: dict[str, Any]) -> str:
    """Return split stratification mode: `label` or `source_label`."""
    mode = str(cfg.get("data", {}).get("split_stratify", "source_label")).lower()
    if mode not in {"label", "source_label"}:
        raise ValueError("data.split_stratify must be one of: label, source_label")
    return mode


def _lidc_train_only(cfg: dict[str, Any]) -> bool:
    """Whether LIDC samples must be forced into train split only."""
    return bool(cfg.get("data", {}).get("classification", {}).get("lidc_train_only", False))


def _split_cache_suffix(cfg: dict[str, Any]) -> str:
    """Build a split-cache suffix that encodes split semantics."""
    strat = _split_stratify_mode(cfg)
    lidc_flag = int(_lidc_train_only(cfg))
    return f"strat-{strat}_lidc-train-{lidc_flag}"


def _split_targets(df: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """Build stratification targets according to configured split strategy."""
    mode = _split_stratify_mode(cfg)
    if mode == "label":
        return df["label"].astype(str).to_numpy()
    if "source" not in df.columns:
        raise ValueError("`source` column is required for data.split_stratify=source_label")
    return (df["source"].astype(str) + "_" + df["label"].astype(str)).to_numpy()


def _split_pool_indices(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return `(split_pool_idx, always_train_idx)` according to split policy."""
    all_idx = np.arange(len(df), dtype=np.int64)
    if not _lidc_train_only(cfg):
        return all_idx, np.array([], dtype=np.int64)
    if "source" not in df.columns:
        raise ValueError("`source` column is required for data.classification.lidc_train_only=true")

    source_values = df["source"].astype(str).to_numpy()
    always_train_idx = np.where(source_values == "LIDC")[0].astype(np.int64)
    split_pool_idx = np.where(source_values != "LIDC")[0].astype(np.int64)
    if len(split_pool_idx) == 0:
        raise ValueError("No non-LIDC samples available for validation split.")
    return split_pool_idx, always_train_idx


def _preprocessing_profile(cfg: dict[str, Any]) -> str:
    """Return selected preprocessing profile."""
    profile = str(cfg.get("data", {}).get("preprocessing", {}).get("profile", "v1")).lower()
    if profile not in {"v1", "team_v2"}:
        raise ValueError("data.preprocessing.profile must be one of: v1, team_v2")
    return profile


def _clahe_settings(cfg: dict[str, Any]) -> tuple[float, int] | None:
    """Return CLAHE settings from config or None when disabled.

    Expected config block:
    data:
      preprocessing:
        clahe:
          enabled: false
          clip_limit: 2.0
          tile_grid_size: 8
    """
    if _preprocessing_profile(cfg) != "v1":
        return None

    clahe_cfg = cfg.get("data", {}).get("preprocessing", {}).get("clahe", {})
    if not bool(clahe_cfg.get("enabled", False)):
        return None

    clip_limit = float(clahe_cfg.get("clip_limit", 2.0))
    tile_grid_size = int(clahe_cfg.get("tile_grid_size", 8))
    if clip_limit <= 0:
        raise ValueError("data.preprocessing.clahe.clip_limit must be > 0")
    if tile_grid_size < 1:
        raise ValueError("data.preprocessing.clahe.tile_grid_size must be >= 1")
    return clip_limit, tile_grid_size


def _augmentation_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return train-time augmentation settings.

    Expected config block:
    data:
      augmentation:
        horizontal_flip_p: 0.5
        color_jitter:
          enabled: true
          brightness: 0.06
          contrast: 0.10
          saturation: 0.0
          hue: 0.0
    """
    aug_cfg = cfg.get("data", {}).get("augmentation", {})
    horizontal_flip_p = float(aug_cfg.get("horizontal_flip_p", 0.5))
    if horizontal_flip_p < 0.0 or horizontal_flip_p > 1.0:
        raise ValueError("data.augmentation.horizontal_flip_p must be in [0, 1]")

    jitter_cfg = aug_cfg.get("color_jitter", {})
    enabled = bool(jitter_cfg.get("enabled", True))
    brightness = float(jitter_cfg.get("brightness", 0.06))
    contrast = float(jitter_cfg.get("contrast", 0.10))
    saturation = float(jitter_cfg.get("saturation", 0.0))
    hue = float(jitter_cfg.get("hue", 0.0))

    for key, value in {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
    }.items():
        if value < 0.0:
            raise ValueError(f"data.augmentation.color_jitter.{key} must be >= 0")
    if hue < 0.0 or hue > 0.5:
        raise ValueError("data.augmentation.color_jitter.hue must be in [0, 0.5]")

    return {
        "horizontal_flip_p": horizontal_flip_p,
        "color_jitter_enabled": enabled,
        "color_jitter_brightness": brightness,
        "color_jitter_contrast": contrast,
        "color_jitter_saturation": saturation,
        "color_jitter_hue": hue,
    }


def _build_train_augment_ops(augmentation_settings: dict[str, Any]) -> list[Any]:
    """Build train-time augmentation operations for PIL images or tensors."""
    ops: list[Any] = []
    hflip_p = float(augmentation_settings["horizontal_flip_p"])
    if hflip_p > 0.0:
        ops.append(transforms.RandomHorizontalFlip(p=hflip_p))

    if bool(augmentation_settings["color_jitter_enabled"]):
        ops.append(
            transforms.ColorJitter(
                brightness=float(augmentation_settings["color_jitter_brightness"]),
                contrast=float(augmentation_settings["color_jitter_contrast"]),
                saturation=float(augmentation_settings["color_jitter_saturation"]),
                hue=float(augmentation_settings["color_jitter_hue"]),
            )
        )
    return ops


def _normalization_transform(normalization: tuple[list[float], list[float]] | None) -> transforms.Normalize | None:
    """Return torchvision normalization transform when normalization is enabled."""
    if normalization is None:
        return None
    mean, std = normalization
    return transforms.Normalize(mean=mean, std=std)


class CLAHETransform:
    """Apply OpenCV CLAHE on a grayscale PIL image.

    This transform is intentionally implemented as a top-level class so it
    remains picklable with dataloader multiprocessing.
    """

    def __init__(self, clip_limit: float, tile_grid_size: int) -> None:
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = int(tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        if cv2 is None:
            raise RuntimeError(
                "CLAHE is enabled but OpenCV is not installed. "
                "Install it with: uv sync --extra cv"
            )

        gray = img if img.mode == "L" else img.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size),
        )
        out = clahe.apply(arr)
        return Image.fromarray(out, mode="L")


def _image_tfm(
    img_size: int,
    train: bool,
    normalization: tuple[list[float], list[float]] | None,
    clahe: tuple[float, int] | None,
    augmentation_settings: dict[str, Any],
) -> transforms.Compose:
    """Build image transforms for classification."""
    ops: list[Any] = [transforms.Resize((img_size, img_size))]
    if clahe is not None:
        clip_limit, tile_grid_size = clahe
        ops.append(CLAHETransform(clip_limit=clip_limit, tile_grid_size=tile_grid_size))
    if train:
        ops.extend(_build_train_augment_ops(augmentation_settings))
    ops.extend(
        [
            # Use torchvision native transform instead of a local lambda so the
            # dataset stays picklable with Python 3.14 forkserver workers.
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    normalize = _normalization_transform(normalization)
    if normalize is not None:
        ops.append(normalize)
    return transforms.Compose(ops)


def load_classification_dataframe(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and sanitize classification metadata.

    Returns a DataFrame with numeric label and resolved `image_path`, filtered
    to rows whose image file exists.
    """
    root = Path(cfg["data"]["dataset_root"]).resolve()
    labels_path = _resolve(root, cfg["data"]["classification"]["labels_csv"])
    source_dirs = _classification_dirs(cfg, root)

    df = pd.read_csv(labels_path)
    required = {"file_name", "label", "LIDC_ID"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing classification columns: expected {required}")

    df = df.copy()
    df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Unknown labels found in classification CSV")

    # Source routing: LIDC rows (has LIDC_ID) map to LIDC directory.
    df["source"] = np.where(df["LIDC_ID"].notna(), "LIDC", "NIH")
    df["image_path"] = _resolve_image_paths_by_source(df, source_dirs)
    exists = df["image_path"].map(lambda x: isinstance(x, str) and Path(x).is_file())
    df = df[exists].reset_index(drop=True)
    df["label"] = df["label"].astype(np.int64)
    return df


def _validation_mode(cfg: dict[str, Any]) -> str:
    """Return validation mode from config (`holdout` or `kfold`)."""
    mode = str(cfg.get("validation", {}).get("mode", "holdout")).lower()
    if mode not in {"holdout", "kfold"}:
        raise ValueError("validation.mode must be either 'holdout' or 'kfold'")
    return mode


def _split_dir(cfg: dict[str, Any]) -> Path:
    """Return and create split directory."""
    split_dir = Path(cfg["data"].get("split_dir", "artifacts/splits"))
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def _holdout_split_file(cfg: dict[str, Any]) -> Path:
    """Return split file path used for reproducible holdout split."""
    suffix = _split_cache_suffix(cfg)
    return _split_dir(cfg) / f"classification_holdout_seed{int(cfg.get('seed', 42))}_{suffix}.json"


def _kfold_split_file(cfg: dict[str, Any], k: int) -> Path:
    """Return split file path used for reproducible k-fold split."""
    suffix = _split_cache_suffix(cfg)
    return _split_dir(cfg) / f"classification_kfold_k{k}_seed{int(cfg.get('seed', 42))}_{suffix}.json"


def _holdout_indices(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build or load cached train/val indices for holdout mode."""
    split_path = _holdout_split_file(cfg)
    val_size = float(cfg["data"].get("val_size", 0.2))

    if split_path.is_file():
        payload = json.loads(split_path.read_text())
        return np.array(payload["train_idx"], dtype=np.int64), np.array(payload["val_idx"], dtype=np.int64)

    split_pool_idx, always_train_idx = _split_pool_indices(df, cfg)
    split_pool_df = df.iloc[split_pool_idx].reset_index(drop=True)
    split_targets = _split_targets(split_pool_df, cfg)

    train_idx, val_idx = train_test_split(
        split_pool_idx,
        test_size=val_size,
        random_state=int(cfg.get("seed", 42)),
        stratify=split_targets,
    )
    if len(always_train_idx) > 0:
        train_idx = np.concatenate([train_idx, always_train_idx]).astype(np.int64)
    split_path.write_text(
        json.dumps(
            {
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
            },
            indent=2,
        )
    )
    return train_idx, val_idx


def _kfold_indices(df: pd.DataFrame, cfg: dict[str, Any], k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build or load cached fold indices for stratified k-fold mode."""
    if k < 2:
        raise ValueError("validation.k must be >= 2 for kfold mode")

    split_path = _kfold_split_file(cfg, k)
    if split_path.is_file():
        payload = json.loads(split_path.read_text())
        folds = []
        for fold in payload["folds"]:
            folds.append(
                (
                    np.array(fold["train_idx"], dtype=np.int64),
                    np.array(fold["val_idx"], dtype=np.int64),
                )
            )
        return folds

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(cfg.get("seed", 42)))
    split_pool_idx, always_train_idx = _split_pool_indices(df, cfg)
    split_pool_df = df.iloc[split_pool_idx].reset_index(drop=True)
    split_targets = _split_targets(split_pool_df, cfg)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    serializable: list[dict[str, list[int]]] = []
    for train_pos, val_pos in skf.split(np.arange(len(split_pool_idx)), split_targets):
        train_idx = split_pool_idx[train_pos].astype(np.int64)
        val_idx = split_pool_idx[val_pos].astype(np.int64)
        if len(always_train_idx) > 0:
            train_idx = np.concatenate([train_idx, always_train_idx]).astype(np.int64)
        folds.append((train_idx, val_idx))
        serializable.append(
            {
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
            }
        )

    split_path.write_text(json.dumps({"k": k, "folds": serializable}, indent=2))
    return folds


def _split_dataframe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    fold_index: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe according to configured validation mode."""
    mode = _validation_mode(cfg)

    if mode == "holdout":
        train_idx, val_idx = _holdout_indices(df, cfg)
    else:
        k = int(cfg.get("validation", {}).get("k", 5))
        folds = _kfold_indices(df, cfg, k)
        selected_fold = int(cfg.get("validation", {}).get("fold_index", 0)) if fold_index is None else int(fold_index)
        if selected_fold < 0 or selected_fold >= k:
            raise ValueError(f"Fold index out of range: {selected_fold}. Expected in [0, {k - 1}]")
        train_idx, val_idx = folds[selected_fold]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


class ClassificationDataset(Dataset):
    """PyTorch dataset for binary classification samples."""

    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int,
        train: bool,
        normalization: tuple[list[float], list[float]] | None,
        clahe: tuple[float, int] | None,
        augmentation_settings: dict[str, Any],
        preprocessing_profile: str,
        cfg: dict[str, Any],
    ) -> None:
        """Store metadata and build per-split transforms.

        We materialize file paths and labels as arrays once to reduce per-sample
        pandas indexing overhead in `__getitem__`.
        """
        self.df = df.reset_index(drop=True)
        self._image_paths = self.df["image_path"].astype(str).tolist()
        self._file_names = self.df["file_name"].astype(str).tolist()
        self._labels = self.df["label"].astype(np.float32).to_numpy()
        self._preprocessing_profile = preprocessing_profile
        self._normalize = _normalization_transform(normalization)
        if preprocessing_profile == "team_v2":
            self._team_v2_preprocessor = TeamV2Preprocessor(
                img_size=img_size,
                settings=TeamV2Settings.from_cfg(cfg),
            )
            train_aug_ops = _build_train_augment_ops(augmentation_settings) if train else []
            self._tensor_aug = transforms.Compose(train_aug_ops) if len(train_aug_ops) > 0 else None
            self.tfm = None
        else:
            self._team_v2_preprocessor = None
            self._tensor_aug = None
            self.tfm = _image_tfm(
                img_size=img_size,
                train=train,
                normalization=normalization,
                clahe=clahe,
                augmentation_settings=augmentation_settings,
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one sample with image tensor and binary label."""
        if self._team_v2_preprocessor is not None:
            image = self._team_v2_preprocessor(Path(self._image_paths[idx]))
            if self._tensor_aug is not None:
                image = self._tensor_aug(image)
            if self._normalize is not None:
                image = self._normalize(image)
        else:
            img = _load_image(Path(self._image_paths[idx]))
            image = self.tfm(img)

        return {
            "image": image,
            "label": torch.tensor(float(self._labels[idx]), dtype=torch.float32),
            "file_name": self._file_names[idx],
        }


class MilPatchDataset(Dataset):
    """MIL dataset returning a bag of patches per image.

    Each sample contains:
    - `image`: Tensor `[N, 3, P, P]` where `N=num_patches`
    - `label`: scalar tensor
    - `file_name`: source file name
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int,
        train: bool,
        normalization: tuple[list[float], list[float]] | None,
        clahe: tuple[float, int] | None,
        augmentation_settings: dict[str, Any],
        preprocessing_profile: str,
        cfg: dict[str, Any],
    ) -> None:
        self._train = bool(train)
        self._seed = int(cfg.get("seed", 42))
        self.df = df.reset_index(drop=True)
        self._mil = _mil_settings(cfg)
        self._num_patches = int(self._mil["num_patches"])
        self._patch_size = int(self._mil["patch_size"])
        self._min_scale = float(self._mil["min_scale"])
        self._max_scale = float(self._mil["max_scale"])
        self._positive_patch_prob = float(self._mil["positive_patch_prob"])
        self._localization_jitter = float(self._mil["localization_jitter"])
        self._sampling_mode = str(self._mil["sampling_mode"])
        self._candidate_grid = int(self._mil["candidate_grid"])
        self._topk_fraction = float(self._mil["topk_fraction"])
        self._train_explore_prob = float(self._mil["train_explore_prob"])
        self._guided_jitter = float(self._mil["guided_jitter"])
        self._score_kernel_size = int(self._mil["score_kernel_size"])
        self._eval_scale = float(0.5 * (self._min_scale + self._max_scale))
        self._normalization = normalization
        if normalization is not None:
            mean, std = normalization
            self._norm_mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self._norm_std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        else:
            self._norm_mean_t = None
            self._norm_std_t = None

        # Reuse existing full-image preprocessing/augmentation pipeline.
        self._base_ds = ClassificationDataset(
            self.df,
            img_size=img_size,
            train=train,
            normalization=normalization,
            clahe=clahe,
            augmentation_settings=augmentation_settings,
            preprocessing_profile=preprocessing_profile,
            cfg=cfg,
        )

        self._loc_norm_by_file: dict[str, list[tuple[float, float]]] = {}
        if self._train and bool(self._mil["use_localization_priors"]):
            raw_points = _load_localization_points(cfg)
            if raw_points:
                self._loc_norm_by_file = self._normalize_localization_points(raw_points)

    def _normalize_localization_points(
        self,
        raw_points: dict[str, list[tuple[float, float]]],
    ) -> dict[str, list[tuple[float, float]]]:
        """Normalize raw pixel coordinates to [0, 1] using original image size."""
        out: dict[str, list[tuple[float, float]]] = {}
        for row in self.df.itertuples(index=False):
            file_name = str(row.file_name)
            points = raw_points.get(file_name)
            if not points:
                continue

            try:
                with Image.open(Path(str(row.image_path))) as img:
                    w, h = img.size
            except Exception:
                continue
            if w < 2 or h < 2:
                continue

            norm_points: list[tuple[float, float]] = []
            for x, y in points:
                nx = float(np.clip(float(x) / float(w - 1), 0.0, 1.0))
                ny = float(np.clip(float(y) / float(h - 1), 0.0, 1.0))
                norm_points.append((nx, ny))
            if norm_points:
                out[file_name] = norm_points
        return out

    def __len__(self) -> int:
        return len(self._base_ds)

    @staticmethod
    def _sample_center_in_thorax(rng: np.random.Generator) -> tuple[float, float]:
        """Sample a center point in a coarse thorax ellipse."""
        for _ in range(24):
            x = float(rng.uniform(0.1, 0.9))
            y = float(rng.uniform(0.1, 0.9))
            in_ellipse = (((x - 0.5) / 0.45) ** 2 + ((y - 0.5) / 0.45) ** 2) <= 1.0
            if in_ellipse:
                return x, y
        return 0.5, 0.5

    def _rng_for_item(self, idx: int, file_name: str) -> np.random.Generator:
        """Build item RNG (stochastic for train, deterministic for eval/predict)."""
        if self._train:
            return np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
        key = zlib.crc32(file_name.encode("utf-8")) & 0xFFFFFFFF
        mixed = (self._seed * 1664525 + int(idx) * 1013904223 + key) % (2**32)
        return np.random.default_rng(mixed)

    def _score_view(self, image: torch.Tensor) -> torch.Tensor:
        """Build [H,W] view in [0,1] used for guided patch scoring."""
        x = image.detach().float().cpu()
        if self._norm_mean_t is not None and self._norm_std_t is not None:
            x = x * self._norm_std_t + self._norm_mean_t
        x = x.clamp(0.0, 1.0)
        return x.mean(dim=0)

    def _guided_candidate_centers(self, image: torch.Tensor) -> list[tuple[float, float, float]]:
        """Compute candidate centers ranked by local variance score."""
        gray = self._score_view(image)
        k = int(self._score_kernel_size)
        mean = F.avg_pool2d(gray.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=k // 2)
        mean2 = F.avg_pool2d((gray * gray).unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=k // 2)
        var_map = torch.clamp(mean2 - mean * mean, min=0.0).squeeze(0).squeeze(0)
        h, w = int(var_map.shape[0]), int(var_map.shape[1])

        xs = np.linspace(0.08, 0.92, self._candidate_grid, dtype=np.float32)
        ys = np.linspace(0.08, 0.92, self._candidate_grid, dtype=np.float32)
        candidates: list[tuple[float, float, float]] = []
        for y in ys:
            for x in xs:
                in_ellipse = (((float(x) - 0.5) / 0.45) ** 2 + ((float(y) - 0.5) / 0.45) ** 2) <= 1.0
                if not in_ellipse:
                    continue
                px = int(round(float(x) * (w - 1)))
                py = int(round(float(y) * (h - 1)))
                score = float(var_map[py, px].item())
                candidates.append((float(x), float(y), score))

        if not candidates:
            return [(0.5, 0.5, 0.0)]
        candidates.sort(key=lambda t: t[2], reverse=True)
        return candidates

    def _extract_patch(
        self,
        image: torch.Tensor,
        center_x: float,
        center_y: float,
        crop_size: int,
    ) -> torch.Tensor:
        """Extract one square crop then resize to MIL patch size."""
        _, h, w = image.shape
        crop = max(8, min(int(crop_size), h, w))
        cx = int(round(center_x * (w - 1)))
        cy = int(round(center_y * (h - 1)))
        x0 = max(0, min(w - crop, cx - crop // 2))
        y0 = max(0, min(h - crop, cy - crop // 2))
        patch = image[:, y0 : y0 + crop, x0 : x0 + crop]
        if patch.shape[1] != self._patch_size or patch.shape[2] != self._patch_size:
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(self._patch_size, self._patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return patch

    def _sample_patch_centers(
        self,
        *,
        image: torch.Tensor,
        label: float,
        file_name: str,
        rng: np.random.Generator,
    ) -> list[tuple[float, float]]:
        """Sample patch centers with guided or random strategy plus optional localization prior."""
        centers: list[tuple[float, float]] = []

        loc_points = self._loc_norm_by_file.get(file_name, [])
        if label > 0.5 and loc_points and float(rng.uniform(0.0, 1.0)) <= self._positive_patch_prob:
            x0, y0 = loc_points[int(rng.integers(0, len(loc_points)))]
            x = float(np.clip(x0 + rng.normal(0.0, self._localization_jitter), 0.0, 1.0))
            y = float(np.clip(y0 + rng.normal(0.0, self._localization_jitter), 0.0, 1.0))
            centers.append((x, y))

        if self._sampling_mode == "guided":
            candidates = self._guided_candidate_centers(image)
            top_n = max(1, int(round(len(candidates) * self._topk_fraction)))
            top_candidates = candidates[:top_n]

            if self._train:
                while len(centers) < self._num_patches:
                    if float(rng.uniform(0.0, 1.0)) < self._train_explore_prob:
                        x, y = self._sample_center_in_thorax(rng)
                    else:
                        cx, cy, _ = top_candidates[int(rng.integers(0, len(top_candidates)))]
                        x = float(np.clip(cx + rng.normal(0.0, self._guided_jitter), 0.0, 1.0))
                        y = float(np.clip(cy + rng.normal(0.0, self._guided_jitter), 0.0, 1.0))
                    centers.append((x, y))
            else:
                for cx, cy, _ in top_candidates:
                    centers.append((float(cx), float(cy)))
                    if len(centers) >= self._num_patches:
                        break
                if len(centers) < self._num_patches:
                    for cx, cy, _ in candidates[top_n:]:
                        centers.append((float(cx), float(cy)))
                        if len(centers) >= self._num_patches:
                            break

        while len(centers) < self._num_patches:
            centers.append(self._sample_center_in_thorax(rng))
        return centers

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_item = self._base_ds[idx]
        image = base_item["image"]
        label_t = base_item["label"]
        file_name = str(base_item["file_name"])
        rng = self._rng_for_item(idx, file_name)
        centers = self._sample_patch_centers(
            image=image,
            label=float(label_t.item()),
            file_name=file_name,
            rng=rng,
        )

        patches: list[torch.Tensor] = []
        for cx, cy in centers:
            scale = float(rng.uniform(self._min_scale, self._max_scale)) if self._train else self._eval_scale
            crop_size = int(round(self._patch_size * scale))
            patches.append(self._extract_patch(image, center_x=cx, center_y=cy, crop_size=crop_size))
        patch_tensor = torch.stack(patches, dim=0)

        return {
            "image": patch_tensor,
            "label": label_t,
            "file_name": file_name,
        }


def _build_dataset(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    *,
    img_size: int,
    train: bool,
    normalization: tuple[list[float], list[float]] | None,
    clahe: tuple[float, int] | None,
    augmentation_settings: dict[str, Any],
    preprocessing_profile: str,
) -> Dataset:
    """Build global-image dataset or MIL patch dataset depending on model type."""
    model_type = _model_type(cfg)
    if model_type == "mil_patch":
        return MilPatchDataset(
            df,
            img_size=img_size,
            train=train,
            normalization=normalization,
            clahe=clahe,
            augmentation_settings=augmentation_settings,
            preprocessing_profile=preprocessing_profile,
            cfg=cfg,
        )
    return ClassificationDataset(
        df,
        img_size=img_size,
        train=train,
        normalization=normalization,
        clahe=clahe,
        augmentation_settings=augmentation_settings,
        preprocessing_profile=preprocessing_profile,
        cfg=cfg,
    )


def _classification_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Build class-balanced sampler to mitigate strong label imbalance."""
    labels = df["label"].to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    inv = 1.0 / np.maximum(counts, 1)
    sample_w = inv[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def prepare_frames(
    cfg: dict[str, Any],
    fold_index: int | None = None,
    full_train: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load dataframe and return train/val frames for requested mode.

    If `full_train` is True, the full dataset is returned as train and val is None.
    """
    df = load_classification_dataframe(cfg)
    if full_train:
        return df, None
    return _split_dataframe(df, cfg, fold_index=fold_index)


def build_dataloaders(
    cfg: dict[str, Any],
    fold_index: int | None = None,
    full_train: bool = False,
) -> tuple[DataLoader, DataLoader | None, pd.DataFrame, pd.DataFrame | None]:
    """Create train/validation dataloaders for classification."""
    train_df, val_df = prepare_frames(cfg, fold_index=fold_index, full_train=full_train)
    batch_size = int(cfg["train"]["batch_size"])
    workers = int(cfg["train"].get("num_workers", 8))
    img_size = int(cfg["train"].get("img_size", 512))
    normalization = _normalization_stats(cfg)
    augmentation_settings = _augmentation_settings(cfg)
    preprocessing_profile = _preprocessing_profile(cfg)
    clahe = _clahe_settings(cfg)
    pin_memory = bool(cfg["train"].get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(cfg["train"].get("persistent_workers", True))
    prefetch_factor = int(cfg["train"].get("prefetch_factor", 4))
    drop_last_train = bool(cfg["train"].get("drop_last_train", False))

    train_ds = _build_dataset(
        cfg,
        train_df,
        img_size=img_size,
        train=True,
        normalization=normalization,
        clahe=clahe,
        augmentation_settings=augmentation_settings,
        preprocessing_profile=preprocessing_profile,
    )
    sampler = None
    if bool(cfg["train"].get("use_weighted_sampler", True)):
        sampler = _classification_sampler(train_df)

    train_loader_kwargs: dict[str, Any] = {
        "dataset": train_ds,
        "batch_size": batch_size,
        "shuffle": (sampler is None),
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last_train,
    }
    if workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(**train_loader_kwargs)

    if val_df is None:
        return train_loader, None, train_df, None

    val_ds = _build_dataset(
        cfg,
        val_df,
        img_size=img_size,
        train=False,
        normalization=normalization,
        clahe=clahe,
        augmentation_settings=augmentation_settings,
        preprocessing_profile=preprocessing_profile,
    )
    val_loader_kwargs: dict[str, Any] = {
        "dataset": val_ds,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        val_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["prefetch_factor"] = prefetch_factor
    val_loader = DataLoader(**val_loader_kwargs)
    return train_loader, val_loader, train_df, val_df


def build_prediction_loader(
    cfg: dict[str, Any],
    split: str = "val",
    fold_index: int | None = None,
) -> DataLoader:
    """Create inference dataloader for `train`, `val`, or `all` split."""
    img_size = int(cfg["train"].get("img_size", 512))
    workers = int(cfg["train"].get("num_workers", 8))
    batch_size = int(cfg["train"]["batch_size"])
    normalization = _normalization_stats(cfg)
    augmentation_settings = _augmentation_settings(cfg)
    preprocessing_profile = _preprocessing_profile(cfg)
    clahe = _clahe_settings(cfg)
    pin_memory = bool(cfg["train"].get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(cfg["train"].get("persistent_workers", True))
    prefetch_factor = int(cfg["train"].get("prefetch_factor", 4))

    if split not in {"train", "val", "all"}:
        raise ValueError("split must be one of: train, val, all")

    if split == "all":
        train_df, _ = prepare_frames(cfg, full_train=True)
        df = train_df
    else:
        train_df, val_df = prepare_frames(cfg, fold_index=fold_index, full_train=False)
        if split == "train":
            df = train_df
        else:
            df = val_df
            if df is None:
                raise ValueError("No validation split available in full-data mode")

    ds = _build_dataset(
        cfg,
        df,
        img_size=img_size,
        train=False,
        normalization=normalization,
        clahe=clahe,
        augmentation_settings=augmentation_settings,
        preprocessing_profile=preprocessing_profile,
    )
    pred_loader_kwargs: dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        pred_loader_kwargs["persistent_workers"] = persistent_workers
        pred_loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**pred_loader_kwargs)
