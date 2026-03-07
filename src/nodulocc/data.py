"""Data loading and dataset construction for classification.

This module handles CSV parsing, source routing, split persistence, dataset
construction, and dataloader creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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

    train_ds = ClassificationDataset(
        train_df,
        img_size=img_size,
        train=True,
        normalization=normalization,
        clahe=clahe,
        augmentation_settings=augmentation_settings,
        preprocessing_profile=preprocessing_profile,
        cfg=cfg,
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

    val_ds = ClassificationDataset(
        val_df,
        img_size=img_size,
        train=False,
        normalization=normalization,
        clahe=clahe,
        augmentation_settings=augmentation_settings,
        preprocessing_profile=preprocessing_profile,
        cfg=cfg,
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

    ds = ClassificationDataset(
        df,
        img_size=img_size,
        train=False,
        normalization=normalization,
        clahe=clahe,
        augmentation_settings=augmentation_settings,
        preprocessing_profile=preprocessing_profile,
        cfg=cfg,
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
