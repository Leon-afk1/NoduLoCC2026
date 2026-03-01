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


def _image_tfm(
    img_size: int,
    train: bool,
    normalization: tuple[list[float], list[float]] | None,
) -> transforms.Compose:
    """Build image transforms for classification."""
    ops: list[Any] = [transforms.Resize((img_size, img_size))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.06, contrast=0.1),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
        ]
    )
    if normalization is not None:
        mean, std = normalization
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def load_classification_dataframe(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and sanitize classification metadata.

    Returns a DataFrame with numeric label and resolved `image_path`, filtered
    to rows whose image file exists.
    """
    root = Path(cfg["data"]["dataset_root"]).resolve()
    labels_path = _resolve(root, cfg["data"]["classification"]["labels_csv"])
    images_dir = _resolve(root, cfg["data"]["classification"]["images_subdir"])

    df = pd.read_csv(labels_path)
    required = {"file_name", "label", "LIDC_ID"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing classification columns: expected {required}")

    df = df.copy()
    df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Unknown labels found in classification CSV")

    df["image_path"] = df["file_name"].astype(str).map(lambda x: str(images_dir / x))
    exists = df["image_path"].map(lambda x: Path(x).is_file())
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
    return _split_dir(cfg) / f"classification_holdout_seed{int(cfg.get('seed', 42))}.json"


def _kfold_split_file(cfg: dict[str, Any], k: int) -> Path:
    """Return split file path used for reproducible k-fold split."""
    return _split_dir(cfg) / f"classification_kfold_k{k}_seed{int(cfg.get('seed', 42))}.json"


def _holdout_indices(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build or load cached train/val indices for holdout mode."""
    split_path = _holdout_split_file(cfg)
    val_size = float(cfg["data"].get("val_size", 0.2))

    if split_path.is_file():
        payload = json.loads(split_path.read_text())
        return np.array(payload["train_idx"], dtype=np.int64), np.array(payload["val_idx"], dtype=np.int64)

    indices = np.arange(len(df), dtype=np.int64)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=int(cfg.get("seed", 42)),
        stratify=df["label"],
    )
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
    labels = df["label"].to_numpy(dtype=np.int64)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    serializable: list[dict[str, list[int]]] = []
    for train_idx, val_idx in skf.split(np.arange(len(df)), labels):
        train_idx = train_idx.astype(np.int64)
        val_idx = val_idx.astype(np.int64)
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
    ) -> None:
        """Store metadata and build per-split transforms.

        We materialize file paths and labels as arrays once to reduce per-sample
        pandas indexing overhead in `__getitem__`.
        """
        self.df = df.reset_index(drop=True)
        self._image_paths = self.df["image_path"].astype(str).tolist()
        self._file_names = self.df["file_name"].astype(str).tolist()
        self._labels = self.df["label"].astype(np.float32).to_numpy()
        self.tfm = _image_tfm(img_size=img_size, train=train, normalization=normalization)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one sample with image tensor and binary label."""
        img = _load_image(Path(self._image_paths[idx]))

        return {
            "image": self.tfm(img),
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
    pin_memory = bool(cfg["train"].get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(cfg["train"].get("persistent_workers", True))
    prefetch_factor = int(cfg["train"].get("prefetch_factor", 4))
    drop_last_train = bool(cfg["train"].get("drop_last_train", False))

    train_ds = ClassificationDataset(train_df, img_size=img_size, train=True, normalization=normalization)
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

    val_ds = ClassificationDataset(val_df, img_size=img_size, train=False, normalization=normalization)
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

    ds = ClassificationDataset(df, img_size=img_size, train=False, normalization=normalization)
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
