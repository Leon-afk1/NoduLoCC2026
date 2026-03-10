"""Utilities to ensemble two classification checkpoints on a shared split.

This module exists for the practical deadline case where two trained models
must be compared and averaged on the exact same validation subset, even if
their configs use different preprocessing pipelines or source mixes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import (
    _augmentation_settings,
    _build_dataset,
    _clahe_settings,
    _normalization_stats,
    _preprocessing_profile,
    load_classification_dataframe,
    prepare_frames,
)
from .engine import (
    _autocast_context,
    _configure_runtime,
    _device_from_cfg,
    _prepare_batch_input,
    _resolve_precision,
    _to_numpy_float32,
    _use_channels_last,
)
from .metrics import classification_metrics, find_best_f1_threshold
from .models import build_model


def _parser() -> argparse.ArgumentParser:
    """Build argument parser for the standalone two-model ensemble script."""
    parser = argparse.ArgumentParser(description="Ensemble two classification checkpoints on a shared split")
    parser.add_argument("--config-a", required=True, help="Config for model A")
    parser.add_argument("--ckpt-a", required=True, help="Checkpoint for model A")
    parser.add_argument("--config-b", required=True, help="Config for model B")
    parser.add_argument("--ckpt-b", required=True, help="Checkpoint for model B")
    parser.add_argument("--out-dir", required=True, help="Directory where predictions and summary are written")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "all"],
        help="Reference split to evaluate on",
    )
    parser.add_argument(
        "--reference",
        default="a",
        choices=["a", "b"],
        help="Which config defines the target split membership",
    )
    parser.add_argument(
        "--weight-a",
        type=float,
        default=0.5,
        help="Weight for model A in the probability average; model B gets 1-weight_a",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="Threshold used for the default ensemble metrics before sweeping",
    )
    parser.add_argument("--fold-a", type=int, default=None, help="Optional fold index for config A")
    parser.add_argument("--fold-b", type=int, default=None, help="Optional fold index for config B")
    parser.add_argument("--override-a", action="append", default=[], help="Override for config A, key=value")
    parser.add_argument("--override-b", action="append", default=[], help="Override for config B, key=value")
    return parser


def _apply_optional_fold(cfg: dict[str, Any], fold: int | None) -> dict[str, Any]:
    """Inject fold selection into a config when requested."""
    if fold is not None:
        cfg.setdefault("validation", {})["fold_index"] = int(fold)
    return cfg


def _validate_classification_cfg(name: str, cfg: dict[str, Any]) -> None:
    """Fail fast if one of the provided configs is not classification-only."""
    task = str(cfg.get("task", "classification"))
    if task != "classification":
        raise ValueError(f"Config {name} must use task=classification, got {task!r}")


def _target_dataframe(cfg: dict[str, Any], split: str, fold_index: int | None) -> pd.DataFrame:
    """Return the dataframe that defines the shared evaluation subset."""
    if split == "all":
        df, _ = prepare_frames(cfg, full_train=True)
        return df.reset_index(drop=True)

    train_df, val_df = prepare_frames(cfg, fold_index=fold_index, full_train=False)
    if split == "train":
        return train_df.reset_index(drop=True)
    if val_df is None:
        raise ValueError("No validation split is available for the requested reference config.")
    return val_df.reset_index(drop=True)


def _target_keys(df: pd.DataFrame) -> list[str]:
    """Return ordered unique file keys used to align predictions across models."""
    file_names = df["file_name"].astype(str)
    if file_names.duplicated().any():
        dupes = file_names[file_names.duplicated()].unique().tolist()
        preview = ", ".join(dupes[:5])
        raise ValueError(
            "Reference split contains duplicate file_name values, so predictions cannot be aligned safely. "
            f"Example duplicates: {preview}"
        )
    return file_names.tolist()


def _filtered_dataframe(cfg: dict[str, Any], target_file_names: list[str]) -> pd.DataFrame:
    """Filter one config dataframe down to the shared target file list."""
    df = load_classification_dataframe(cfg).copy()
    target_set = set(target_file_names)
    filtered = df[df["file_name"].astype(str).isin(target_set)].copy()
    if len(filtered) == 0:
        raise ValueError("No shared files found between target split and model dataframe.")
    if filtered["file_name"].astype(str).duplicated().any():
        raise ValueError("Filtered dataframe contains duplicate file_name values; cannot align predictions safely.")
    return filtered.reset_index(drop=True)


def _build_shared_loader(cfg: dict[str, Any], df: pd.DataFrame) -> DataLoader:
    """Build inference loader with the config's own preprocessing settings."""
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

    dataset = _build_dataset(
        cfg,
        df,
        img_size=img_size,
        train=False,
        normalization=normalization,
        clahe=clahe,
        augmentation_settings=augmentation_settings,
        preprocessing_profile=preprocessing_profile,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def _load_model_for_inference(cfg: dict[str, Any], ckpt_path: str, device: torch.device, channels_last: bool) -> torch.nn.Module:
    """Build model from config and restore checkpoint state for inference."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["pretrained"] = False
    model = build_model(task="classification", model_cfg=model_cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _predict_probabilities(
    cfg: dict[str, Any],
    ckpt_path: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run one checkpoint on a filtered dataframe and return `file_name, prob_nodule`."""
    device = _device_from_cfg(cfg)
    _configure_runtime(cfg, device)
    precision = _resolve_precision(cfg, device)
    channels_last = _use_channels_last(cfg, device)
    non_blocking = device.type == "cuda"

    loader = _build_shared_loader(cfg, df)
    model = _load_model_for_inference(cfg, ckpt_path=ckpt_path, device=device, channels_last=channels_last)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x = _prepare_batch_input(
                batch=batch,
                device=device,
                non_blocking=non_blocking,
                channels_last=channels_last,
            )
            with _autocast_context(device, precision):
                prob = _to_numpy_float32(torch.sigmoid(model(x)))
            for idx, file_name in enumerate(batch["file_name"]):
                rows.append(
                    {
                        "file_name": str(file_name),
                        "prob_nodule": float(prob[idx]),
                    }
                )
    pred_df = pd.DataFrame(rows)
    if pred_df["file_name"].duplicated().any():
        raise ValueError("Predictions contain duplicate file_name values; cannot align safely.")
    return pred_df


def _aligned_merge(
    target_df: pd.DataFrame,
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
) -> pd.DataFrame:
    """Align reference labels and both probability vectors on the same rows."""
    merged = target_df[["file_name", "label"]].copy()
    merged = merged.merge(pred_a.rename(columns={"prob_nodule": "prob_a"}), on="file_name", how="inner")
    merged = merged.merge(pred_b.rename(columns={"prob_nodule": "prob_b"}), on="file_name", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping predictions were found after alignment.")
    if len(merged) != len(target_df):
        missing = sorted(set(target_df["file_name"].astype(str)) - set(merged["file_name"].astype(str)))
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Aligned predictions do not cover the full target split. "
            f"Missing {len(missing)} files. Example: {preview}"
        )
    return merged


def _threshold_sweep_dataframe(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 181,
) -> pd.DataFrame:
    """Return a table of metrics across a threshold grid."""
    thresholds = np.linspace(min_threshold, max_threshold, steps, dtype=np.float32)
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        metrics = classification_metrics(y_true=y_true, y_prob=y_prob, threshold=float(thr))
        rows.append(
            {
                "threshold": float(thr),
                **{key: float(value) for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def run_two_model_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full two-model ensemble workflow and persist outputs."""
    if args.weight_a < 0.0 or args.weight_a > 1.0:
        raise ValueError("--weight-a must be in [0, 1]")
    if args.default_threshold < 0.0 or args.default_threshold > 1.0:
        raise ValueError("--default-threshold must be in [0, 1]")

    cfg_a = _apply_optional_fold(load_config(args.config_a, overrides=args.override_a), args.fold_a)
    cfg_b = _apply_optional_fold(load_config(args.config_b, overrides=args.override_b), args.fold_b)
    _validate_classification_cfg("A", cfg_a)
    _validate_classification_cfg("B", cfg_b)

    ref_cfg = cfg_a if args.reference == "a" else cfg_b
    ref_fold = args.fold_a if args.reference == "a" else args.fold_b
    target_df = _target_dataframe(ref_cfg, split=args.split, fold_index=ref_fold)
    target_file_names = _target_keys(target_df)

    shared_a_df = _filtered_dataframe(cfg_a, target_file_names)
    shared_b_df = _filtered_dataframe(cfg_b, target_file_names)

    pred_a = _predict_probabilities(cfg_a, ckpt_path=args.ckpt_a, df=shared_a_df)
    pred_b = _predict_probabilities(cfg_b, ckpt_path=args.ckpt_b, df=shared_b_df)
    merged = _aligned_merge(target_df=target_df, pred_a=pred_a, pred_b=pred_b)

    weight_a = float(args.weight_a)
    weight_b = 1.0 - weight_a
    merged["prob_ensemble"] = weight_a * merged["prob_a"] + weight_b * merged["prob_b"]

    y_true = merged["label"].to_numpy(dtype=np.int32)
    y_prob_a = merged["prob_a"].to_numpy(dtype=np.float32)
    y_prob_b = merged["prob_b"].to_numpy(dtype=np.float32)
    y_prob_ensemble = merged["prob_ensemble"].to_numpy(dtype=np.float32)

    threshold_a = float(cfg_a.get("eval", {}).get("threshold", 0.5))
    threshold_b = float(cfg_b.get("eval", {}).get("threshold", 0.5))
    model_a_metrics = classification_metrics(y_true=y_true, y_prob=y_prob_a, threshold=threshold_a)
    model_b_metrics = classification_metrics(y_true=y_true, y_prob=y_prob_b, threshold=threshold_b)
    ensemble_default_metrics = classification_metrics(
        y_true=y_true,
        y_prob=y_prob_ensemble,
        threshold=float(args.default_threshold),
    )
    ensemble_best_threshold = find_best_f1_threshold(y_true=y_true, y_prob=y_prob_ensemble)
    threshold_sweep = _threshold_sweep_dataframe(y_true=y_true, y_prob=y_prob_ensemble)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_a_out = out_dir / "model_a_probs.csv"
    pred_b_out = out_dir / "model_b_probs.csv"
    ensemble_out = out_dir / "ensemble_probs.csv"
    sweep_out = out_dir / "threshold_sweep.csv"
    summary_out = out_dir / "summary.json"

    pred_a.sort_values("file_name").to_csv(pred_a_out, index=False)
    pred_b.sort_values("file_name").to_csv(pred_b_out, index=False)
    merged.sort_values("file_name").to_csv(ensemble_out, index=False)
    threshold_sweep.to_csv(sweep_out, index=False)

    summary = {
        "reference": args.reference,
        "split": args.split,
        "rows": int(len(merged)),
        "config_a": str(Path(args.config_a).resolve()),
        "ckpt_a": str(Path(args.ckpt_a).resolve()),
        "config_b": str(Path(args.config_b).resolve()),
        "ckpt_b": str(Path(args.ckpt_b).resolve()),
        "weight_a": weight_a,
        "weight_b": weight_b,
        "model_a_threshold": threshold_a,
        "model_b_threshold": threshold_b,
        "model_a_metrics": {key: float(value) for key, value in model_a_metrics.items()},
        "model_b_metrics": {key: float(value) for key, value in model_b_metrics.items()},
        "ensemble_default_threshold": float(args.default_threshold),
        "ensemble_default_metrics": {key: float(value) for key, value in ensemble_default_metrics.items()},
        "ensemble_best_threshold": {key: float(value) for key, value in ensemble_best_threshold.items()},
        "outputs": {
            "model_a_probs": str(pred_a_out.resolve()),
            "model_b_probs": str(pred_b_out.resolve()),
            "ensemble_probs": str(ensemble_out.resolve()),
            "threshold_sweep": str(sweep_out.resolve()),
            "summary": str(summary_out.resolve()),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Run CLI entrypoint for the standalone ensemble module."""
    summary = run_two_model_ensemble(_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
