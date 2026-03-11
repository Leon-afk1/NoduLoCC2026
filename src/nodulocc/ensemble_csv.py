"""Fast ensembling from already-exported prediction CSV files.

This module is the deadline-oriented path: model inference is done separately
with `predict`, then this script aligns the two probability files, rebuilds the
reference split labels from metadata only, averages probabilities, and sweeps
the decision threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .data import LABEL_MAP, _resolve, _split_dataframe
from .metrics import classification_metrics, find_best_f1_threshold


def _parser() -> argparse.ArgumentParser:
    """Build CLI parser for the CSV-based ensemble utility."""
    parser = argparse.ArgumentParser(description="Ensemble two prediction CSVs on a shared labeled split")
    parser.add_argument("--pred-a", required=True, help="CSV exported for model A")
    parser.add_argument("--pred-b", required=True, help="CSV exported for model B")
    parser.add_argument("--config-ref", required=True, help="Reference config used to rebuild the split labels")
    parser.add_argument("--out-dir", required=True, help="Directory where summary and merged outputs are written")
    parser.add_argument("--labels-csv", default=None, help="Optional path to classification_labels.csv")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--fold", type=int, default=None, help="Optional fold index for k-fold runs")
    parser.add_argument("--weight-a", type=float, default=0.5, help="Weight of model A in the average")
    parser.add_argument("--threshold-a", type=float, default=0.5, help="Threshold used to score model A")
    parser.add_argument("--threshold-b", type=float, default=0.5, help="Threshold used to score model B")
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="Threshold used to score the ensemble before threshold sweep",
    )
    parser.add_argument("--override-ref", action="append", default=[], help="Override for the reference config, key=value")
    return parser


def _validate_classification_cfg(cfg: dict[str, Any]) -> None:
    """Ensure the reference config still targets the classification task."""
    task = str(cfg.get("task", "classification"))
    if task != "classification":
        raise ValueError(f"Reference config must use task=classification, got {task!r}")


def _load_reference_metadata(
    cfg: dict[str, Any],
    *,
    split: str,
    fold_index: int | None,
    labels_csv: str | None,
) -> pd.DataFrame:
    """Load labels and rebuild the requested split without touching image files."""
    if labels_csv is None:
        root = Path(cfg["data"]["dataset_root"]).resolve()
        labels_path = _resolve(root, str(cfg["data"]["classification"]["labels_csv"]))
    else:
        labels_path = Path(labels_csv).resolve()
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels CSV not found: {labels_path}")

    df = pd.read_csv(labels_path)
    required = {"file_name", "label", "LIDC_ID"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing classification columns: expected {required}")

    df = df.copy()
    df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Unknown labels found in classification CSV")
    df["label"] = df["label"].astype(np.int64)
    df["source"] = np.where(df["LIDC_ID"].notna(), "LIDC", "NIH")

    if split == "all":
        out = df
    else:
        train_df, val_df = _split_dataframe(df, cfg, fold_index=fold_index)
        out = train_df if split == "train" else val_df

    out = out[["file_name", "label", "source"]].copy().reset_index(drop=True)
    file_names = out["file_name"].astype(str)
    if file_names.duplicated().any():
        dupes = file_names[file_names.duplicated()].unique().tolist()
        preview = ", ".join(dupes[:5])
        raise ValueError(
            "Reference split contains duplicate file_name values, so CSV predictions cannot be aligned safely. "
            f"Example duplicates: {preview}"
        )
    return out


def _load_prediction_csv(path: str, prob_column_name: str) -> pd.DataFrame:
    """Load prediction CSV and standardize its probability column name."""
    csv_path = Path(path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"file_name", "prob_nodule"}
    if not required.issubset(df.columns):
        raise ValueError(f"Prediction CSV {csv_path} must contain columns {required}")

    df = df[["file_name", "prob_nodule"]].copy()
    df["file_name"] = df["file_name"].astype(str)
    df["prob_nodule"] = pd.to_numeric(df["prob_nodule"], errors="coerce")
    if df["prob_nodule"].isna().any():
        raise ValueError(f"Prediction CSV {csv_path} contains non-numeric prob_nodule values")
    if df["file_name"].duplicated().any():
        dupes = df.loc[df["file_name"].duplicated(), "file_name"].astype(str).unique().tolist()
        preview = ", ".join(dupes[:5])
        raise ValueError(f"Prediction CSV {csv_path} contains duplicate file_name values. Example: {preview}")
    return df.rename(columns={"prob_nodule": prob_column_name})


def _threshold_sweep_dataframe(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 181,
) -> pd.DataFrame:
    """Return a full threshold sweep table for later inspection."""
    thresholds = np.linspace(min_threshold, max_threshold, steps, dtype=np.float32)
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        metrics = classification_metrics(y_true=y_true, y_prob=y_prob, threshold=float(thr))
        rows.append({"threshold": float(thr), **{key: float(value) for key, value in metrics.items()}})
    return pd.DataFrame(rows)


def run_csv_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    """Run the lightweight CSV ensemble workflow and persist outputs."""
    if not (0.0 <= args.weight_a <= 1.0):
        raise ValueError("--weight-a must be in [0, 1]")
    for flag_name, value in {
        "--threshold-a": args.threshold_a,
        "--threshold-b": args.threshold_b,
        "--default-threshold": args.default_threshold,
    }.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{flag_name} must be in [0, 1]")

    cfg = load_config(args.config_ref, overrides=args.override_ref)
    _validate_classification_cfg(cfg)
    reference_df = _load_reference_metadata(
        cfg,
        split=args.split,
        fold_index=args.fold,
        labels_csv=args.labels_csv,
    )
    pred_a = _load_prediction_csv(args.pred_a, prob_column_name="prob_a")
    pred_b = _load_prediction_csv(args.pred_b, prob_column_name="prob_b")

    merged = reference_df.merge(pred_a, on="file_name", how="left").merge(pred_b, on="file_name", how="left")
    missing_a = merged["prob_a"].isna()
    missing_b = merged["prob_b"].isna()
    if bool(missing_a.any()) or bool(missing_b.any()):
        missing_rows = merged.loc[missing_a | missing_b, "file_name"].astype(str).tolist()
        preview = ", ".join(missing_rows[:5])
        raise ValueError(
            "Prediction CSVs do not fully cover the reference split. "
            f"Missing rows: {len(missing_rows)}. Example: {preview}"
        )

    weight_a = float(args.weight_a)
    weight_b = 1.0 - weight_a
    merged["prob_ensemble"] = weight_a * merged["prob_a"] + weight_b * merged["prob_b"]

    y_true = merged["label"].to_numpy(dtype=np.int32)
    y_prob_a = merged["prob_a"].to_numpy(dtype=np.float32)
    y_prob_b = merged["prob_b"].to_numpy(dtype=np.float32)
    y_prob_ensemble = merged["prob_ensemble"].to_numpy(dtype=np.float32)

    metrics_a = classification_metrics(y_true=y_true, y_prob=y_prob_a, threshold=float(args.threshold_a))
    metrics_b = classification_metrics(y_true=y_true, y_prob=y_prob_b, threshold=float(args.threshold_b))
    ensemble_default_metrics = classification_metrics(
        y_true=y_true,
        y_prob=y_prob_ensemble,
        threshold=float(args.default_threshold),
    )
    ensemble_best_threshold = find_best_f1_threshold(y_true=y_true, y_prob=y_prob_ensemble)
    sweep_df = _threshold_sweep_dataframe(y_true=y_true, y_prob=y_prob_ensemble)

    merged["pred_a"] = (merged["prob_a"] >= float(args.threshold_a)).astype(np.int32)
    merged["pred_b"] = (merged["prob_b"] >= float(args.threshold_b)).astype(np.int32)
    merged["pred_ensemble_default"] = (merged["prob_ensemble"] >= float(args.default_threshold)).astype(np.int32)
    merged["pred_ensemble_best"] = (
        merged["prob_ensemble"] >= float(ensemble_best_threshold["threshold"])
    ).astype(np.int32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_out = out_dir / "ensemble_probs.csv"
    sweep_out = out_dir / "threshold_sweep.csv"
    summary_out = out_dir / "summary.json"

    merged.sort_values("file_name").to_csv(merged_out, index=False)
    sweep_df.to_csv(sweep_out, index=False)

    summary = {
        "rows": int(len(merged)),
        "split": args.split,
        "fold": None if args.fold is None else int(args.fold),
        "config_ref": str(Path(args.config_ref).resolve()),
        "labels_csv": str(Path(args.labels_csv).resolve()) if args.labels_csv is not None else None,
        "pred_a": str(Path(args.pred_a).resolve()),
        "pred_b": str(Path(args.pred_b).resolve()),
        "weight_a": weight_a,
        "weight_b": weight_b,
        "threshold_a": float(args.threshold_a),
        "threshold_b": float(args.threshold_b),
        "model_a_metrics": {key: float(value) for key, value in metrics_a.items()},
        "model_b_metrics": {key: float(value) for key, value in metrics_b.items()},
        "ensemble_default_threshold": float(args.default_threshold),
        "ensemble_default_metrics": {key: float(value) for key, value in ensemble_default_metrics.items()},
        "ensemble_best_threshold": {key: float(value) for key, value in ensemble_best_threshold.items()},
        "outputs": {
            "ensemble_probs": str(merged_out.resolve()),
            "threshold_sweep": str(sweep_out.resolve()),
            "summary": str(summary_out.resolve()),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Run standalone module entrypoint."""
    summary = run_csv_ensemble(_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
