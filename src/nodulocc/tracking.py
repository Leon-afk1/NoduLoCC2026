"""Experiment tracking wrapper with optional advanced W&B diagnostics.

Current provider support: Weights & Biases (`wandb`).
If unavailable, tracking is disabled gracefully.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


class Tracker:
    """Light abstraction to keep training code independent from tracker APIs."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        run_name_suffix: str | None = None,
        job_type: str | None = None,
        run_group: str | None = None,
    ) -> None:
        """Initialize tracker from config and start run when enabled."""
        self._cfg = cfg
        self.enabled = bool(cfg.get("tracking", {}).get("enabled", False))
        self.provider = cfg.get("tracking", {}).get("provider", "wandb")
        self._run = None
        self._wandb = None
        self._tracking_cfg = cfg.get("tracking", {})
        self.group = run_group or self._tracking_cfg.get("group")
        self.run_id: str | None = None

        if not self.enabled:
            return

        if self.provider != "wandb":
            print(f"[tracking] provider '{self.provider}' not supported, disabling.")
            self.enabled = False
            return

        try:
            import wandb  # type: ignore[import-not-found]

            base_name = self._tracking_cfg.get("run_name")
            run_name: str | None
            if base_name:
                run_name = f"{base_name}-{run_name_suffix}" if run_name_suffix else str(base_name)
            elif bool(self._tracking_cfg.get("auto_run_name", True)):
                run_name = self._build_auto_run_name(run_name_suffix=run_name_suffix, job_type=job_type)
            else:
                # Let W&B generate a random name.
                run_name = None

            tags = self._tracking_cfg.get("tags")
            if tags is not None and not isinstance(tags, list):
                tags = [str(tags)]

            self._wandb = wandb
            self._run = wandb.init(
                project=self._tracking_cfg.get("project", "nodulocc"),
                entity=self._tracking_cfg.get("entity"),
                name=run_name,
                job_type=job_type,
                group=self.group,
                tags=tags,
                config=cfg,
            )
            self.run_id = str(getattr(self._run, "id", ""))
        except Exception as exc:
            print(f"[tracking] wandb init failed: {exc}")
            self.enabled = False

    @staticmethod
    def _slug(value: str, max_len: int = 32) -> str:
        """Normalize strings for safe, compact run names."""
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        if not normalized:
            normalized = "na"
        return normalized[:max_len]

    def _build_auto_run_name(self, *, run_name_suffix: str | None, job_type: str | None) -> str:
        """Build informative run name from config when `tracking.run_name` is unset."""
        task = self._slug(str(self._cfg.get("task", "classification")))

        mode = "full" if bool(self._cfg.get("train", {}).get("full_data", False)) else str(
            self._cfg.get("validation", {}).get("mode", "holdout")
        )
        mode = self._slug(mode, max_len=16)
        if mode == "kfold":
            k = self._cfg.get("validation", {}).get("k")
            if k is not None:
                mode = self._slug(f"kfold{k}", max_len=16)
            fold_index = self._cfg.get("validation", {}).get("fold_index")
            if fold_index is not None and job_type in {"eval", "predict"}:
                mode = self._slug(f"{mode}-f{int(fold_index)}", max_len=16)

        backbone_raw = str(self._cfg.get("model", {}).get("backbone", "model"))
        backbone = self._slug(backbone_raw.split("/")[-1].split(".")[-1], max_len=24)
        seed = int(self._cfg.get("seed", 0))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        jt = self._slug(job_type or "run", max_len=12)

        parts: list[str] = []
        prefix = self._tracking_cfg.get("run_name_prefix")
        if prefix:
            parts.append(self._slug(str(prefix), max_len=24))
        parts.extend([task, jt, mode, backbone, f"s{seed}", timestamp])
        if run_name_suffix and run_name_suffix not in {"train", "eval", "predict"}:
            parts.append(self._slug(run_name_suffix, max_len=16))
        return "-".join(parts)

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        """Log values for the current run."""
        if self.enabled and self._run is not None:
            self._run.log(values, step=step)

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        """Log a tabular payload as a W&B Table."""
        if not self.enabled or self._wandb is None:
            return
        table = self._wandb.Table(columns=columns, data=rows)
        self.log({key: table})

    def log_file_artifact(
        self,
        path: str | Path,
        *,
        artifact_name: str,
        artifact_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upload a local file as a W&B artifact."""
        if not self.enabled or self._run is None or self._wandb is None:
            return
        file_path = Path(path)
        if not file_path.is_file():
            return
        try:
            artifact = self._wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata or {},
            )
            artifact.add_file(str(file_path))
            self._run.log_artifact(artifact)
        except Exception as exc:
            print(f"[tracking] artifact logging failed for {file_path}: {exc}")

    def log_classification_diagnostics(
        self,
        *,
        prefix: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        file_names: list[str] | None = None,
    ) -> None:
        """Log confusion matrix, curves, histograms, and optional prediction table."""
        if not self.enabled or self._wandb is None:
            return

        y_true_i = y_true.astype(np.int32)
        y_prob_f = y_prob.astype(np.float32)
        y_pred_i = (y_prob_f >= float(threshold)).astype(np.int32)

        log_confusion = bool(self._tracking_cfg.get("log_confusion_matrix", True))
        log_curves = bool(self._tracking_cfg.get("log_curves", True))
        log_hist = bool(self._tracking_cfg.get("log_prob_histograms", True))
        log_table = bool(self._tracking_cfg.get("log_prediction_table", True))
        max_rows = int(self._tracking_cfg.get("prediction_table_max_rows", 2000))

        payload: dict[str, Any] = {}

        if log_confusion:
            try:
                payload[f"{prefix}/confusion_matrix"] = self._wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_i.tolist(),
                    preds=y_pred_i.tolist(),
                    class_names=["No Finding", "Nodule"],
                )
            except Exception as exc:
                print(f"[tracking] confusion matrix logging failed: {exc}")

        if log_curves:
            probs_2d = np.column_stack([1.0 - y_prob_f, y_prob_f])
            try:
                payload[f"{prefix}/roc_curve"] = self._wandb.plot.roc_curve(
                    y_true_i,
                    probs_2d,
                    labels=["No Finding", "Nodule"],
                )
            except Exception as exc:
                print(f"[tracking] ROC curve logging failed: {exc}")
            try:
                payload[f"{prefix}/pr_curve"] = self._wandb.plot.pr_curve(
                    y_true_i,
                    probs_2d,
                    labels=["No Finding", "Nodule"],
                )
            except Exception as exc:
                print(f"[tracking] PR curve logging failed: {exc}")

        if log_hist:
            neg_scores = y_prob_f[y_true_i == 0]
            pos_scores = y_prob_f[y_true_i == 1]
            if neg_scores.size > 0:
                payload[f"{prefix}/prob_hist_negative"] = self._wandb.Histogram(neg_scores)
            if pos_scores.size > 0:
                payload[f"{prefix}/prob_hist_positive"] = self._wandb.Histogram(pos_scores)

        if payload:
            self.log(payload)

        if log_table:
            if file_names is None:
                file_names = [f"sample_{i}" for i in range(len(y_true_i))]
            indices = np.arange(len(y_true_i))
            if len(indices) > max_rows:
                rng = np.random.default_rng(42)
                indices = np.sort(rng.choice(indices, size=max_rows, replace=False))

            rows: list[list[Any]] = []
            for idx in indices:
                rows.append(
                    [
                        str(file_names[idx]),
                        int(y_true_i[idx]),
                        float(y_prob_f[idx]),
                        int(y_pred_i[idx]),
                    ]
                )
            self.log_table(
                key=f"{prefix}/predictions_table",
                columns=["file_name", "y_true", "y_prob", "y_pred"],
                rows=rows,
            )

    def _balanced_sample_indices(
        self,
        labels: np.ndarray,
        sample_count: int,
        seed: int,
    ) -> np.ndarray:
        """Pick indices with basic class balancing for binary labels."""
        if sample_count <= 0 or labels.size == 0:
            return np.array([], dtype=np.int64)

        rng = np.random.default_rng(seed)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]

        half = sample_count // 2
        take_pos = min(len(pos_idx), half)
        take_neg = min(len(neg_idx), half)

        chosen = []
        if take_pos > 0:
            chosen.extend(rng.choice(pos_idx, size=take_pos, replace=False).tolist())
        if take_neg > 0:
            chosen.extend(rng.choice(neg_idx, size=take_neg, replace=False).tolist())

        remaining = sample_count - len(chosen)
        if remaining > 0:
            all_idx = np.arange(labels.size)
            mask = np.ones(labels.size, dtype=bool)
            if chosen:
                mask[np.array(chosen, dtype=np.int64)] = False
            candidates = all_idx[mask]
            if candidates.size > 0:
                extra = rng.choice(candidates, size=min(remaining, candidates.size), replace=False)
                chosen.extend(extra.tolist())

        return np.array(sorted(set(chosen)), dtype=np.int64)

    def log_data_exploration(
        self,
        *,
        prefix: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        seed: int,
    ) -> None:
        """Log dataset exploration stats and sample images to W&B."""
        if not self.enabled or self._wandb is None:
            return
        if not bool(self._tracking_cfg.get("log_data_exploration", True)):
            return

        def _count(df: pd.DataFrame | None, label: int) -> int:
            if df is None or len(df) == 0:
                return 0
            return int((df["label"] == label).sum())

        n_primary = int(len(train_df))
        n_val = int(len(val_df)) if val_df is not None else 0
        n_total = n_primary + n_val

        train_pos = _count(train_df, 1)
        train_neg = _count(train_df, 0)
        val_pos = _count(val_df, 1)
        val_neg = _count(val_df, 0)
        total_pos = train_pos + val_pos
        total_neg = train_neg + val_neg

        scalar_payload: dict[str, Any] = {
            f"{prefix}/data/n_total": float(n_total),
            f"{prefix}/data/nodule_total": float(total_pos),
            f"{prefix}/data/no_finding_total": float(total_neg),
            f"{prefix}/data/imbalance_no_finding_to_nodule": float(total_neg / max(1, total_pos)),
        }
        if val_df is None:
            # When no split dataframe is provided, treat input as global dataset.
            scalar_payload[f"{prefix}/data/global_pos_rate"] = float(total_pos / max(1, n_total))
            scalar_payload[f"{prefix}/data/n_global"] = float(n_total)
        else:
            scalar_payload[f"{prefix}/data/n_train"] = float(n_primary)
            scalar_payload[f"{prefix}/data/n_val"] = float(n_val)
            scalar_payload[f"{prefix}/data/train_pos_rate"] = float(train_pos / max(1, n_primary))
            scalar_payload[f"{prefix}/data/val_pos_rate"] = float(val_pos / max(1, n_val))
        self.log(scalar_payload)

        split_rows: list[tuple[str, pd.DataFrame | None, int, int]]
        if val_df is None:
            split_rows = [("global", train_df, total_pos, total_neg)]
        else:
            split_rows = [("train", train_df, train_pos, train_neg), ("val", val_df, val_pos, val_neg)]

        for split_name, split_df, pos, neg in split_rows:
            if split_df is None or len(split_df) == 0:
                continue
            try:
                count_table = self._wandb.Table(
                    columns=["class", "count"],
                    data=[["No Finding", float(neg)], ["Nodule", float(pos)]],
                )
                self.log(
                    {
                        f"{prefix}/data/{split_name}_class_distribution": self._wandb.plot.bar(
                            count_table,
                            "class",
                            "count",
                            title=f"{split_name.capitalize()} class distribution",
                        )
                    }
                )
            except Exception as exc:
                print(f"[tracking] class distribution logging failed for {split_name}: {exc}")

        sample_count = int(self._tracking_cfg.get("data_samples_per_run", 8))
        if sample_count <= 0:
            return

        rows: list[list[Any]] = []
        per_split = max(1, sample_count // 2) if val_df is not None and len(val_df) > 0 else sample_count
        if val_df is None:
            sample_splits = [("global", train_df)]
        else:
            sample_splits = [("train", train_df), ("val", val_df)]

        for split_offset, (split_name, split_df) in enumerate(sample_splits):
            if split_df is None or len(split_df) == 0:
                continue
            labels = split_df["label"].to_numpy(dtype=np.int64)
            chosen_idx = self._balanced_sample_indices(labels, per_split, seed + split_offset)
            for idx in chosen_idx:
                row = split_df.iloc[int(idx)]
                image_path = str(row["image_path"])
                label_int = int(row["label"])
                label_name = "Nodule" if label_int == 1 else "No Finding"
                image_obj: Any = image_path
                try:
                    image_obj = self._wandb.Image(
                        image_path,
                        caption=f"{split_name}:{row['file_name']} | {label_name}",
                    )
                except Exception:
                    pass
                rows.append(
                    [
                        split_name,
                        str(row["file_name"]),
                        label_int,
                        label_name,
                        image_obj,
                    ]
                )

        if rows:
            self.log_table(
                key=f"{prefix}/data/samples",
                columns=["split", "file_name", "label", "label_name", "image"],
                rows=rows,
            )

    def finish(self) -> None:
        """Close tracker run cleanly."""
        if self.enabled and self._run is not None:
            self._run.finish()
