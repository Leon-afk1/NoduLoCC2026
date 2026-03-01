"""Training, evaluation, and prediction engine for classification."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .data import build_dataloaders, build_prediction_loader, load_classification_dataframe
from .metrics import classification_metrics, find_best_f1_threshold
from .models import build_model
from .tracking import Tracker


def _sanitize_artifact_name(value: str) -> str:
    """Convert arbitrary strings to artifact-safe names."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return safe.strip("-") or "artifact"


def _device_from_cfg(cfg: dict[str, Any]) -> torch.device:
    """Resolve execution device from config (`auto` chooses CUDA when available)."""
    value = str(cfg["train"].get("device", "auto"))
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _configure_runtime(cfg: dict[str, Any], device: torch.device) -> None:
    """Apply backend settings that improve GPU throughput."""
    train_cfg = cfg.get("train", {})
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    allow_tf32 = bool(train_cfg.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    matmul_precision = str(train_cfg.get("matmul_precision", "high")).lower()
    if matmul_precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)


def _resolve_precision(cfg: dict[str, Any], device: torch.device) -> str:
    """Resolve train/eval precision mode for autocast."""
    precision = str(cfg.get("train", {}).get("precision", "bf16")).lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("train.precision must be one of: fp32, bf16, fp16")
    if device.type != "cuda" and precision != "fp32":
        # Non-CUDA backends use standard fp32 path in this project.
        return "fp32"
    return precision


def _autocast_context(device: torch.device, precision: str) -> Any:
    """Return autocast context manager for configured precision."""
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _use_channels_last(cfg: dict[str, Any], device: torch.device) -> bool:
    """Whether to use channels-last memory format for convolutions."""
    return bool(cfg.get("train", {}).get("channels_last", True)) and device.type == "cuda"


def _compile_model_if_enabled(cfg: dict[str, Any], model: nn.Module) -> nn.Module:
    """Compile model with torch.compile when enabled in config."""
    if not bool(cfg.get("train", {}).get("compile", False)):
        return model
    if not hasattr(torch, "compile"):
        return model
    mode = str(cfg.get("train", {}).get("compile_mode", "default"))
    try:
        return torch.compile(model, mode=mode)
    except Exception as exc:
        print(f"[train] torch.compile failed, fallback to eager mode: {exc}")
        return model


def _build_grad_scaler(device: torch.device, precision: str) -> Any:
    """Create a GradScaler with compatibility for old/new PyTorch APIs."""
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _show_progress_bars() -> bool:
    """Enable tqdm bars only for interactive terminals."""
    return bool(sys.stderr.isatty())


def _classification_pos_weight(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Compute `pos_weight` for BCEWithLogits from class frequencies."""
    n_pos = float((train_df["label"] == 1).sum())
    n_neg = float((train_df["label"] == 0).sum())
    return torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32, device=device)


class BinaryFocalLoss(nn.Module):
    """Binary focal loss with optional class reweighting via `pos_weight`."""

    def __init__(self, gamma: float, alpha: float | None = None, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss on logits/targets of shape `[B]`."""
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if self.pos_weight is not None:
            pos_w = float(self.pos_weight.detach().cpu().item())
            sample_w = torch.where(targets > 0.5, pos_w, 1.0)
            bce = bce * sample_w

        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        focal_term = torch.pow(1.0 - pt, self.gamma)

        if self.alpha is not None:
            alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
            focal_term = focal_term * alpha_t

        return (focal_term * bce).mean()


def _build_loss_fn(cfg: dict[str, Any], train_df: pd.DataFrame, device: torch.device) -> nn.Module:
    """Build training loss from config (`bce` or `focal`)."""
    loss_cfg = cfg.get("train", {}).get("loss", {})
    loss_name = str(loss_cfg.get("name", "bce")).lower()
    use_pos_weight = bool(loss_cfg.get("use_pos_weight", True))
    pos_weight = _classification_pos_weight(train_df, device) if use_pos_weight else None

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == "focal":
        return BinaryFocalLoss(
            gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            alpha=float(loss_cfg["focal_alpha"]) if loss_cfg.get("focal_alpha") is not None else None,
            pos_weight=pos_weight,
        )

    raise ValueError("train.loss.name must be one of: bce, focal")


def _build_epoch_lrs(cfg: dict[str, Any], epochs: int, base_lr: float) -> list[float]:
    """Build per-epoch learning rates from scheduler config."""
    sched_cfg = cfg.get("train", {}).get("scheduler", {})
    name = str(sched_cfg.get("name", "none")).lower()
    if name in {"none", "off", "disabled"}:
        return [float(base_lr)] * epochs
    if name != "cosine":
        raise ValueError("train.scheduler.name must be one of: none, cosine")

    warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
    warmup_epochs = max(0, min(warmup_epochs, epochs))
    min_lr = float(sched_cfg.get("min_lr", base_lr * 0.05))
    if min_lr <= 0.0:
        raise ValueError("train.scheduler.min_lr must be > 0")

    lrs: list[float] = []
    for epoch_idx in range(epochs):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            lr = base_lr * float(epoch_idx + 1) / float(warmup_epochs)
        else:
            remaining = max(1, epochs - warmup_epochs)
            progress = float(epoch_idx - warmup_epochs + 1) / float(remaining)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            lr = min_lr + (base_lr - min_lr) * cosine
        lrs.append(float(lr))
    return lrs


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate on all optimizer parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _threshold_output_path(cfg: dict[str, Any], k: int) -> Path:
    """Return path where OOF threshold search results are persisted."""
    out_dir = Path(cfg.get("eval", {}).get("threshold_output_dir", "artifacts/thresholds"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"classification_kfold_k{k}_seed{int(cfg.get('seed', 42))}.json"


def _resolve_eval_threshold(cfg: dict[str, Any], mode: str) -> float:
    """Resolve threshold for eval/predict, optionally from OOF k-fold search file."""
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
    if mode != "kfold":
        return threshold
    if not bool(cfg.get("eval", {}).get("use_oof_threshold", False)):
        return threshold

    k = int(cfg.get("validation", {}).get("k", 5))
    threshold_path = _threshold_output_path(cfg, k)
    if not threshold_path.is_file():
        print(f"[eval] OOF threshold file not found at {threshold_path}, fallback to eval.threshold={threshold:.4f}")
        return threshold

    payload = json.loads(threshold_path.read_text())
    return float(payload.get("oof_best_threshold", threshold))


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
    precision: str,
    channels_last: bool,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str]]:
    """Run classification evaluation on one dataloader and return predictions."""
    model.eval()
    ys, ps = [], []
    file_names: list[str] = []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=non_blocking)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y = batch["label"].to(device, non_blocking=non_blocking)
            with _autocast_context(device, precision):
                logit = model(x)
                prob = torch.sigmoid(logit)
            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())
            file_names.extend([str(x) for x in batch["file_name"]])

    y_true = np.concatenate(ys).astype(np.float32)
    y_prob = np.concatenate(ps).astype(np.float32)
    metrics = classification_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    return metrics, y_true, y_prob, file_names


def _run_single_training(
    cfg: dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None,
    train_df: pd.DataFrame,
    device: torch.device,
    ckpt_path: Path,
    tracker: Tracker,
    log_prefix: str,
    tracking_group: str,
) -> dict[str, Any]:
    """Train one model run (holdout fold, k-fold fold, or full-data run)."""
    base_lr = float(cfg["train"].get("lr", 2e-4))
    precision = _resolve_precision(cfg, device)
    channels_last = _use_channels_last(cfg, device)

    model = build_model(task="classification", model_cfg=cfg["model"]).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = _compile_model_if_enabled(cfg, model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["train"].get("epochs", 1))
    loss_fn = _build_loss_fn(cfg, train_df, device)
    epoch_lrs = _build_epoch_lrs(cfg, epochs=epochs, base_lr=base_lr)
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))

    if val_loader is None:
        monitor = "train_loss"
        best_value = np.inf
    else:
        monitor = "f1"
        best_value = -np.inf
    best_outputs: tuple[np.ndarray, np.ndarray, list[str]] | None = None
    non_blocking = device.type == "cuda"
    scaler = _build_grad_scaler(device, precision)

    for epoch in range(1, epochs + 1):
        lr = float(epoch_lrs[epoch - 1])
        _set_optimizer_lr(optimizer, lr)

        model.train()
        running = 0.0
        seen = 0

        for batch in tqdm(
            train_loader,
            desc=f"train-{log_prefix}-e{epoch}",
            leave=False,
            disable=not _show_progress_bars(),
        ):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, precision):
                logit = model(x)
                loss = loss_fn(logit, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = running / max(1, seen)

        if val_loader is None:
            metrics = {"train_loss": float(train_loss), "lr": lr}
            current = float(train_loss)
            improved = current < best_value
        else:
            metrics, y_true, y_prob, file_names = _evaluate(
                model,
                val_loader,
                device,
                threshold=threshold,
                precision=precision,
                channels_last=channels_last,
            )
            metrics["train_loss"] = float(train_loss)
            metrics["lr"] = lr
            current = float(metrics["f1"])
            improved = current > best_value

        tracker.log({f"{log_prefix}/{k}": v for k, v in metrics.items()}, step=epoch)

        if not _show_progress_bars():
            if val_loader is None:
                print(
                    f"[{log_prefix}] epoch {epoch}/{epochs} train_loss={float(train_loss):.4f}"
                )
            else:
                print(
                    f"[{log_prefix}] epoch {epoch}/{epochs} "
                    f"train_loss={float(train_loss):.4f} "
                    f"f1={float(metrics.get('f1', 0.0)):.4f} "
                    f"auc={float(metrics.get('roc_auc', 0.0)):.4f}"
                )

        if improved:
            best_value = current
            if val_loader is not None:
                best_outputs = (y_true, y_prob, file_names)
            torch.save(
                {
                    "task": "classification",
                    "state_dict": model.state_dict(),
                    "cfg": copy.deepcopy(cfg),
                    "best_metric": best_value,
                    "monitor": monitor,
                    "tracking_group": tracking_group,
                    "tracking_run_id": tracker.run_id,
                },
                ckpt_path,
            )

    if best_outputs is not None:
        y_true, y_prob, file_names = best_outputs
        tracker.log_classification_diagnostics(
            prefix=f"{log_prefix}/best",
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            file_names=file_names,
        )
    if bool(cfg.get("tracking", {}).get("log_artifacts", True)):
        artifact_name = _sanitize_artifact_name(f"{tracking_group}-{log_prefix}-{ckpt_path.stem}-model")
        tracker.log_file_artifact(
            ckpt_path,
            artifact_name=artifact_name,
            artifact_type="model",
            metadata={
                "monitor": monitor,
                "best_metric": float(best_value),
                "tracking_group": tracking_group,
                "log_prefix": log_prefix,
            },
        )

    return {
        "checkpoint": str(ckpt_path),
        "best_metric": float(best_value),
        "monitor": monitor,
        "_best_outputs": best_outputs,
    }


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train model with holdout, k-fold, or full-data mode.

    Config knobs:
    - validation.mode: holdout | kfold
    - validation.k: number of folds (kfold mode)
    - train.full_data: if true, train once on 100% data without validation
    """
    task = cfg["task"]
    if task != "classification":
        raise ValueError("Only classification task is supported.")

    _set_seed(int(cfg.get("seed", 42)))
    device = _device_from_cfg(cfg)
    _configure_runtime(cfg, device)

    output_dir = Path(cfg["train"].get("output_dir", "artifacts/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = str(cfg.get("validation", {}).get("mode", "holdout")).lower()
    full_data = bool(cfg["train"].get("full_data", False))

    cfg_tracking_group = cfg.get("tracking", {}).get("group")
    tracking_group = str(cfg_tracking_group) if cfg_tracking_group else f"train-{int(time.time())}"
    tracker = Tracker(cfg, job_type="train", run_group=tracking_group)
    start = time.time()
    # Log dataset exploration once per training run (not once per fold).
    tracker.log_data_exploration(
        prefix="global",
        train_df=load_classification_dataframe(cfg),
        val_df=None,
        seed=int(cfg.get("seed", 42)),
    )

    if full_data:
        train_loader, _, train_df, _ = build_dataloaders(cfg, full_train=True)
        ckpt_path = output_dir / "classification_full_best.pt"
        run = _run_single_training(
            cfg,
            train_loader=train_loader,
            val_loader=None,
            train_df=train_df,
            device=device,
            ckpt_path=ckpt_path,
            tracker=tracker,
            log_prefix="full",
            tracking_group=tracking_group,
        )
        elapsed = time.time() - start
        tracker.log({"run/train_seconds": elapsed})
        tracker.finish()
        return {
            "task": task,
            "mode": "full",
            "checkpoint": run["checkpoint"],
            "best_metric": run["best_metric"],
            "monitor": run["monitor"],
            "train_seconds": float(elapsed),
        }

    if mode == "kfold":
        k = int(cfg.get("validation", {}).get("k", 5))
        fold_results: list[dict[str, Any]] = []
        split_summary_rows: list[list[Any]] = []
        oof_true: list[np.ndarray] = []
        oof_prob: list[np.ndarray] = []

        for fold_idx in range(k):
            train_loader, val_loader, train_df, val_df = build_dataloaders(cfg, fold_index=fold_idx)
            ckpt_path = output_dir / f"classification_fold{fold_idx}_best.pt"
            run = _run_single_training(
                cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                train_df=train_df,
                device=device,
                ckpt_path=ckpt_path,
                tracker=tracker,
                log_prefix=f"fold{fold_idx}",
                tracking_group=tracking_group,
            )
            fold_results.append(
                {
                    "fold": fold_idx,
                    "checkpoint": run["checkpoint"],
                    "f1": run["best_metric"],
                }
            )
            best_outputs = run.get("_best_outputs")
            if best_outputs is not None:
                fold_y_true, fold_y_prob, _ = best_outputs
                oof_true.append(fold_y_true)
                oof_prob.append(fold_y_prob)
            n_train = int(len(train_df))
            n_val = int(len(val_df)) if val_df is not None else 0
            train_pos = int((train_df["label"] == 1).sum())
            val_pos = int((val_df["label"] == 1).sum()) if val_df is not None else 0
            split_summary_rows.append(
                [
                    int(fold_idx),
                    n_train,
                    n_val,
                    float(train_pos / max(1, n_train)),
                    float(val_pos / max(1, n_val)) if n_val > 0 else 0.0,
                ]
            )

        f1_values = np.array([x["f1"] for x in fold_results], dtype=np.float32)
        elapsed = time.time() - start
        summary = {
            "task": task,
            "mode": "kfold",
            "k": k,
            "fold_results": fold_results,
            "mean_f1": float(np.mean(f1_values)),
            "std_f1": float(np.std(f1_values)),
            "best_metric": float(np.mean(f1_values)),
            "monitor": "mean_f1",
            "train_seconds": float(elapsed),
        }

        auto_threshold_enabled = bool(cfg.get("eval", {}).get("auto_threshold_search", True))
        if auto_threshold_enabled and oof_true and oof_prob:
            y_true_oof = np.concatenate(oof_true).astype(np.float32)
            y_prob_oof = np.concatenate(oof_prob).astype(np.float32)
            default_threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
            best_threshold = find_best_f1_threshold(
                y_true_oof,
                y_prob_oof,
                min_threshold=float(cfg.get("eval", {}).get("threshold_search_min", 0.05)),
                max_threshold=float(cfg.get("eval", {}).get("threshold_search_max", 0.95)),
                steps=int(cfg.get("eval", {}).get("threshold_search_steps", 181)),
            )
            default_metrics = classification_metrics(y_true=y_true_oof, y_prob=y_prob_oof, threshold=default_threshold)
            threshold_payload = {
                "k": k,
                "seed": int(cfg.get("seed", 42)),
                "oof_size": int(len(y_true_oof)),
                "default_threshold": float(default_threshold),
                "f1_at_default_threshold": float(default_metrics["f1"]),
                "oof_best_threshold": float(best_threshold["threshold"]),
                "oof_best_f1": float(best_threshold["f1"]),
                "oof_best_precision": float(best_threshold["precision"]),
                "oof_best_recall": float(best_threshold["recall"]),
            }
            threshold_path = _threshold_output_path(cfg, k)
            threshold_path.write_text(json.dumps(threshold_payload, indent=2))

            summary["oof_best_threshold"] = float(best_threshold["threshold"])
            summary["oof_best_f1"] = float(best_threshold["f1"])
            summary["oof_f1_at_default_threshold"] = float(default_metrics["f1"])
            summary["threshold_file"] = str(threshold_path)

        tracker.log(
            {
                "kfold/mean_f1": summary["mean_f1"],
                "kfold/std_f1": summary["std_f1"],
                "run/train_seconds": summary["train_seconds"],
            }
        )
        if "oof_best_threshold" in summary:
            tracker.log(
                {
                    "kfold/oof_best_threshold": summary["oof_best_threshold"],
                    "kfold/oof_best_f1": summary["oof_best_f1"],
                    "kfold/oof_f1_at_default_threshold": summary["oof_f1_at_default_threshold"],
                }
            )
        tracker.log_table(
            key="kfold/fold_summary",
            columns=["fold", "f1", "checkpoint"],
            rows=[[int(x["fold"]), float(x["f1"]), str(x["checkpoint"])] for x in fold_results],
        )
        tracker.log_table(
            key="kfold/split_summary",
            columns=["fold", "n_train", "n_val", "train_pos_rate", "val_pos_rate"],
            rows=split_summary_rows,
        )
        tracker.finish()
        return summary

    if mode != "holdout":
        tracker.finish()
        raise ValueError("validation.mode must be one of: holdout, kfold")

    train_loader, val_loader, train_df, _ = build_dataloaders(cfg)
    ckpt_path = output_dir / "classification_holdout_best.pt"
    run = _run_single_training(
        cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        train_df=train_df,
        device=device,
        ckpt_path=ckpt_path,
        tracker=tracker,
        log_prefix="holdout",
        tracking_group=tracking_group,
    )

    elapsed = time.time() - start
    tracker.log({"run/train_seconds": elapsed})
    tracker.finish()

    return {
        "task": task,
        "mode": "holdout",
        "checkpoint": run["checkpoint"],
        "best_metric": run["best_metric"],
        "monitor": run["monitor"],
        "train_seconds": float(elapsed),
    }


def evaluate(cfg: dict[str, Any], ckpt_path: str) -> dict[str, float]:
    """Load checkpoint and evaluate on the configured validation split.

    In k-fold mode, `validation.fold_index` selects the fold to evaluate.
    """
    if cfg["task"] != "classification":
        raise ValueError("Only classification task is supported.")

    if bool(cfg["train"].get("full_data", False)):
        raise ValueError("Evaluation is not available when train.full_data=true.")

    device = _device_from_cfg(cfg)
    _configure_runtime(cfg, device)
    mode = str(cfg.get("validation", {}).get("mode", "holdout")).lower()
    fold_index = int(cfg.get("validation", {}).get("fold_index", 0)) if mode == "kfold" else None
    threshold = _resolve_eval_threshold(cfg, mode)
    precision = _resolve_precision(cfg, device)
    channels_last = _use_channels_last(cfg, device)

    _, val_loader, _, _ = build_dataloaders(cfg, fold_index=fold_index)
    if val_loader is None:
        raise ValueError("No validation loader available.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_group = checkpoint.get("tracking_group")
    model = build_model(task="classification", model_cfg=cfg["model"]).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["state_dict"])

    metrics, y_true, y_prob, file_names = _evaluate(
        model,
        val_loader,
        device,
        threshold=threshold,
        precision=precision,
        channels_last=channels_last,
    )
    if mode == "kfold":
        metrics["fold"] = float(fold_index if fold_index is not None else 0)
    metrics["threshold"] = float(threshold)

    if bool(cfg.get("tracking", {}).get("log_eval_runs", True)):
        tracker = Tracker(
            cfg,
            run_name_suffix="eval",
            job_type="eval",
            run_group=str(checkpoint_group) if checkpoint_group else None,
        )
        tracker.log({f"eval/{k}": v for k, v in metrics.items()})
        tracker.log_classification_diagnostics(
            prefix="eval",
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            file_names=file_names,
        )
        tracker.finish()
    return metrics


def predict(cfg: dict[str, Any], ckpt_path: str, out_path: str, split: str = "val") -> str:
    """Generate classification predictions and export CSV.

    Output columns:
    - file_name
    - prob_nodule
    - pred_label

    In k-fold mode, `validation.fold_index` selects fold for train/val splits.
    """
    if cfg["task"] != "classification":
        raise ValueError("Only classification task is supported.")

    device = _device_from_cfg(cfg)
    _configure_runtime(cfg, device)
    mode = str(cfg.get("validation", {}).get("mode", "holdout")).lower()
    fold_index = int(cfg.get("validation", {}).get("fold_index", 0)) if mode == "kfold" else None
    precision = _resolve_precision(cfg, device)
    channels_last = _use_channels_last(cfg, device)

    loader = build_prediction_loader(cfg, split=split, fold_index=fold_index)

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_group = checkpoint.get("tracking_group")
    model = build_model(task="classification", model_cfg=cfg["model"]).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    threshold = _resolve_eval_threshold(cfg, mode)
    out_rows: list[dict[str, Any]] = []
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=non_blocking)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with _autocast_context(device, precision):
                prob = torch.sigmoid(model(x)).detach().cpu().numpy()

            for i, file_name in enumerate(batch["file_name"]):
                p = float(prob[i])
                out_rows.append(
                    {
                        "file_name": str(file_name),
                        "prob_nodule": p,
                        "pred_label": int(p >= threshold),
                    }
                )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False)

    if bool(cfg.get("tracking", {}).get("log_predict_runs", True)):
        tracker = Tracker(
            cfg,
            run_name_suffix="predict",
            job_type="predict",
            run_group=str(checkpoint_group) if checkpoint_group else None,
        )
        tracker.log(
            {
                "predict/rows": float(len(out_rows)),
                "predict/threshold": threshold,
                "predict/split": split,
            }
        )
        if bool(cfg.get("tracking", {}).get("log_artifacts", True)):
            artifact_name = _sanitize_artifact_name(
                f"{str(checkpoint_group) if checkpoint_group else 'predict'}-{Path(ckpt_path).stem}-{split}-predictions"
            )
            tracker.log_file_artifact(
                out,
                artifact_name=artifact_name,
                artifact_type="predictions",
                metadata={
                    "split": split,
                    "rows": len(out_rows),
                    "checkpoint": str(ckpt_path),
                    "tracking_group": checkpoint_group,
                },
            )
        tracker.finish()
    return str(out)
