"""Metric helpers for classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics from probabilities.

    Returns accuracy, precision, recall, f1, auc (ROC), avg_precision (PR-AUC),
    specificity, npv, and raw confusion matrix counts (tp, fp, tn, fn).
    """
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    auc = float("nan")
    avg_precision = float("nan")
    if len(np.unique(y_true)) > 1:
        if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
            n_nan = int(np.sum(np.isnan(y_prob)))
            print(f"[metrics] WARNING: y_prob contient {n_nan} NaN/Inf — AUC/PR-AUC mis à NaN")
        else:
            auc = float(roc_auc_score(y_true, y_prob))
            avg_precision = float(average_precision_score(y_true, y_prob))

    # Confusion matrix → tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    specificity = tn / max(1, tn + fp)           # true negative rate
    npv = tn / max(1, tn + fn)                   # negative predictive value

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "avg_precision": avg_precision,
        "specificity": specificity,
        "npv": npv,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 181,
) -> dict[str, float]:
    """Search threshold that maximizes F1 on a probability vector.

    Returns:
        Dict containing:
        - `threshold`: best threshold found on the grid
        - `f1`: best F1 achieved
        - `precision`: precision at the selected threshold
        - `recall`: recall at the selected threshold
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")
    if min_threshold >= max_threshold:
        raise ValueError("min_threshold must be < max_threshold")

    y_true_i = y_true.astype(np.int32)
    y_prob_f = y_prob.astype(np.float32)

    thresholds = np.linspace(min_threshold, max_threshold, steps, dtype=np.float32)
    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for thr in thresholds:
        y_pred = (y_prob_f >= thr).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_i,
            y_pred,
            average="binary",
            zero_division=0,
        )
        # Stable tie-break: prefer threshold closer to 0.5.
        if (f1 > best["f1"]) or (f1 == best["f1"] and abs(float(thr) - 0.5) < abs(best["threshold"] - 0.5)):
            best = {
                "threshold": float(thr),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best
