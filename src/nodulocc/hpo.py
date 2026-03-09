"""Hyperparameter optimization utilities (Optuna)."""

from __future__ import annotations

import copy
from typing import Any

from .engine import train


def run_sweep(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run Optuna search on classification configuration."""
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("Optuna is required for sweep. Install with: uv sync --extra exp") from exc

    if cfg.get("task") != "classification":
        raise ValueError("Only classification task is supported.")

    sweep_cfg = cfg.get("sweep", {})
    n_trials = int(sweep_cfg.get("n_trials", 10))
    direction = str(sweep_cfg.get("direction", "maximize"))

    def objective(trial: "optuna.trial.Trial") -> float:
        """Train one trial configuration and return objective value."""
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.setdefault("tracking", {})["enabled"] = False

        # Keep sweep in holdout mode for speed and consistent objective.
        trial_cfg.setdefault("validation", {})["mode"] = "holdout"
        trial_cfg["train"]["full_data"] = False

        trial_cfg["train"]["lr"] = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        trial_cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        trial_cfg["train"]["epochs"] = int(sweep_cfg.get("epochs_per_trial", 2))
        trial_cfg["train"]["img_size"] = trial.suggest_categorical("img_size", [256, 384, 512])
        trial_cfg["model"]["backbone"] = trial.suggest_categorical(
            "backbone",
            ["tiny_cnn", "tf_efficientnet_b0.ns_jft_in1k", "tf_efficientnet_b3.ns_jft_in1k"],
        )

        result = train(trial_cfg)
        return float(result["best_metric"])

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": n_trials,
        "direction": direction,
    }
