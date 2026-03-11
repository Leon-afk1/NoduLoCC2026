"""Command line interface for classification-only workflows.

Entry point:
`python -m nodulocc.cli <command> ...`
"""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .ensemble_csv import run_csv_ensemble
from .ensemble import run_two_model_ensemble
from .engine import evaluate, predict, train
from .hpo import run_sweep


def _parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    p = argparse.ArgumentParser(description="NoduLoCC compact CLI (classification only)")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--override", action="append", default=[])

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--fold", type=int, required=False)
    p_eval.add_argument("--override", action="append", default=[])

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--config", required=True)
    p_pred.add_argument("--ckpt", required=True)
    p_pred.add_argument("--out", required=True)
    p_pred.add_argument("--split", default="val", choices=["train", "val", "all"])
    p_pred.add_argument("--fold", type=int, required=False)
    p_pred.add_argument("--override", action="append", default=[])

    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--config", required=True)
    p_sweep.add_argument("--override", action="append", default=[])

    p_ensemble = sub.add_parser("ensemble")
    p_ensemble.add_argument("--config-a", required=True)
    p_ensemble.add_argument("--ckpt-a", required=True)
    p_ensemble.add_argument("--config-b", required=True)
    p_ensemble.add_argument("--ckpt-b", required=True)
    p_ensemble.add_argument("--out-dir", required=True)
    p_ensemble.add_argument("--split", default="val", choices=["train", "val", "all"])
    p_ensemble.add_argument("--reference", default="a", choices=["a", "b"])
    p_ensemble.add_argument("--weight-a", type=float, default=0.5)
    p_ensemble.add_argument("--default-threshold", type=float, default=0.5)
    p_ensemble.add_argument("--fold-a", type=int, required=False)
    p_ensemble.add_argument("--fold-b", type=int, required=False)
    p_ensemble.add_argument("--override-a", action="append", default=[])
    p_ensemble.add_argument("--override-b", action="append", default=[])

    p_ensemble_csv = sub.add_parser("ensemble-csv")
    p_ensemble_csv.add_argument("--pred-a", required=True)
    p_ensemble_csv.add_argument("--pred-b", required=True)
    p_ensemble_csv.add_argument("--config-ref", required=True)
    p_ensemble_csv.add_argument("--out-dir", required=True)
    p_ensemble_csv.add_argument("--labels-csv", required=False, default=None)
    p_ensemble_csv.add_argument("--split", default="val", choices=["train", "val", "all"])
    p_ensemble_csv.add_argument("--fold", type=int, required=False)
    p_ensemble_csv.add_argument("--weight-a", type=float, default=0.5)
    p_ensemble_csv.add_argument("--threshold-a", type=float, default=0.5)
    p_ensemble_csv.add_argument("--threshold-b", type=float, default=0.5)
    p_ensemble_csv.add_argument("--default-threshold", type=float, default=0.5)
    p_ensemble_csv.add_argument("--override-ref", action="append", default=[])

    return p


def _validate_task(cfg: dict) -> dict:
    """Ensure config is aligned with classification-only project scope."""
    task = cfg.get("task", "classification")
    if task != "classification":
        raise ValueError("This project currently supports classification only.")
    cfg["task"] = "classification"
    return cfg


def _apply_fold(cfg: dict, fold: int | None) -> dict:
    """Apply optional fold selection for k-fold eval/predict."""
    if fold is not None:
        cfg.setdefault("validation", {})["fold_index"] = int(fold)
    return cfg


def main() -> None:
    """Execute the requested CLI subcommand and print JSON output."""
    args = _parser().parse_args()

    if args.command == "ensemble":
        result = run_two_model_ensemble(args)
        print(json.dumps(result, indent=2))
        return

    if args.command == "ensemble-csv":
        result = run_csv_ensemble(args)
        print(json.dumps(result, indent=2))
        return

    cfg = _validate_task(load_config(args.config, overrides=args.override))

    if args.command == "train":
        result = train(cfg)
        print(json.dumps(result, indent=2))
        return

    if args.command == "eval":
        cfg = _apply_fold(cfg, args.fold)
        result = evaluate(cfg, ckpt_path=args.ckpt)
        print(json.dumps(result, indent=2))
        return

    if args.command == "predict":
        cfg = _apply_fold(cfg, args.fold)
        out_path = predict(cfg, ckpt_path=args.ckpt, out_path=args.out, split=args.split)
        print(json.dumps({"out_path": out_path}, indent=2))
        return

    if args.command == "sweep":
        result = run_sweep(cfg)
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
