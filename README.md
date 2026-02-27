# NoduLoCC - Compact Modular Base (Classification)

Minimal, readable baseline focused on:
- Binary classification (`Nodule` vs `No Finding`)

## Project layout
- `configs/`: base + classification config
- `src/nodulocc/`: core code (CLI, data, models, engine, metrics, tracking, HPO)
- `NARVAL_SLURM_TUTORIAL.md`: step-by-step HPC/Slurm guide for Narval (Calcul Quebec)

## Setup
```bash
cd nodulocc
uv sync
```

Optional experiment tools:
```bash
uv sync --extra exp
```

## Weights & Biases login
If you want online experiment tracking, login once before training:
```bash
uv sync --extra exp
uv run wandb login
```
Then paste your API key from `https://wandb.ai/authorize`.

## W&B logging content
When `tracking.enabled=true`, the project logs:
- dataset exploration once at training start:
  - number of images (global)
  - number of nodules / no-finding
  - imbalance ratio (`no_finding / nodule`)
  - class distribution chart (global)
  - sample images table (global)
- training/validation scalar curves by epoch (`loss`, `f1`, `auc`, etc.)
- k-fold split summary table (`n_train`, `n_val`, `train_pos_rate`, `val_pos_rate`) when `validation.mode=kfold`
- confusion matrix (best epoch on each run/fold)
- ROC and PR curves
- probability histograms for positive/negative classes
- prediction table (`file_name`, `y_true`, `y_prob`, `y_pred`) with row cap
- checkpoint artifacts (`*.pt`)
- prediction CSV artifacts for `predict`

Main tracking switches (in `configs/base.yaml`):
- `tracking.log_data_exploration`
- `tracking.data_samples_per_run`
- `tracking.auto_run_name` (auto-generate informative names when `run_name=null`)
- `tracking.run_name_prefix` (prefix added to auto-generated names)
- `tracking.group` (optional: force grouping across train/eval/predict runs)
- `tracking.log_artifacts`
- `tracking.log_confusion_matrix`
- `tracking.log_curves`
- `tracking.log_prob_histograms`
- `tracking.log_prediction_table`
- `tracking.prediction_table_max_rows`
- `tracking.log_eval_runs`
- `tracking.log_predict_runs`

Run naming behavior:
- if `tracking.run_name` is set: exact base name is used (`-eval` / `-predict` suffix for those jobs)
- if `tracking.run_name` is null and `tracking.auto_run_name=true`: name is generated from task/job/mode/backbone/seed/timestamp
- if `tracking.run_name` is null and `tracking.auto_run_name=false`: W&B picks a random name

## Validation modes
Config keys:
- `validation.mode`: `holdout` or `kfold`
- `validation.k`: number of folds when `mode=kfold`
- `validation.fold_index`: fold selected for `eval` / `predict`
- `train.full_data`: `true` to train on 100% data (no validation)

## CLI
### Holdout training (80/20)
```bash
uv run python -m nodulocc.cli train --config configs/classification.yaml
```

### K-fold training
```bash
uv run python -m nodulocc.cli train --config configs/classification.yaml \
  --override validation.mode=kfold \
  --override validation.k=5
```

### Evaluate one fold model
```bash
uv run python -m nodulocc.cli eval --config configs/classification.yaml \
  --ckpt artifacts/checkpoints/classification_fold0_best.pt \
  --fold 0 \
  --override validation.mode=kfold \
  --override validation.k=5
```

### Full-data final training (100%)
```bash
uv run python -m nodulocc.cli train --config configs/classification.yaml \
  --override train.full_data=true
```

### Predict
```bash
uv run python -m nodulocc.cli predict --config configs/classification.yaml \
  --ckpt artifacts/checkpoints/classification_holdout_best.pt \
  --out artifacts/preds_classification.csv
```

## Notes
- Split indices are cached in `artifacts/splits/` for reproducibility.
- Classification uses `nih_filtered_images`.
