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

Optional preprocessing dependency (for CLAHE):
```bash
uv sync --extra cv
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

## Training knobs
- `data.normalization`: input normalization (`enabled`, `mean`, `std`)
- `data.preprocessing.clahe`: CLAHE preprocessing (`enabled`, `clip_limit`, `tile_grid_size`)
- `train.loss.name`: `bce` or `focal`
- `train.loss.focal_gamma`, `train.loss.focal_alpha`, `train.loss.use_pos_weight`
- `train.scheduler.name`: `none` or `cosine`
- `train.scheduler.warmup_epochs`, `train.scheduler.min_lr`
- `train.num_workers`, `train.pin_memory`, `train.persistent_workers`, `train.prefetch_factor`
- `train.precision`: `fp32` / `bf16` / `fp16` (A100: `bf16` recommended)
- `train.channels_last`, `train.cudnn_benchmark`, `train.tf32`, `train.matmul_precision`
- `train.compile`, `train.compile_mode` (optional acceleration with `torch.compile`)
- `eval.auto_threshold_search`: in `kfold`, compute best OOF threshold (F1) and save it to `artifacts/thresholds/`
- `eval.use_oof_threshold`: in `kfold` eval/predict, load and use saved OOF threshold automatically

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
This run now saves OOF threshold search results in `artifacts/thresholds/classification_kfold_k5_seed42.json`.

### Evaluate one fold model
```bash
uv run python -m nodulocc.cli eval --config configs/classification.yaml \
  --ckpt artifacts/checkpoints/classification_fold0_best.pt \
  --fold 0 \
  --override validation.mode=kfold \
  --override validation.k=5 \
  --override eval.use_oof_threshold=true
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

### Focal loss + cosine scheduler example
```bash
uv run python -m nodulocc.cli train --config configs/classification.yaml \
  --override train.loss.name=focal \
  --override train.loss.focal_gamma=2.0 \
  --override train.loss.focal_alpha=0.25 \
  --override train.scheduler.name=cosine \
  --override train.scheduler.warmup_epochs=1 \
  --override train.scheduler.min_lr=1e-5
```

### A100 speed-oriented example
```bash
uv run python -m nodulocc.cli train --config configs/classification.yaml \
  --override train.batch_size=32 \
  --override train.num_workers=12 \
  --override train.precision=bf16 \
  --override train.channels_last=true \
  --override train.cudnn_benchmark=true \
  --override train.tf32=true
```

### Enable CLAHE
```bash
uv run python -m nodulocc.cli train --config configs/classification.yaml \
  --override data.preprocessing.clahe.enabled=true \
  --override data.preprocessing.clahe.clip_limit=2.0 \
  --override data.preprocessing.clahe.tile_grid_size=8
```

## Notes
- Split indices are cached in `artifacts/splits/` for reproducibility.
- Classification uses `nih_filtered_images`.
