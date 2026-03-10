# NoduLoCC Codebase Analysis

## 1. Project Scope

This repository implements a compact, modular pipeline for binary chest X-ray classification:

- `0`: `No Finding`
- `1`: `Nodule`

The codebase supports:

- training (`holdout`, `kfold`, `full-data`)
- evaluation and CSV prediction export
- optional experiment tracking (Weights & Biases)
- optional hyperparameter search (Optuna)
- optional OpenCV-based preprocessing profile
- optional external checkpoint initialization

Core implementation lives in `src/nodulocc/`.

## 2. End-to-End Architecture

High-level flow:

1. CLI loads YAML config and CLI overrides.
2. Config loader resolves `base_config` inheritance and merge rules.
3. Data module reads metadata CSV, resolves image paths, and builds train/val splits.
4. Dataset applies preprocessing, augmentation, and normalization.
5. Model factory builds either a global classifier or a MIL patch classifier.
6. Engine runs training, checkpointing, validation, and metrics.
7. Optional tracker logs metrics, diagnostics, tables, and artifacts.
8. Evaluation and prediction reuse checkpoints and split logic.

Design principle:

- each module owns one concern (config, data, model, engine, metrics, tracking)
- most runtime behavior is controlled by YAML (`configs/base.yaml` + derived presets)

## 3. Repository Map

Main package:

- `src/nodulocc/config.py`: hierarchical config loading and overrides
- `src/nodulocc/cli.py`: command entry points (`train`, `eval`, `predict`, `sweep`)
- `src/nodulocc/data.py`: dataframe loading, split caching, datasets, dataloaders
- `src/nodulocc/preprocess_profiles.py`: advanced preprocessing profile `team_v2`
- `src/nodulocc/models.py`: model and backbone factory
- `src/nodulocc/engine.py`: training, evaluation, prediction, checkpoint lifecycle
- `src/nodulocc/metrics.py`: classification metrics and threshold search
- `src/nodulocc/tracking.py`: W&B abstraction layer
- `src/nodulocc/hpo.py`: Optuna sweep wrapper
- `src/nodulocc/preview_preprocessing.py`: preprocessing visualization utility
- `src/nodulocc/export_xrv_checkpoint.py`: TorchXRayVision checkpoint converter

Configuration:

- `configs/base.yaml`: complete default configuration
- `configs/classification*.yaml`: baseline variants
- `configs/r*.yaml`: experiment presets

Operations and HPC:

- `train_gpu.slurm` and `train_gpu.slurm.example`
- `submit_and_sync.sh`
- `sweep_threshold.slurm`
- `docs/NARVAL_SLURM_TUTORIAL.md`

## 4. Module-Level Analysis

## 4.1 `config.py`

Responsibility:

- load one YAML file
- resolve optional `base_config` chain recursively
- apply runtime dotted-key overrides

Key behavior:

- `_merge_dicts` performs recursive merge where child overrides parent keys
- `_parse_value` converts CLI text values to `bool`, `None`, `int`, `float` when possible
- `_set_nested` builds nested dictionaries from dotted paths
- `load_config` appends `_config_path`, reused later to resolve relative file references

Practical implication:

- config inheritance is deterministic and easy to audit
- runtime experiments can be changed without editing YAML files

## 4.2 `cli.py`

Responsibility:

- expose user-facing commands
- enforce the project scope (`classification` only)

Commands:

- `train`
- `eval`
- `predict`
- `sweep`

Important details:

- `--override` can be repeated and is applied before command execution
- `--fold` injects `validation.fold_index` for `eval/predict` in k-fold mode
- outputs are printed as JSON for scripting compatibility

## 4.3 `data.py`

This is the largest module and includes data routing, split logic, transforms, datasets, and dataloaders.

### 4.3.1 Metadata loading and path routing

`load_classification_dataframe(cfg)`:

- reads labels CSV (`data.classification.labels_csv`)
- validates required columns (`file_name`, `label`, `LIDC_ID`)
- maps labels with `LABEL_MAP`
- derives source:
  - `LIDC` when `LIDC_ID` exists
  - `NIH` otherwise
- resolves file paths via source-specific folders
- drops rows whose files do not exist

Routing strategy:

- fast path: `source_dir / file_name`
- fallback path: recursive `.png` index only for unresolved files

### 4.3.2 Split strategy and persistence

Supported modes:

- `holdout`: stratified `train_test_split`
- `kfold`: stratified `StratifiedKFold`

Split controls:

- `data.split_stratify`: `label` or `source_label`
- `data.classification.lidc_train_only`: keeps LIDC out of validation splits
- `seed`: deterministic split generation

Split caching:

- split indices are persisted to JSON in `data.split_dir`
- cache name includes seed and split semantics
- this prevents accidental split drift across runs

### 4.3.3 Preprocessing and transforms

Profile selection:

- `v1`: torchvision-based pipeline
- `team_v2`: OpenCV-heavy robust pipeline from `preprocess_profiles.py`

`v1` transform chain:

1. resize to square
2. optional CLAHE
3. optional train-time augmentation
4. grayscale to 3 channels
5. tensor conversion
6. optional normalization

Normalization:

- controlled by `data.normalization.enabled`
- expects 3-element mean/std lists

Augmentation controls:

- horizontal flip probability
- color jitter settings (`brightness`, `contrast`, `saturation`, `hue`)

### 4.3.4 Dataset classes

`ClassificationDataset`:

- returns a dict with `image`, `label`, `file_name`
- stores paths/labels as arrays for lower per-item overhead
- supports both `v1` and `team_v2` preprocessing paths

`MilPatchDataset`:

- wraps `ClassificationDataset` and converts each image into a bag of patches
- output shape per sample: `[N, 3, P, P]`
- supports guided or random center sampling

MIL center sampling details:

- optional positive localization prior from CSV (`x`, `y` points)
- guided mode computes a local variance map and ranks thorax candidates
- train mode mixes guided exploitation and random exploration
- eval/predict mode is deterministic per sample

### 4.3.5 Dataloaders and imbalance controls

`build_dataloaders`:

- builds train and optional val loaders
- optional `WeightedRandomSampler` for imbalance

`_classification_sampler`:

- sample weights are inverse class frequency
- replacement sampling keeps minority exposure high

`build_prediction_loader`:

- supports `train`, `val`, `all` prediction splits
- always disables train-time augmentation for inference

## 4.4 `preprocess_profiles.py`

Responsibility:

- implement advanced profile `team_v2`
- keep heavy image logic out of `data.py`

`TeamV2Settings`:

- validated config object from `data.preprocessing.team_v2.*`
- enforces valid percentile bounds and odd kernel sizes

`TeamV2Preprocessor` pipeline:

1. load image with `cv2.IMREAD_UNCHANGED`
2. convert to grayscale if needed
3. percentile clipping and robust normalization
4. optional smooth thorax masking
5. optional CLAHE
6. optional light Gaussian blur
7. aspect-ratio-preserving resize + reflection padding
8. float tensor in `[0, 1]`, replicated to 3 channels

Note:

- no train-time augmentation in this module
- no mean/std normalization in this module
- augmentation and normalization remain in dataset flow

## 4.5 `models.py`

Responsibility:

- central model factory

Available backbones:

- `tiny_cnn` test backbone
- any valid timm backbone (`timm.create_model(..., num_classes=0, global_pool="avg")`)

Model types:

- `global`:
  - image-level backbone features
  - dropout + linear head
- `mil_patch`:
  - patch-wise backbone encoding
  - attention network over patches
  - weighted pooled feature
  - dropout + linear head

Output convention:

- raw logits (sigmoid applied in engine when needed)

## 4.6 `metrics.py`

`classification_metrics`:

- computes `accuracy`, `precision`, `recall`, `f1`, `auc`
- returns `auc = nan` if only one class exists in the target vector

`find_best_f1_threshold`:

- grid search over threshold range
- objective: maximize F1
- tie-break: prefer threshold closer to `0.5`

## 4.7 `engine.py`

This module orchestrates runtime setup, training loops, evaluation, prediction, checkpoint IO, and threshold handling.

### 4.7.1 Runtime and precision setup

- automatic device resolution (`cuda` if available)
- CUDA backend tuning (`cudnn_benchmark`, TF32, matmul precision)
- precision modes: `fp32`, `bf16`, `fp16`
- optional channels-last memory format
- optional `torch.compile` with graceful fallback
- `GradScaler` only when `fp16` on CUDA

### 4.7.2 External checkpoint initialization

`_load_external_init_weights` supports heterogeneous checkpoint formats:

- accepts nested dict keys (`state_dict`, `model`, `weights`, etc.)
- strips configured key prefixes
- optional auto-prefixing into model keys (`""` or `"backbone."`)
- can skip likely head layers for transfer learning
- loads only keys with exact shape compatibility

This enables robust transfer from external CXR-pretrained checkpoints.

### 4.7.3 Losses and learning-rate scheduling

Loss options:

- BCE with logits
- custom binary focal loss with optional alpha and class reweighting

Class imbalance support:

- `pos_weight` computed from train split class counts

Scheduler options:

- `none`
- `cosine` with warmup and min learning rate

### 4.7.4 Single-run training loop

`_run_single_training`:

- builds model and optimizer
- runs epoch loop with mixed precision when configured
- computes train loss and optional validation metrics
- tracks best checkpoint by monitor:
  - `f1` when validation exists
  - `train_loss` in full-data mode
- applies optional early stopping
- logs diagnostics and model artifact when tracking is enabled

Saved checkpoint payload includes:

- `state_dict`
- full config snapshot
- best metric and monitor name
- tracking group metadata

### 4.7.5 Training modes

`train(cfg)` supports:

- `full-data`: one run on the entire dataset, no validation
- `holdout`: one train/val split
- `kfold`: loop over folds, return fold-level and aggregate statistics

K-fold extras:

- per-fold best F1 collection
- out-of-fold probability aggregation
- optional automatic threshold search on OOF predictions
- threshold result persisted to JSON

### 4.7.6 Evaluation flow

`evaluate(cfg, ckpt_path)`:

- rebuilds model architecture from config
- forces `pretrained=false` for checkpoint loading consistency
- evaluates on configured validation split
- supports optional OOF-threshold loading in k-fold mode
- can create a dedicated tracking run (`job_type=eval`)

### 4.7.7 Prediction flow

`predict(cfg, ckpt_path, out_path, split)`:

- runs inference on `train`, `val`, or `all`
- writes CSV with:
  - `file_name`
  - `prob_nodule`
  - `pred_label`
- supports optional OOF threshold usage in k-fold mode
- can log prediction artifacts through tracking

## 4.8 `tracking.py`

Responsibility:

- isolate tracking concerns from core training logic
- degrade gracefully when tracking is disabled or unavailable

Provider:

- currently `wandb` only

Capabilities:

- scalar logging
- table logging
- file artifact logging
- classification diagnostics:
  - confusion matrix
  - ROC and PR curves
  - positive and negative score histograms
  - optional prediction table sampling
- dataset exploration logging:
  - class counts and imbalance ratios
  - class distribution charts
  - balanced sample image tables

Naming and grouping:

- supports explicit run names
- supports generated run names from task/mode/backbone/seed/timestamp
- supports run grouping across train/eval/predict

## 4.9 `hpo.py`

Responsibility:

- Optuna-based hyperparameter sweep wrapper

Behavior:

- disables tracking inside trials
- forces holdout mode and disables full-data mode
- samples:
  - learning rate
  - weight decay
  - image size
  - backbone choice
- objective is `train(cfg)["best_metric"]`

## 4.10 `preview_preprocessing.py`

Responsibility:

- visualize preprocessed outputs before training

Workflow:

- sample NIH and LIDC examples
- run them through `ClassificationDataset`
- build side-by-side image grids:
  - left: robustly scaled original
  - right: model-input tensor rendered to RGB
- save:
  - source preview PNGs
  - JSON index with selected file names

## 4.11 `export_xrv_checkpoint.py`

Responsibility:

- convert TorchXRayVision checkpoints into a format compatible with NoduLoCC init loading

Behavior:

- instantiates an XRV model from a weights identifier
- extracts and normalizes state-dict keys
- optionally converts first convolution weights from 1-channel to 3-channel
- writes output payload with `state_dict` and conversion metadata

## 4.12 `__init__.py`

Exports package modules through `__all__`.

## 5. Configuration Reference

## 5.1 `data.*`

- dataset location and split cache location
- NIH/LIDC image subdirectories and label CSV paths
- optional localization CSV for MIL priors
- split stratification policy (`label` or `source_label`)
- optional LIDC-train-only policy
- preprocessing profile selection and profile-specific settings
- augmentation and normalization controls

## 5.2 `validation.*`

- validation mode (`holdout` or `kfold`)
- number of folds
- fold index for evaluation and prediction

## 5.3 `model.*`

- model type (`global` or `mil_patch`)
- backbone id and pretrained flag
- dropout
- optional external init checkpoint controls
- MIL sampling and attention settings

## 5.4 `train.*`

- runtime performance toggles (`precision`, `channels_last`, `compile`, TF32)
- dataloader options (`batch_size`, `num_workers`, `pin_memory`, worker persistence)
- optimization settings (`epochs`, `lr`, `weight_decay`)
- imbalance controls (`use_weighted_sampler`, `loss.use_pos_weight`)
- loss type (`bce` or `focal`) and focal parameters
- scheduler configuration (`none` or `cosine`)
- early stopping controls
- output directory
- `full_data` mode

## 5.5 `eval.*`

- default classification threshold
- k-fold OOF threshold search settings
- threshold output directory
- OOF threshold usage toggle for eval/predict

## 5.6 `tracking.*`

- provider enablement and project metadata
- naming and grouping controls
- artifact and diagnostics toggles
- separate toggles for eval and predict run logging

## 5.7 `sweep.*`

- number of trials
- optimization direction
- epochs per trial

## 6. Preset Configuration Summary

Main preset families:

- `classification*.yaml`: baseline architecture and training variants
- `r1_*`: BCE-focused B4 variants, including `team_v2`
- `r1_*_mil_patch_*`: MIL patch-based guided sampling variant
- `r2_*`: focal-loss-focused variant
- `r3_*`: BCE without positive weighting
- `r_final_*`: deadline/final-stage global runs and external CXR init

Recommended way to understand a preset:

1. open the preset file
2. inspect `base_config`
3. compare only overridden keys against parent config

## 7. Generated Artifacts

Checkpoints:

- holdout: `classification_holdout_best.pt`
- full-data: `classification_full_best.pt`
- k-fold: `classification_fold{idx}_best.pt`

Split cache files:

- holdout split JSON with train and val indices
- k-fold split JSON with fold-wise train and val indices

Threshold files:

- k-fold OOF threshold summary JSON

Prediction exports:

- CSV with probabilities and hard labels

Preprocessing previews:

- per-source PNG grids and an index JSON

## 8. HPC and Operations Scripts

`train_gpu.slurm`:

- stages dataset archive to node-local storage (`$SLURM_TMPDIR`)
- runs training in offline W&B mode

`submit_and_sync.sh`:

- prefetches timm backbone weights
- submits Slurm training job
- waits for completion
- syncs offline W&B runs

`sweep_threshold.slurm`:

- produces validation probabilities
- runs threshold sweep offline
- writes per-threshold metric JSON files

## 9. Known Caveats and Operational Notes

Class-imbalance compensation:

- weighted sampling and positive class weighting can both be enabled
- this can overcompensate if not tuned carefully

Split cache persistence:

- split files are reused once written
- changing data composition may require split cache cleanup for strict reproducibility

Dataset filtering:

- missing image files are dropped during dataframe construction
- this is robust at runtime but can hide dataset integrity issues

`team_v2` dependency:

- requires OpenCV at runtime

Evaluation in full-data mode:

- intentionally disabled because no validation split exists

K-fold threshold fallback:

- if OOF threshold file is missing, eval/predict falls back to `eval.threshold`

MIL localization priors:

- priors are used only during training sampling for positive examples
- eval/predict sampling remains deterministic and does not use localization labels

## 10. Common Commands

Train:

```bash
uv run python -m nodulocc.cli train --config configs/classification.yaml
```

K-fold train:

```bash
uv run python -m nodulocc.cli train --config configs/classification.yaml \
  --override validation.mode=kfold \
  --override validation.k=5
```

Evaluate:

```bash
uv run python -m nodulocc.cli eval \
  --config configs/classification.yaml \
  --ckpt artifacts/checkpoints/classification_holdout_best.pt
```

Predict:

```bash
uv run python -m nodulocc.cli predict \
  --config configs/classification.yaml \
  --ckpt artifacts/checkpoints/classification_holdout_best.pt \
  --out artifacts/preds.csv
```

Preview preprocessing:

```bash
uv run python -m nodulocc.preview_preprocessing \
  --config configs/r1_b4_bce_posw_team_v2_no_double_comp.yaml
```

Export TorchXRayVision checkpoint:

```bash
uv run python -m nodulocc.export_xrv_checkpoint \
  --weights resnet50-res512-all \
  --out artifacts/pretrained/xrv_resnet50_res512_all_init.pt
```

## 11. Summary

The repository is organized around clear module boundaries:

- configuration and CLI orchestration
- data loading and preprocessing
- model construction
- training and inference engine
- metrics, tracking, and optional HPO

The current implementation is production-friendly for small to medium experiments and keeps most behavior configurable without code edits.
