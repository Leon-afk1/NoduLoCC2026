# MIL Notes (Baseline -> Guided v2)

This note summarizes what was implemented for MIL and what changed in the latest update.

## 1) MIL Baseline (v1)

Goal: classify one CXR image by aggregating predictions from multiple local patches.

### Data path
- Each image is first preprocessed with the existing pipeline (`v1` or `team_v2`).
- `MilPatchDataset` returns a bag of patches of shape `[N, 3, P, P]`.
- Patch centers were mostly random in a thorax ellipse.
- Optional localization prior (`localization_labels.csv`) could inject one positive-biased center for positive training samples.

### Model path
- `model.type=mil_patch` builds an attention MIL classifier:
  - patch backbone encoder (timm),
  - attention weights over patches,
  - weighted pooled embedding,
  - binary classification head.

## 2) MIL Guided Sampler (v2, current)

Problem observed in v1:
- AUC was decent but F1/precision plateaued.
- Runtime was high because many sampled patches were low-information.

### What changed
- Added `model.mil.sampling_mode`:
  - `guided` (new default),
  - `random` (v1 behavior fallback).
- In guided mode:
  1. Build a local variance score map on preprocessed image.
  2. Evaluate a thorax candidate grid.
  3. Keep top informative candidates (`topk_fraction`).
  4. Sample patch centers from this informative pool (+ exploration probability in train).
  5. Keep localization-prior injection for positive train samples.

### Why this should help
- Higher ratio of informative patches per bag.
- Better precision/F1 potential at similar or lower compute.
- More stable eval behavior (deterministic center ordering in eval).

## 3) Key Config Knobs

Under `model.mil`:
- `sampling_mode`: `guided` or `random`
- `candidate_grid`: candidate points per axis for guided mode
- `topk_fraction`: fraction of highest-scoring candidates to keep
- `train_explore_prob`: probability to sample random thorax center in train
- `guided_jitter`: small train-time jitter around guided centers
- `score_kernel_size`: odd kernel size for local variance score map
- `num_patches`: patches per bag

## 4) Current Recommended Starting Config

`configs/r1_b4_mil_patch_team_v2.yaml`:
- `sampling_mode: guided`
- `num_patches: 10` (reduced from 16 for faster training)
- `batch_size: 8` (adjusted accordingly)

## 5) Practical Next Step

Run one holdout with this config and compare:
- best `holdout/f1`
- `holdout/precision` at best epoch
- runtime (`run/train_seconds`)

If F1 improves and runtime drops, keep guided mode as new default.
