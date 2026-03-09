#!/bin/bash
#SBATCH --account=def-zonata
#SBATCH --job-name=nodu-thr-sweep
#SBATCH --time=0-01:30
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail

module load StdEnv/2023
module load python/3.14.2

PROJECT_DIR="$SCRATCH/NoduLoCC2026"
cd "$PROJECT_DIR"
source .venv/bin/activate
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HF_HOME="$SCRATCH/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TIMM_HOME="$SCRATCH/.cache/timm"
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TIMM_HOME"

# Dataset staging local node
cp "$SCRATCH/nodulocc_dataset.tar" "$SLURM_TMPDIR/"
cd "$SLURM_TMPDIR"
tar -xf nodulocc_dataset.tar
cd "$PROJECT_DIR"

CFG="configs/r1_b4_bce_posw.yaml"
CKPT="artifacts/checkpoints/r1_b4_bce_posw/classification_holdout_best.pt"
OUT="artifacts/threshold_sweeps/r1_seed42"
mkdir -p "$OUT"

srun python -m nodulocc.cli predict \
  --config "$CFG" \
  --ckpt "$CKPT" \
  --split val \
  --out "$OUT/val_probs.csv" \
  --override data.dataset_root="$SLURM_TMPDIR/nodulocc_dataset" \
  --override model.pretrained=false \
  --override tracking.enabled=false

python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nodulocc.config import load_config
from nodulocc.data import prepare_frames
from nodulocc.metrics import classification_metrics

cfg_path = "configs/r1_b4_bce_posw.yaml"
dataset_root = Path(os.environ["SLURM_TMPDIR"]) / "nodulocc_dataset"
out_dir = Path("artifacts/threshold_sweeps/r1_seed42")
out_dir.mkdir(parents=True, exist_ok=True)

cfg = load_config(cfg_path, overrides=[
  f"data.dataset_root={dataset_root}",
  "tracking.enabled=false",
])

_, val_df = prepare_frames(cfg)
if val_df is None:
  raise RuntimeError("Validation split introuvable pour le sweep de threshold.")

pred = pd.read_csv(out_dir / "val_probs.csv")
merged = val_df[["file_name", "label"]].merge(pred[["file_name", "prob_nodule"]], on="file_name", how="inner")

if len(merged) != len(val_df):
  raise RuntimeError(
    f"Mismatch val labels/preds: labels={len(val_df)} preds_merged={len(merged)}"
  )

y_true = merged["label"].to_numpy(dtype=np.int32)
y_prob = merged["prob_nodule"].to_numpy(dtype=np.float32)

rows = []
for t in np.arange(0.05, 1.00, 0.05, dtype=np.float32):
  thr = round(float(t), 2)
  m = classification_metrics(y_true=y_true, y_prob=y_prob, threshold=thr)
  payload = {
    "threshold": thr,
    **{k: float(v) for k, v in m.items()},
  }
  out_file = out_dir / f"t_{thr:.2f}.json"
  out_file.write_text(json.dumps(payload, indent=2))
  rows.append(payload)

best = max(rows, key=lambda x: x["f1"])
print("BEST threshold,f1,auc,precision,recall =", (
  best["threshold"],
  best["f1"],
  best["auc"],
  best["precision"],
  best["recall"],
))
PY
