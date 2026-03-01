#!/bin/bash
set -euo pipefail

cd "$SCRATCH/NoduLoCC2026"
source .venv/bin/activate

CONFIG_PATH="configs/classification.yaml"

# Shared cache on scratch so login + GPU nodes see the same files.
export HF_HOME="$SCRATCH/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TIMM_HOME="$SCRATCH/.cache/timm"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TIMM_HOME"

echo "Prefetching pretrained backbone (if enabled) from config: $CONFIG_PATH"
python - <<'PY'
from nodulocc.config import load_config
import timm

cfg = load_config("configs/classification.yaml")
model_cfg = cfg.get("model", {})
backbone = str(model_cfg.get("backbone", "tiny_cnn"))
pretrained = bool(model_cfg.get("pretrained", True))

if pretrained and backbone != "tiny_cnn":
    timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
    print(f"Cached pretrained weights for: {backbone}")
else:
    print(f"Skipping prefetch (backbone={backbone}, pretrained={pretrained})")
PY

TRAIN_JOB_ID=$(sbatch train_gpu.slurm | awk '{print $4}')
echo "Submitted training job: $TRAIN_JOB_ID"

echo "Waiting for job to finish..."

while squeue -j "$TRAIN_JOB_ID" 2>/dev/null | grep -q "$TRAIN_JOB_ID"; do
    sleep 30
done

echo "Training finished. Syncing wandb..."

wandb sync --sync-all

echo "Done."