#!/bin/bash
# Pré-télécharge les poids du modèle sur le nœud de login,
# soumet le job SLURM, puis synchronise wandb une fois terminé.

set -euo pipefail

PROJECT_DIR="/project/def-zonata/leonmls/NoduLoCC2026"
CONFIG="configs/leon_b4_focal_team_v2.yaml"

cd "$PROJECT_DIR"
source venv/bin/activate

export HF_HOME="$SCRATCH/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TIMM_HOME="$SCRATCH/.cache/timm"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TIMM_HOME"

echo "Pré-téléchargement des poids du backbone..."
export CONFIG
python - <<'PY'
import os
from nodulocc.config import load_config
import timm

cfg = load_config(os.environ["CONFIG"])
model_cfg = cfg.get("model", {})
backbone = str(model_cfg.get("backbone", "tiny_cnn"))
pretrained = bool(model_cfg.get("pretrained", True))

if pretrained and backbone != "tiny_cnn":
    timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
    print(f"Poids mis en cache : {backbone}")
else:
    print(f"Pas de pré-téléchargement (backbone={backbone}, pretrained={pretrained})")
PY

echo "Soumission du job SLURM..."
TRAIN_JOB_ID=$(sbatch train_leon.slurm | awk '{print $4}')
echo "Job soumis : $TRAIN_JOB_ID"

echo "En attente de la fin du job..."
while squeue -j "$TRAIN_JOB_ID" 2>/dev/null | grep -q "$TRAIN_JOB_ID"; do
    sleep 30
done

echo "Job terminé. Synchronisation wandb..."
wandb sync --sync-all
