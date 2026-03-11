#!/bin/bash
#SBATCH --account=def-zonata
#SBATCH --job-name=nodu-ensemble
#SBATCH --time=0-12:00
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=leonard.zipper68@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

module load StdEnv/2023
module load python/3.14.2
module load gcc opencv/4.13.0

PROJECT_DIR="$SCRATCH/NoduLoCC2026"
cd "$PROJECT_DIR"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE=offline
export HF_HOME="$SCRATCH/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TIMM_HOME="$SCRATCH/.cache/timm"
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TIMM_HOME"

python - <<'PY'
try:
    import cv2  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "cv2 is not available in this job environment. "
        "Load the same OpenCV module used when creating the venv before "
        "activating .venv. Original error: "
        + repr(exc)
    )
PY

DATASET_TAR="$SCRATCH/nodulocc_dataset.tar"
echo "Copying archive to local node..."
cp "$DATASET_TAR" "$SLURM_TMPDIR/"

echo "Extracting locally..."
cd "$SLURM_TMPDIR"
tar -xf nodulocc_dataset.tar

echo "Dataset staged. Starting prediction + ensemble..."
cd "$PROJECT_DIR"

CFG_A="configs/r1_b4_bce_posw.yaml"
CKPT_A="artifacts/checkpoints/r1_b4_bce_posw/classification_holdout_best.pt"
CFG_B="configs/r1_b4_bce_posw_team_v2_no_double_comp.yaml"
CKPT_B="artifacts/checkpoints/r1_b4_bce_posw_team_v2_no_double_comp/classification_holdout_best.pt"

OUT_DIR="artifacts/ensemble_b4_fast"
PRED_DIR="$OUT_DIR/preds"
PRED_A="$PRED_DIR/r1_b4_bce_posw_val.csv"
PRED_B="$PRED_DIR/r1_b4_teamv2_val.csv"
LABELS_CSV="$SLURM_TMPDIR/nodulocc_dataset/classification_labels.csv"

mkdir -p "$PRED_DIR" "$OUT_DIR"

COMMON_OVERRIDES=(
  --override "data.dataset_root=$SLURM_TMPDIR/nodulocc_dataset"
  --override "data.classification.lidc_train_only=true"
  --override "tracking.enabled=false"
)

if [ ! -f "$PRED_A" ]; then
  echo "Predicting model A -> $PRED_A"
  srun python -m nodulocc.cli predict \
    --config "$CFG_A" \
    --ckpt "$CKPT_A" \
    --split val \
    --out "$PRED_A" \
    "${COMMON_OVERRIDES[@]}"
else
  echo "Skipping model A prediction, file already exists: $PRED_A"
fi

if [ ! -f "$PRED_B" ]; then
  echo "Predicting model B -> $PRED_B"
  srun python -m nodulocc.cli predict \
    --config "$CFG_B" \
    --ckpt "$CKPT_B" \
    --split val \
    --out "$PRED_B" \
    "${COMMON_OVERRIDES[@]}"
else
  echo "Skipping model B prediction, file already exists: $PRED_B"
fi

echo "Running fast CSV ensemble..."
python -m nodulocc.cli ensemble-csv \
  --pred-a "$PRED_A" \
  --pred-b "$PRED_B" \
  --config-ref "$CFG_A" \
  --labels-csv "$LABELS_CSV" \
  --out-dir "$OUT_DIR" \
  --split val \
  --weight-a 0.5 \
  --threshold-a 0.65 \
  --threshold-b 0.65 \
  --default-threshold 0.65 \
  --override-ref "data.dataset_root=$SLURM_TMPDIR/nodulocc_dataset" \
  --override-ref "data.classification.lidc_train_only=true"

echo "Done. Summary: $OUT_DIR/summary.json"
