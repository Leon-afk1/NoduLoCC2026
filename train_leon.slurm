#!/bin/bash
#SBATCH --account=def-zonata
#SBATCH --job-name=leon-train
#SBATCH --time=0-12:00
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=leon.morales@utbm.fr
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_DIR="/project/def-zonata/leonmls/NoduLoCC2026"
DATA_DIR="$PROJECT_DIR/data"

# opencv doit être chargé avant d'activer le venv
module load StdEnv/2023
module load gcc opencv/4.8.1

# Expose the OpenCV Python bindings from the module into the venv
# opencv/4.8.1 is the only version with Python 3.10 bindings on Narval
export PYTHONPATH="$EBROOTOPENCV/lib/python3.10/site-packages:${PYTHONPATH:-}"

source "$PROJECT_DIR/venv/bin/activate"

# Validate cv2 is importable
python -c "import cv2; print('cv2 OK:', cv2.__version__)" || {
  echo "ERROR: cv2 not importable. EBROOTOPENCV=$EBROOTOPENCV"
  echo "PYTHONPATH=$PYTHONPATH"
  ls "$EBROOTOPENCV/lib/python3.10/site-packages/" 2>/dev/null || echo "Path not found"
  exit 1
}
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export BLIS_NUM_THREADS=1       # évite le crash libblis OpenMP sur Narval
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HF_HOME="$SCRATCH/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TIMM_HOME="$SCRATCH/.cache/timm"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TIMM_HOME"

export WANDB_MODE=offline

# ---- Staging du dataset vers le SSD local du noeud ----
DATASET_ROOT="$SLURM_TMPDIR/nodulocc_dataset"
mkdir -p "$DATASET_ROOT"

echo "Copie des labels..."
cp "$DATA_DIR/classification_labels.csv" "$DATASET_ROOT/"

echo "Extraction NIH images..."
unzip -q "$DATA_DIR/nih_filtered_images.zip" -d "$DATASET_ROOT/"

echo "Extraction LIDC images..."
unzip -q "$DATA_DIR/lidc_png_16_bit.zip" -d "$DATASET_ROOT/"

echo "Dataset prêt dans $DATASET_ROOT"
echo "Contenu : $(ls $DATASET_ROOT)"
# -------------------------------------------------------

echo "Démarrage de l'entraînement..."
srun python -m nodulocc.cli train \
  --config configs/leon_xrv_nih_focal.yaml \
  --override data.dataset_root="$DATASET_ROOT"
