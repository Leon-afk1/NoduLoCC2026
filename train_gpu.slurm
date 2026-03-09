#!/bin/bash
#SBATCH --account=def-zonata
#SBATCH --job-name=nodulocc-train
#SBATCH --time=0-6:00
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=leonard.zipper68@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

module load StdEnv/2023
module load python/3.14.2
cd $SCRATCH/NoduLoCC2026
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HF_HOME="$SCRATCH/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TIMM_HOME="$SCRATCH/.cache/timm"
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TIMM_HOME"

# Important sur Narval: W&B offline est plus sûr.
export WANDB_MODE=offline

# --------- Dataset staging vers le disque local du noeud ----------
DATASET_TAR="$SCRATCH/nodulocc_dataset.tar"

echo "Copying archive to local node..."
cp "$DATASET_TAR" "$SLURM_TMPDIR/"

echo "Extracting locally..."
cd "$SLURM_TMPDIR"
tar -xf nodulocc_dataset.tar

echo "Dataset staged. Starting training..."
# ---------------------------------------------------------------

cd $SCRATCH/NoduLoCC2026
srun python -m nodulocc.cli train --config configs/r3_b4_bce_no_posw.yaml \
  --override data.dataset_root="$SLURM_TMPDIR/nodulocc_dataset"
