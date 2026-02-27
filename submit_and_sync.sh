#!/bin/bash
set -euo pipefail

TRAIN_JOB_ID=$(sbatch train_gpu.slurm | awk '{print $4}')
echo "Submitted training job: $TRAIN_JOB_ID"

echo "Waiting for job to finish..."

while squeue -j "$TRAIN_JOB_ID" 2>/dev/null | grep -q "$TRAIN_JOB_ID"; do
    sleep 30
done

echo "Training finished. Syncing wandb..."

cd $SCRATCH/NoduLoCC2026
source .venv/bin/activate
wandb sync --sync-all

echo "Done."