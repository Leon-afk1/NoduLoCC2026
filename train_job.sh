#!/bin/bash
#SBATCH --job-name=nodule_efficientnet
#SBATCH --account=def-zonata
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err
#SBATCH --mail-user=leon.morales@utbm.fr
#SBATCH --mail-type=END,FAIL


echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# Charger les modules (AVANT d'activer le venv pour OpenCV)
module load python/3.10
module load gcc opencv/4.10.0

# Aller dans le dossier du projet (AVANT de définir les variables)
PROJECT_DIR="/project/def-zonata/leonmls/NoduLoCC2026"
cd $PROJECT_DIR

# Variables d'environnement
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Configuration Wandb (offline mode pour Calcul Quebec)
export WANDB_MODE=offline
export WANDB_DIR=$PROJECT_DIR
export WANDB_DATA_DIR=$PROJECT_DIR/wandb_data
export WANDB_CACHE_DIR=$SLURM_TMPDIR/wandb_cache

# Activer l'environnement virtuel
source venv/bin/activate

# Générer un ID unique pour la run wandb partagée.
# Exploration + Entraînement partageront la MEME run wandb.
export WANDB_SHARED_RUN_NAME="job_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
export WANDB_SHARED_RUN_ID=$(python -c "import random,string; print(''.join(random.choices(string.ascii_lowercase+string.digits, k=8)))")
echo ""
echo "Run wandb partagee: $WANDB_SHARED_RUN_NAME (ID: $WANDB_SHARED_RUN_ID)"
echo "  Exploration + Entrainement → meme run wandb"
echo ""

# Créer les dossiers nécessaires
mkdir -p logs
mkdir -p results/training
mkdir -p results/exploration
mkdir -p checkpoints
mkdir -p wandb_data

# Afficher les informations système
echo ""
echo "Configuration Système"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""
echo "Dossier temporaire SLURM: $SLURM_TMPDIR"
echo "Wandb mode: $WANDB_MODE"
echo ""

# EXPLORATION DES DONNÉES

echo "="
echo "EXPLORATION DES DONNEES"
echo "="
python exploration.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'exploration"
    exit 1
fi
echo "Exploration terminee"

# ENTRAÎNEMENT AVEC TRANSFER LEARNING (EFFICIENTNET-B4)

echo ""
echo "ENTRAINEMENT - EFFICIENTNET-B4 TRANSFER LEARNING"
python train_transfer_learning.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'entraînement"
    exit 1
fi
echo "Entraînement terminé"

# SYNCHRONISATION WANDB

echo ""
echo "="
echo "SYNCHRONISATION WANDB"
echo "="
echo "NOTE: La synchronisation wandb ne peut pas se faire depuis le noeud de calcul (pas d'internet)"
echo "Les runs sont sauvegardees dans: wandb/"
echo ""
echo "Pour synchroniser manuellement depuis le noeud de login:"
echo "  cd /project/def-zonata/leonmls/NoduLoCC2026"
echo "  source venv/bin/activate"
echo "  wandb sync wandb/offline-run-*"
echo ""

# FIN DU JOB

echo ""
echo "Job terminé avec succès !"
echo "End: $(date)"
echo "Notification envoyée à leon.morales@utbm.fr"
echo ""
echo "Résultats disponibles dans:"
echo "   - results/exploration/ (exploration des données)"
echo "   - results/training/ (résultats d'entraînement)"
echo "   - checkpoints/ (modèles sauvegardés)"
echo "   - wandb/ (logs wandb)"
echo "   - logs/run_$SLURM_JOB_ID.out (logs détaillés)"
