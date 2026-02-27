# Tutoriel Narval (Calcul Québec) avec Slurm

Ce guide explique comment installer ce projet et lancer les entraînements sur Narval.

## 1) Pré-requis

- Un compte Alliance/Calcul Québec actif.
- Un compte/projet Slurm (`--account=...`, ex: `def-monpi`).
- Connexion SSH vers Narval.

## 2) Où mettre les fichiers

Recommandation pratique:
- code + checkpoints temporaires: `$SCRATCH`
- résultats finaux à conserver: `$PROJECT` (ou autre stockage persistant)
- dataset: idéalement dans `$SCRATCH` (I/O plus rapide), ou en lecture depuis `$PROJECT`

Exemple:
```bash
mkdir -p $SCRATCH/nodulocc
cd $SCRATCH/nodulocc
```

Si ton dataset est à côté du repo localement, copie-le aussi sur Narval, par exemple:
```bash
rsync -av ~/AI/nodulocc_dataset $SCRATCH/
```

## 3) Préparer l’environnement Python (sur login node)

### Option A (si `uv` disponible)
```bash
cd $SCRATCH/nodulocc/<repo>
module load StdEnv/2023 python
uv sync --extra exp
```

### Option B (fallback standard venv + pip)
```bash
cd $SCRATCH/nodulocc/<repo>
module load StdEnv/2023 python
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[exp]
```

## 4) Vérification rapide avant batch

```bash
cd $SCRATCH/nodulocc/<repo>
uv run python -m nodulocc.cli --help
```

Si tu n'utilises pas `uv`, utilise:
```bash
cd $SCRATCH/nodulocc/<repo>
source .venv/bin/activate
python -m nodulocc.cli --help
```

## 5) Script Slurm GPU (recommandé)

Crée `train_gpu.slurm`:

```bash
#!/bin/bash
#SBATCH --account=def-xxx
#SBATCH --job-name=nodulocc-train
#SBATCH --time=0-06:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail

module load StdEnv/2023
module load python/3.11  # exemple : adapte à ce que "module avail python" te montre
cd $SCRATCH/nodulocc/<repo>
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Important sur Narval: W&B offline est plus sûr.
export WANDB_MODE=offline

srun python -m nodulocc.cli train --config configs/classification.yaml \
  --override data.dataset_root=$SCRATCH/nodulocc_dataset
```

Soumettre:
```bash
sbatch train_gpu.slurm
```

## 6) Commandes utiles Slurm

```bash
squeue -u $USER
sq
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
scancel <JOBID>
```

## 7) K-fold et entraînement final

### K-fold
```bash
python -m nodulocc.cli train --config configs/classification.yaml \
  --override data.dataset_root=$SCRATCH/nodulocc_dataset \
  --override validation.mode=kfold \
  --override validation.k=5
```

### Entraînement final sur 100% des données
```bash
python -m nodulocc.cli train --config configs/classification.yaml \
  --override data.dataset_root=$SCRATCH/nodulocc_dataset \
  --override train.full_data=true
```

## 8) W&B sur Narval (important)

D’après la doc Alliance, sur Narval il est recommandé d’utiliser `wandb` en mode offline pendant les jobs compute.

Dans le job:
```bash
export WANDB_MODE=offline
```

Après le job (login node), synchroniser:
```bash
cd $SCRATCH/nodulocc/<repo>
source .venv/bin/activate
wandb sync wandb/offline-run-*
```

## 9) Bonnes pratiques HPC

- Demander des ressources réalistes (`time`, `mem`, `cpus`, `gpu`) pour réduire la file d’attente.
- Écrire checkpoints/logs dans `$SCRATCH` pendant le run.
- Copier les résultats finaux importants vers `$PROJECT` en fin de campagne.
- Versionner la config de run (`configs/*.yaml` + overrides) avec les checkpoints.

## 10) Sources (à vérifier régulièrement)

- Alliance Doc: W&B sur clusters (Narval inclus) et mode offline.
- Alliance Doc: principes Slurm/job scripts.

Les politiques cluster, modules et quotas peuvent évoluer: vérifier la doc Alliance/Calcul Québec avant une grosse campagne.
