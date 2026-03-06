# NoduLoCC2026

Projet de détection de nodules pulmonaires sur des images radiographiques thoraciques utilisant un CNN profond avec analyse Grad-CAM.

## Description

Ce projet implémente la **classification binaire** de nodules pulmonaires (Nodule vs No Finding) avec :
- **Exploration des données** : Analyse statistique et visualisation
- **CNN profond** : 6 couches convolutionnelles
- **Grad-CAM** : Visualisation des zones d'attention du modèle
- **Analyse des filtres** : Compréhension des features apprises

## Données

- **Total** : ~63,000 images radiographiques thoraciques
- **Avec nodules** : 2,818 images (~4.5%)
- **Saines** : 60,361 images (~95.5%)
- **Format** : PNG (nih_filtered_images.zip)
- **Labels** : classification_labels.csv

## Utilisation sur Cluster SLURM

### Prérequis
```bash
# Charger les modules
module load python/3.10 cuda/11.8

# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install torch torchvision pandas numpy matplotlib seaborn Pillow scikit-learn opencv-python
```

### Lancer le job

```bash
# Soumettre le job SLURM
sbatch train_job.sh

# Suivre les logs en temps réel
tail -f logs/run_JOBID.out

# Vérifier l'état du job
squeue -u $USER
```

Le job exécutera automatiquement :
1. **exploration.py** - Analyse et visualisation des données
2. **train_simple_cnn.py** - Entraînement avec Grad-CAM

### Notification email

Vous recevrez un email à **leon.morales@utbm.fr** lorsque :
- Le job est terminé avec succès
- Le job échoue avec une erreur

## 📁 Structure du Projet

```
NoduLoCC2026/
├── data/
│   ├── classification_labels.csv          # Labels de classification
│   ├── nih_filtered_images.zip            # Images (ZIP - 24GB)
│   └── localization_labels.csv            # Localisation (non utilisé)
│
├── results/
│   ├── exploration/                       # Résultats de l'exploration
│   │   ├── statistics.txt
│   │   ├── grid_healthy_images.png
│   │   ├── grid_nodules_images.png
│   │   ├── distributions.png
│   │   └── comparison.png
│   │
│   └── training/                          # Résultats de l'entraînement
│       ├── test_results.txt
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       ├── gradcam_analysis.png
│       └── conv_filters.png
│
├── checkpoints/
│   └── best_model.pth
│
├── logs/
│   ├── run_JOBID.out
│   └── run_JOBID.err
│
├── exploration.py
├── train_simple_cnn.py
├── train_job.sh
└── README.md
```

## 🔍 Détails des Scripts

### 1. exploration.py

Analyse et visualisation des données :
- Extraction automatique des images depuis le ZIP
- Échantillonnage de 20 images saines + 20 avec nodules
- Statistiques : dimensions, intensités, distributions
- Génération de 4 visualisations

### 2. train_simple_cnn.py

Entraînement d'un CNN profond :
- **Architecture** : 6 couches convolutionnelles (~2M paramètres)
- Équilibrage automatique (ratio 1:1)
- Split 80/10/10
- Augmentation de données
- Early stopping
- **Grad-CAM** : Heatmaps d'attention
- **Filtres** : Visualisation des features

## ⚙️ Configuration

### Hyperparamètres

```python
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
EARLY_STOPPING_PATIENCE = 7
```

### SLURM

```bash
--time=03:00:00
--gpus-per-node=a100:1
--mem=64G
--cpus-per-task=8
--mail-user=leon.morales@utbm.fr
```

## 📈 Résultats Attendus

- **Accuracy** : 85-92%
- **F1-Score** : 0.85-0.90
- **AUC-ROC** : 0.90-0.95

## 🔧 Optimisations SLURM

- Extraction automatique sur SSD local (SLURM_TMPDIR)
- I/O jusqu'à 100x plus rapide
- GPU batch training optimisé

## 🐛 Dépannage

### "CUDA out of memory"
Réduire BATCH_SIZE à 16 ou 8

### Job SLURM tué
Augmenter --mem ou --time

### Pas d'email reçu
Vérifier l'adresse dans train_job.sh

## 📝 Commandes Utiles

```bash
# Voir mes jobs
squeue -u $USER

# Annuler un job
scancel JOBID

# Détails d'un job
sacct -j JOBID --format=JobID,Elapsed,State

# Copier résultats
scp -r user@cluster:/project/.../results ./
```

## 🎯 Prochaines Étapes

- Tâche de localisation (x, y)
- Multi-task learning
- Autres architectures (ResNet, EfficientNet)
- Ensemble de modèles

## 👤 Auteur

Leon Morales - leon.morales@utbm.fr
