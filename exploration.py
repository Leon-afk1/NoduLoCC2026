
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import zipfile
import os
import time
from datetime import datetime
import wandb
from wandb_utils import setup_wandb, close_wandb, get_project_name

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


PROJECT_DIR = "/project/def-zonata/leonmls/NoduLoCC2026"
CSV_PATH = os.path.join(PROJECT_DIR, "data/classification_labels.csv")
ZIP_PATH = os.path.join(PROJECT_DIR, "data/nih_filtered_images.zip")
OUTPUT_DIR = Path("results/exploration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nombre d'images à analyser
N_HEALTHY_SAMPLES = 20
N_NODULE_SAMPLES = 20

print(f"Debut: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# mode offline sur nœud de calcul
offline_mode = os.environ.get('SLURM_TMPDIR') is not None

setup_wandb(
    project_name=get_project_name(),
    run_name=f"data_exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "n_healthy_samples": N_HEALTHY_SAMPLES,
        "n_nodule_samples": N_NODULE_SAMPLES
    },
    job_type="data_exploration",
    offline_mode=offline_mode
)


print(f"\nChargement du CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"   {len(df)} images chargees")

wandb.log({"exploration/total_images": len(df)})

# Filtrer pour ne garder que les images NIH (format 8 chiffres_3 chiffres.png)
# Les images LIDC (format 0026.png) ne sont pas dans le ZIP
df = df[df['file_name'].str.match(r'^\d{8}_\d{3}\.png$')].copy()
print(f"   {len(df)} images NIH (dans le ZIP)")

# Séparer les classes
df_healthy = df[df['label'] == 'No Finding']
df_nodules = df[df['label'] == 'Nodule']

print(f"\nDistribution des classes:")
print(f"   - No Finding (Sain): {len(df_healthy)} images ({len(df_healthy)/len(df)*100:.2f}%)")
print(f"   - Nodule:            {len(df_nodules)} images ({len(df_nodules)/len(df)*100:.2f}%)")
print(f"   - Ratio desequilibre: 1:{len(df_healthy)/len(df_nodules):.1f}")

wandb.log({
    "exploration/healthy_count": len(df_healthy),
    "exploration/nodule_count": len(df_nodules),
    "exploration/healthy_percentage": len(df_healthy)/len(df)*100,
    "exploration/nodule_percentage": len(df_nodules)/len(df)*100,
    "exploration/imbalance_ratio": len(df_healthy)/len(df_nodules)
})

# Sauvegarder les statistiques globales
stats_file = OUTPUT_DIR / "statistics.txt"
with open(stats_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("STATISTIQUES GLOBALES DES DONNÉES\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Total d'images: {len(df)}\n")
    f.write(f"Images saines (No Finding): {len(df_healthy)} ({len(df_healthy)/len(df)*100:.2f}%)\n")
    f.write(f"Images avec nodules: {len(df_nodules)} ({len(df_nodules)/len(df)*100:.2f}%)\n")
    f.write(f"Ratio de déséquilibre: 1:{len(df_healthy)/len(df_nodules):.1f}\n\n")
    
print(f"  Statistiques sauvegardées: {stats_file}")


# Verifier si on est sur un noeud de calcul (SLURM_TMPDIR)
slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
if slurm_tmpdir:
    print(f"\nNoeud de calcul detecte: {slurm_tmpdir}")
    print("   Extraction du ZIP vers le SSD local...")
    
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(slurm_tmpdir)
    
    extraction_time = time.time() - t0
    print(f"   Extraction terminee en {extraction_time:.1f}s")
    IMAGE_DIR = os.path.join(slurm_tmpdir, "nih_filtered_images")
else:
    print("\nMode local detecte (pas de SLURM_TMPDIR)")
    print("   Tentative d'extraction locale...")
    IMAGE_DIR = os.path.join(PROJECT_DIR, "data/temp_images")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        # Extraire seulement les images nécessaires pour l'exploration
        sample_files = list(df_healthy.head(N_HEALTHY_SAMPLES)['file_name']) + \
                      list(df_nodules.head(N_NODULE_SAMPLES)['file_name'])
        for file_name in sample_files:
            try:
                zip_ref.extract(file_name, IMAGE_DIR)
            except:
                pass
    extraction_time = time.time() - t0
    print(f"   Extraction partielle terminee en {extraction_time:.1f}s")

print(f"\nDossier d'images: {IMAGE_DIR}")


# Selectionner aleatoirement des images
np.random.seed(42)
sample_healthy = df_healthy.sample(n=min(N_HEALTHY_SAMPLES, len(df_healthy)), random_state=42)
sample_nodules = df_nodules.sample(n=min(N_NODULE_SAMPLES, len(df_nodules)), random_state=42)

print(f"\nEchantillons selectionnes:")
print(f"   - {len(sample_healthy)} images saines")
print(f"   - {len(sample_nodules)} images avec nodules")


def load_image(file_name, image_dir):
    img_path = os.path.join(image_dir, file_name)
    if not os.path.exists(img_path):
        # Essayer sans sous-dossier
        for root, dirs, files in os.walk(image_dir):
            if file_name in files:
                img_path = os.path.join(root, file_name)
                break
    try:
        return Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"   Erreur chargement {file_name}: {e}")
        return None

print("\nAnalyse des dimensions et statistiques...")

# Analyser les images échantillonnées
image_stats = {
    'healthy': {'widths': [], 'heights': [], 'means': [], 'stds': [], 'images': []},
    'nodules': {'widths': [], 'heights': [], 'means': [], 'stds': [], 'images': []}
}

# Images saines
print(f"\n   Chargement des {len(sample_healthy)} images saines...")
for idx, row in sample_healthy.iterrows():
    img = load_image(row['file_name'], IMAGE_DIR)
    if img is not None:
        image_stats['healthy']['images'].append(img)
        image_stats['healthy']['widths'].append(img.size[0])
        image_stats['healthy']['heights'].append(img.size[1])
        img_array = np.array(img)
        image_stats['healthy']['means'].append(img_array.mean())
        image_stats['healthy']['stds'].append(img_array.std())

# Images avec nodules
print(f"   Chargement des {len(sample_nodules)} images avec nodules...")
for idx, row in sample_nodules.iterrows():
    img = load_image(row['file_name'], IMAGE_DIR)
    if img is not None:
        image_stats['nodules']['images'].append(img)
        image_stats['nodules']['widths'].append(img.size[0])
        image_stats['nodules']['heights'].append(img.size[1])
        img_array = np.array(img)
        image_stats['nodules']['means'].append(img_array.mean())
        image_stats['nodules']['stds'].append(img_array.std())

print(f"\n   {len(image_stats['healthy']['images'])} images saines chargees")
print(f"   {len(image_stats['nodules']['images'])} images avec nodules chargees")

# Statistiques detaillees
print(f"\nStatistiques des dimensions:")
print(f"\n   Images saines:")
print(f"      Largeur:  {np.mean(image_stats['healthy']['widths']):.0f} ± {np.std(image_stats['healthy']['widths']):.0f} pixels")
print(f"      Hauteur:  {np.mean(image_stats['healthy']['heights']):.0f} ± {np.std(image_stats['healthy']['heights']):.0f} pixels")
print(f"      Intensité moyenne: {np.mean(image_stats['healthy']['means']):.1f} ± {np.std(image_stats['healthy']['means']):.1f}")
print(f"\n   Images avec nodules:")
print(f"      Largeur:  {np.mean(image_stats['nodules']['widths']):.0f} ± {np.std(image_stats['nodules']['widths']):.0f} pixels")
print(f"      Hauteur:  {np.mean(image_stats['nodules']['heights']):.0f} ± {np.std(image_stats['nodules']['heights']):.0f} pixels")
print(f"      Intensité moyenne: {np.mean(image_stats['nodules']['means']):.1f} ± {np.std(image_stats['nodules']['means']):.1f}")

# Ajouter aux statistiques
with open(stats_file, 'a') as f:
    f.write("STATISTIQUES DES IMAGES ÉCHANTILLONNÉES\n")
    f.write(f"Images saines analysées: {len(image_stats['healthy']['images'])}\n")
    f.write(f"  Dimensions moyennes: {np.mean(image_stats['healthy']['widths']):.0f} x {np.mean(image_stats['healthy']['heights']):.0f} pixels\n")
    f.write(f"  Intensité moyenne: {np.mean(image_stats['healthy']['means']):.1f} ± {np.std(image_stats['healthy']['means']):.1f}\n\n")
    f.write(f"Images avec nodules analysées: {len(image_stats['nodules']['images'])}\n")
    f.write(f"  Dimensions moyennes: {np.mean(image_stats['nodules']['widths']):.0f} x {np.mean(image_stats['nodules']['heights']):.0f} pixels\n")
    f.write(f"  Intensité moyenne: {np.mean(image_stats['nodules']['means']):.1f} ± {np.std(image_stats['nodules']['means']):.1f}\n\n")


print("\nCreation de la grille d'images echantillons")

# Grille pour images saines
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('Échantillons d\'images SAINES (No Finding)', fontsize=20, fontweight='bold', y=0.995)
for idx, (ax, img) in enumerate(zip(axes.flat, image_stats['healthy']['images'])):
    if img is not None:
        ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
        ax.set_title(f'Image {idx+1}', fontsize=10)
        ax.axis('off')
plt.tight_layout()
healthy_grid_path = OUTPUT_DIR / "grid_healthy_images.png"
plt.savefig(healthy_grid_path, dpi=150, bbox_inches='tight')
wandb.log({"exploration/healthy_samples": wandb.Image(str(healthy_grid_path))})
plt.close()
print(f"   Grille images saines: {healthy_grid_path}")

# Grille pour images avec nodules
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('Échantillons d\'images avec NODULES', fontsize=20, fontweight='bold', y=0.995)
for idx, (ax, img) in enumerate(zip(axes.flat, image_stats['nodules']['images'])):
    if img is not None:
        ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
        ax.set_title(f'Image {idx+1}', fontsize=10)
        ax.axis('off')
plt.tight_layout()
nodules_grid_path = OUTPUT_DIR / "grid_nodules_images.png"
plt.savefig(nodules_grid_path, dpi=150, bbox_inches='tight')
wandb.log({"exploration/nodule_samples": wandb.Image(str(nodules_grid_path))})
plt.close()
print(f"   Grille images nodules: {nodules_grid_path}")


print("\nCreation des graphiques de distribution...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribution des largeurs
axes[0, 0].hist(image_stats['healthy']['widths'], bins=20, alpha=0.6, label='Saines', color='green')
axes[0, 0].hist(image_stats['nodules']['widths'], bins=20, alpha=0.6, label='Nodules', color='red')
axes[0, 0].set_xlabel('Largeur (pixels)', fontsize=12)
axes[0, 0].set_ylabel('Fréquence', fontsize=12)
axes[0, 0].set_title('Distribution des largeurs d\'images', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Distribution des hauteurs
axes[0, 1].hist(image_stats['healthy']['heights'], bins=20, alpha=0.6, label='Saines', color='green')
axes[0, 1].hist(image_stats['nodules']['heights'], bins=20, alpha=0.6, label='Nodules', color='red')
axes[0, 1].set_xlabel('Hauteur (pixels)', fontsize=12)
axes[0, 1].set_ylabel('Fréquence', fontsize=12)
axes[0, 1].set_title('Distribution des hauteurs d\'images', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Distribution des intensités moyennes
axes[1, 0].hist(image_stats['healthy']['means'], bins=30, alpha=0.6, label='Saines', color='green', density=True)
axes[1, 0].hist(image_stats['nodules']['means'], bins=30, alpha=0.6, label='Nodules', color='red', density=True)
axes[1, 0].set_xlabel('Intensité moyenne', fontsize=12)
axes[1, 0].set_ylabel('Densité', fontsize=12)
axes[1, 0].set_title('Distribution des intensités moyennes', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Distribution des écarts-types
axes[1, 1].hist(image_stats['healthy']['stds'], bins=30, alpha=0.6, label='Saines', color='green', density=True)
axes[1, 1].hist(image_stats['nodules']['stds'], bins=30, alpha=0.6, label='Nodules', color='red', density=True)
axes[1, 1].set_xlabel('Écart-type de l\'intensité', fontsize=12)
axes[1, 1].set_ylabel('Densité', fontsize=12)
axes[1, 1].set_title('Distribution des écarts-types d\'intensité', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
distributions_path = OUTPUT_DIR / "distributions.png"
plt.savefig(distributions_path, dpi=150, bbox_inches='tight')
wandb.log({"exploration/distributions": wandb.Image(str(distributions_path))})
plt.close()
print(f"   Distributions: {distributions_path}")


print("\nCreation du graphique de comparaison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot des intensités
data_to_plot = [image_stats['healthy']['means'], image_stats['nodules']['means']]
bp = axes[0].boxplot(data_to_plot, tick_labels=['Saines', 'Nodules'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
axes[0].set_ylabel('Intensité moyenne', fontsize=12)
axes[0].set_title('Comparaison des intensités moyennes', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Pie chart de la distribution
sizes = [len(df_healthy), len(df_nodules)]
labels = [f'Saines\n({len(df_healthy)} images)', f'Nodules\n({len(df_nodules)} images)']
colors = ['lightgreen', 'lightcoral']
explode = (0.05, 0.05)
axes[1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Distribution globale des classes', fontsize=14, fontweight='bold')

plt.tight_layout()
comparison_path = OUTPUT_DIR / "comparison.png"
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
wandb.log({"exploration/comparison": wandb.Image(str(comparison_path))})
plt.close()
print(f"   Comparaison: {comparison_path}")


print(f"\nTous les resultats sont sauvegardes dans: {OUTPUT_DIR}")

print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Close wandb run
close_wandb()
