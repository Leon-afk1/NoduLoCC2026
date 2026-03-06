
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

# Configuration des threads AVANT d'importer les bibliothèques numériques
# Cela évite les conflits BLIS/OpenMP lors des visualisations
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour éviter les problèmes de threading
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import cv2
import wandb
from wandb_utils import setup_wandb, close_wandb, get_project_name

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')


# Chemins
PROJECT_DIR = "/project/def-zonata/leonmls/NoduLoCC2026"
CSV_PATH = os.path.join(PROJECT_DIR, "data/classification_labels.csv")
ZIP_PATH = os.path.join(PROJECT_DIR, "data/nih_filtered_images.zip")
OUTPUT_DIR = Path("results/training_transfer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparamètres
BATCH_SIZE = 24  # Réduit pour image 512×512
EPOCHS = 500
LR_HEAD = 5e-4       # LR pour la tête de classification
LR_BACKBONE = 1e-5   # LR faible pour le backbone (fine-tuning)
IMAGE_SIZE = 512  # Plus grand pour détecter les petits nodules
NUM_WORKERS = 4  # Réduit pour éviter conflits de threads
EARLY_STOPPING_PATIENCE = 30  # Plus large pour laisser le modèle converger
MIN_EPOCHS = 25      # Epochs minimum avant d'activer l'early stopping
EARLY_STOP_METRIC = 'val_f1'  # Metric surveillé (F1 > loss sur dataset déséquilibré)
WEIGHT_DECAY = 1e-4

# Preprocessing CLAHE
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8


USE_ALL_DATA = True  # Pas d'équilibrage, toutes les images
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Transfer Learning
FREEZE_BACKBONE = False  # Fine-tune tout le modèle
MODEL_NAME = "efficientnet_b4"
USE_CBAM = True         # Attention spatiale + par canal (meilleur pour petits objets)
FOCAL_ALPHA = 0.25      # Focal Loss alpha (poids classe positive)
FOCAL_GAMMA = 2.0       # Focal Loss gamma (focus sur exemples difficiles)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Debut: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


offline_mode = os.environ.get('SLURM_TMPDIR') is not None

setup_wandb(
    project_name=get_project_name(),
    run_name=f"efficientnet_b4_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr_backbone": LR_BACKBONE,
        "lr_head": LR_HEAD,
        "image_size": IMAGE_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stop_metric": EARLY_STOP_METRIC,
        "min_epochs": MIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "use_clahe": USE_CLAHE,
        "use_all_data": USE_ALL_DATA,
        "random_seed": RANDOM_SEED,
        "model": MODEL_NAME,
        "use_cbam": USE_CBAM,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "freeze_backbone": FREEZE_BACKBONE,
        "architecture": "EfficientNet-B4 + CBAM + FocalLoss + WeightedSampler"
    },
    job_type="train",
    offline_mode=offline_mode
)


print(f"\nChargement du CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"   {len(df)} images chargees")

# Filtrer images NIH
df = df[df['file_name'].str.match(r'^\d{8}_\d{3}\.png$')].copy()
print(f"   {len(df)} images NIH (dans le ZIP)")

# Add dataset source
df['dataset_source'] = 'NIH'

wandb.log({"dataset/total_images": len(df)})

# Séparer les classes
df_nodules = df[df['label'] == 'Nodule'].copy()
df_healthy = df[df['label'] == 'No Finding'].copy()

print(f"\nDistribution originale:")
print(f"   - Nodules: {len(df_nodules)} ({len(df_nodules)/len(df)*100:.1f}%)")
print(f"   - Saines:  {len(df_healthy)} ({len(df_healthy)/len(df)*100:.1f}%)")
print(f"   - Ratio desequilibre: 1:{len(df_healthy)/len(df_nodules):.1f}")

wandb.log({
    "dataset/nodules": len(df_nodules),
    "dataset/healthy": len(df_healthy),
    "dataset/nodule_percentage": len(df_nodules)/len(df)*100,
    "dataset/imbalance_ratio": len(df_healthy)/len(df_nodules)
})

# On garde TOUTES les images (pas d'équilibrage)
print(f"\nUtilisation du dataset COMPLET (pas d'equilibrage)")
print(f"   Dataset final: {len(df)} images")

# Shuffle
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Split par classe pour avoir des proportions similaires
def stratified_split(df, train_r, val_r, test_r, label_col='label'):
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for label in df[label_col].unique():
        df_label = df[df[label_col] == label]
        n = len(df_label)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        
        train_dfs.append(df_label.iloc[:n_train])
        val_dfs.append(df_label.iloc[n_train:n_train+n_val])
        test_dfs.append(df_label.iloc[n_train+n_val:])
    
    train = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    val = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    test = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    
    return train, val, test

train_df, val_df, test_df = stratified_split(df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

train_nod = len(train_df[train_df['label'] == 'Nodule'])
train_hea = len(train_df[train_df['label'] == 'No Finding'])
val_nod = len(val_df[val_df['label'] == 'Nodule'])
val_hea = len(val_df[val_df['label'] == 'No Finding'])
test_nod = len(test_df[test_df['label'] == 'Nodule'])
test_hea = len(test_df[test_df['label'] == 'No Finding'])

print(f"\nSplits ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}):")
print(f"   - Train: {len(train_df)} images (Nodules: {train_nod} | Saines: {train_hea})")
print(f"   - Val:   {len(val_df)} images (Nodules: {val_nod} | Saines: {val_hea})")
print(f"   - Test:  {len(test_df)} images (Nodules: {test_nod} | Saines: {test_hea})")

wandb.log({
    "split/train_total": len(train_df),
    "split/train_nodules": train_nod,
    "split/train_healthy": train_hea,
    "split/val_total": len(val_df),
    "split/val_nodules": val_nod,
    "split/val_healthy": val_hea,
    "split/test_total": len(test_df),
    "split/test_nodules": test_nod,
    "split/test_healthy": test_hea
})


slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
if slurm_tmpdir:
    print(f"\nNoeud de calcul detecte: {slurm_tmpdir}")
    print("   Extraction du ZIP vers le SSD local...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(slurm_tmpdir)
    print(f"   Extraction terminee en {time.time() - t0:.1f}s")
    IMAGE_DIR = os.path.join(slurm_tmpdir, "nih_filtered_images")
else:
    print("\nMode local detecte")
    IMAGE_DIR = os.path.join(PROJECT_DIR, "data/temp_images")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"   Extraction vers: {IMAGE_DIR}")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(IMAGE_DIR)
    print(f"   Extraction terminee en {time.time() - t0:.1f}s")

print(f"\nDossier d'images: {IMAGE_DIR}")


class NoduleDataset(Dataset):
    
    def __init__(self, dataframe, image_dir, transform=None, use_clahe=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = {'No Finding': 0, 'Nodule': 1}
        self.use_clahe = use_clahe
        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, 
                                         tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['file_name']
        
        # Chercher l'image
        img_path = os.path.join(self.image_dir, img_name)
        if not os.path.exists(img_path):
            for root, dirs, files in os.walk(self.image_dir):
                if img_name in files:
                    img_path = os.path.join(root, img_name)
                    break
        
        try:
            image = Image.open(img_path).convert('L')  # Grayscale
            
            # Apply CLAHE
            if self.use_clahe:
                img_array = np.array(image)
                img_array = self.clahe.apply(img_array)
                image = Image.fromarray(img_array)
            
            # Convert to RGB for pretrained models
            image = image.convert('RGB')
            label = self.label_map[row['label']]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_name
        except Exception as e:
            print(f"Erreur chargement {img_name}: {e}")
            dummy_img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return dummy_img, 0, img_name

# ImageNet normalization (pour transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print(f"\nNormalisation ImageNet (transfer learning):")
print(f"   Mean: {IMAGENET_MEAN}")
print(f"   Std:  {IMAGENET_STD}")

# Transformations — augmentations agressives pour mieux détecter les petits nodules
train_transform = transforms.Compose([
    transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),  # Légèrement plus grand, puis crop
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.3),  # Simule variabilité radiographique
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.05))  # Simule petites occlusions
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Créer les datasets
train_dataset = NoduleDataset(train_df, IMAGE_DIR, transform=train_transform, use_clahe=USE_CLAHE)
val_dataset = NoduleDataset(val_df, IMAGE_DIR, transform=val_transform, use_clahe=USE_CLAHE)
test_dataset = NoduleDataset(test_df, IMAGE_DIR, transform=val_transform, use_clahe=USE_CLAHE)

# WeightedRandomSampler: chaque batch contient 50% nodules / 50% sains
# Le modèle voit les nodules beaucoup plus souvent → améliore considérablement le recall
train_labels_for_sampler = train_df['label'].map({'No Finding': 0, 'Nodule': 1}).values
class_sample_counts = np.bincount(train_labels_for_sampler)
sample_weights = np.where(train_labels_for_sampler == 1,
                          1.0 / class_sample_counts[1],
                          1.0 / class_sample_counts[0])
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)
print(f"\nWeightedRandomSampler:")
print(f"   Poids classe 0 (Sain):  {1.0/class_sample_counts[0]:.6f}")
print(f"   Poids classe 1 (Nodule): {1.0/class_sample_counts[1]:.6f}")

# Créer les dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f"\nDatasets crees:")
print(f"   - Train: {len(train_dataset)} images, {len(train_loader)} batches")
print(f"   - Val:   {len(val_dataset)} images, {len(val_loader)} batches")
print(f"   - Test:  {len(test_dataset)} images, {len(test_loader)} batches")


# Module CBAM (Channel + Spatial Attention) 
# Aide le modèle à se focaliser sur les zones avec petits nodules
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x, dim=1, keepdim=True).values
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# Modèle EfficientNet-B4 avec CBAM 
print("\nChargement d'EfficientNet-B4 avec poids ImageNet...")
_backbone = models.efficientnet_b4(weights='IMAGENET1K_V1')
num_ftrs = _backbone.classifier[1].in_features  # 1792

class EfficientNetCBAM(nn.Module):
    def __init__(self, backbone, num_classes=2, use_cbam=True):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.cbam = CBAM(num_ftrs) if use_cbam else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = EfficientNetCBAM(_backbone, num_classes=2, use_cbam=USE_CBAM).to(device)

# Compter les paramètres
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModele EfficientNet-B4 + CBAM cree:")
print(f"   Architecture: EfficientNet-B4 + CBAM Attention")
print(f"   Poids pre-entraines: ImageNet (1.2M images)")
print(f"   CBAM Attention: {USE_CBAM} (meilleur detection petits objets)")
print(f"   Freeze backbone: {FREEZE_BACKBONE}")
print(f"   Preprocessing: CLAHE actif" if USE_CLAHE else "   Preprocessing: desactive")
print(f"   Normalisation: ImageNet")
print(f"   Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"   Parametres totaux: {total_params:,}")
print(f"   Parametres entrainables: {trainable_params:,}")

wandb.log({
    "model/total_params": total_params,
    "model/trainable_params": trainable_params
})


# Focal Loss 
# Bien meilleure que la CrossEntropyLoss pondérée pour détecter des objets rares.
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        # alpha_t: poids selon la classe réelle
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * ce
        return loss.mean() if self.reduction == 'mean' else loss

criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

# Layer-wise learning rates: backbone LR très faible (ne pas détruire les features ImageNet), mais têtes de classification et CBAM ont un LR plus grand.
backbone_params = list(model.features.parameters())
head_params = list(model.cbam.parameters()) + list(model.classifier.parameters())

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LR_BACKBONE},
    {'params': head_params,    'lr': LR_HEAD}
], weight_decay=WEIGHT_DECAY)

# CosineAnnealing sur 40 epochs avec redémarrages (T_mult=2 → 40, 80, 160, 320)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=1e-7)

print(f"\nConfiguration:")
print(f"   Loss: Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
print(f"   Optimizer: AdamW layer-wise (backbone={LR_BACKBONE}, head={LR_HEAD})")
print(f"   Scheduler: CosineAnnealingWarmRestarts (T_0=40, T_mult=2)")
print(f"   Early Stopping: patience={EARLY_STOPPING_PATIENCE} epochs, min={MIN_EPOCHS} epochs, metric={EARLY_STOP_METRIC}")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, precision, recall, f1, auc, cm


best_val_metric = -float('inf')  # On maximise F1 (ou AUC)
best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
           'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []}

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Entraînement
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validation
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm = validate(
        model, val_loader, criterion, device
    )
    
    # Scheduler
    scheduler.step()
    
    epoch_time = time.time() - epoch_start
    
    # Affichage
    print(f"\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    print(f"        Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
    print(f"Temps: {epoch_time:.1f}s")
    
        wandb.log({
        "epoch": epoch + 1,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "val/precision": val_prec,
        "val/recall": val_rec,
        "val/f1": val_f1,
        "val/auc": val_auc,
        "lr/backbone": optimizer.param_groups[0]['lr'],
        "lr/head": optimizer.param_groups[1]['lr'],
        "epoch_time": epoch_time
    })
    
    # Historique
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_precision'].append(val_prec)
    history['val_recall'].append(val_rec)
    history['val_f1'].append(val_f1)
    history['val_auc'].append(val_auc)
    
    # Sauvegarder le meilleur modèle selon le metric choisi
    # On suit val_f1 qui est robuste sur dataset déséquilibré
    # (val_loss peut diminuer même si le recall sur nodules stagne ou baisse)
    current_metric = val_f1 if EARLY_STOP_METRIC == 'val_f1' else val_auc
    if current_metric > best_val_metric:
        best_val_metric = current_metric
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'history': history
        }, CHECKPOINT_DIR / "best_model_efficientnet.pth")
        print(f"Meilleur modele sauvegarde ({EARLY_STOP_METRIC}: {current_metric:.4f})")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE} | best_{EARLY_STOP_METRIC}: {best_val_metric:.4f}")

    # Early stopping (uniquement après MIN_EPOCHS pour laisser le modèle converger)
    if epoch + 1 >= MIN_EPOCHS and patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping a l'epoque {epoch+1} (patience {EARLY_STOPPING_PATIENCE} atteinte apres epoch {MIN_EPOCHS})")
        break


# Charger le meilleur modèle
checkpoint = torch.load(CHECKPOINT_DIR / "best_model_efficientnet.pth", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_cm = validate(
    model, test_loader, criterion, device
)

print(f"\nResultats finaux (Test Set):")
print(f"   Accuracy:  {test_acc:.4f}")
print(f"   Precision: {test_prec:.4f}")
print(f"   Recall:    {test_rec:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   AUC-ROC:   {test_auc:.4f}")
print(f"\n   Confusion Matrix:")
print(f"                Pred 0  Pred 1")
print(f"   True 0 (Sain)  {test_cm[0,0]:4d}    {test_cm[0,1]:4d}")
print(f"   True 1 (Nodule){test_cm[1,0]:4d}    {test_cm[1,1]:4d}")

wandb.log({
    "test/accuracy": test_acc,
    "test/precision": test_prec,
    "test/recall": test_rec,
    "test/f1": test_f1,
    "test/auc": test_auc,
    "test/true_negatives": int(test_cm[0, 0]),
    "test/false_positives": int(test_cm[0, 1]),
    "test/false_negatives": int(test_cm[1, 0]),
    "test/true_positives": int(test_cm[1, 1])
})

# Sauvegarder les résultats
results_file = OUTPUT_DIR / "test_results_efficientnet.txt"
with open(results_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("RÉSULTATS - EFFICIENTNET-B4 TRANSFER LEARNING\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Accuracy:  {test_acc:.4f}\n")
    f.write(f"Precision: {test_prec:.4f}\n")
    f.write(f"Recall:    {test_rec:.4f}\n")
    f.write(f"F1-Score:  {test_f1:.4f}\n")
    f.write(f"AUC-ROC:   {test_auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"                Pred 0  Pred 1\n")
    f.write(f"True 0 (Sain)    {test_cm[0,0]:4d}    {test_cm[0,1]:4d}\n")
    f.write(f"True 1 (Nodule)  {test_cm[1,0]:4d}    {test_cm[1,1]:4d}\n")

print(f"\nResultats sauvegardes: {results_file}")


# Training curves
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
epochs_range = range(1, len(history['train_loss']) + 1)

# Loss
axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=2)
axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(epochs_range, history['train_acc'], 'b-', label='Train', linewidth=2)
axes[0, 1].plot(epochs_range, history['val_acc'], 'r-', label='Val', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Métriques
axes[1, 0].plot(epochs_range, history['val_precision'], label='Precision', linewidth=2)
axes[1, 0].plot(epochs_range, history['val_recall'], label='Recall', linewidth=2)
axes[1, 0].plot(epochs_range, history['val_f1'], label='F1-Score', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Score', fontsize=12)
axes[1, 0].set_title('Métriques de Validation', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)


axes[1, 1].plot(epochs_range, history['val_auc'], 'g-', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('AUC-ROC', fontsize=12)
axes[1, 1].set_title('AUC-ROC de Validation', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
curves_path = OUTPUT_DIR / "training_curves_efficientnet.png"
plt.savefig(curves_path, dpi=150, bbox_inches='tight')
wandb.log({"visualizations/training_curves": wandb.Image(str(curves_path))})
plt.close()
print(f"\nCourbes d'entrainement: {curves_path}")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
cm_percent = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis] * 100
annot = np.empty_like(test_cm, dtype=object)
for i in range(test_cm.shape[0]):
    for j in range(test_cm.shape[1]):
        annot[i, j] = f'{test_cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

sns.heatmap(test_cm, annot=annot, fmt='', cmap='Blues', 
            xticklabels=['No Finding', 'Nodule'], 
            yticklabels=['No Finding', 'Nodule'],
            cbar_kws={'label': 'Count'}, ax=ax, square=True, annot_kws={"size": 14})
ax.set_xlabel('Prédiction', fontsize=14)
ax.set_ylabel('Vérité terrain', fontsize=14)
ax.set_title('Matrice de Confusion (Test Set) - EfficientNet-B4', fontsize=16, fontweight='bold')
plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix_efficientnet.png"
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
wandb.log({"visualizations/confusion_matrix": wandb.Image(str(cm_path))})
plt.close()
print(f"Matrice de confusion: {cm_path}")


print(f"\nModele sauvegarde: {CHECKPOINT_DIR / 'best_model_efficientnet.pth'}")
print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

close_wandb()
