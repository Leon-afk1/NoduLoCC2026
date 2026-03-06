#!/bin/bash
# Script pour configurer facilement la clé API Wandb

echo "========================================"
echo "Configuration Wandb - Clé API"
echo "========================================"
echo ""

KEY_FILE=".wandb_key"

# Vérifier si le fichier existe déjà et contient une clé
if [ -f "$KEY_FILE" ]; then
    # Lire la première ligne non commentée
    CURRENT_KEY=$(grep -v '^#' "$KEY_FILE" | grep -v '^[[:space:]]*$' | head -1)
    
    if [ -n "$CURRENT_KEY" ]; then
        echo "✓ Une clé API existe déjà dans $KEY_FILE"
        echo ""
        echo "Voulez-vous la remplacer? (y/N)"
        read -r REPLACE
        
        if [[ ! "$REPLACE" =~ ^[Yy]$ ]]; then
            echo "Configuration annulée"
            exit 0
        fi
    fi
fi

echo ""
echo "Pour obtenir votre clé API Wandb:"
echo "1. Allez sur: https://wandb.ai/authorize"
echo "2. Connectez-vous (ou créez un compte)"
echo "3. Copiez votre clé API"
echo ""
echo "Entrez votre clé API Wandb:"
read -r WANDB_KEY

# Validation basique
if [ -z "$WANDB_KEY" ]; then
    echo "✗ Erreur: La clé ne peut pas être vide"
    exit 1
fi

if [ ${#WANDB_KEY} -lt 20 ]; then
    echo "✗ Erreur: La clé semble trop courte (${#WANDB_KEY} caractères)"
    echo "Une clé API Wandb fait généralement 40 caractères"
    exit 1
fi

# Écrire la clé dans le fichier
echo "$WANDB_KEY" > "$KEY_FILE"
chmod 600 "$KEY_FILE"  # Permissions restrictives

echo ""
echo "✓ Clé API sauvegardée dans $KEY_FILE"
echo "✓ Permissions configurées (600)"
echo ""

# Tester la clé
echo "Test de la connexion..."
python3 << EOF
from wandb_utils import load_wandb_key, setup_wandb
import sys

key = load_wandb_key()
if key:
    print("✓ Clé chargée avec succès!")
    print(f"  Longueur: {len(key)} caractères")
    print("")
    print("Configuration du projet:")
    print("  Projet: leonard-zipper-utbm-utbm/Nodulocc_challenge")
    sys.exit(0)
else:
    print("✗ Erreur lors du chargement de la clé")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Configuration terminée avec succès!"
    echo "========================================"
    echo ""
    echo "Vous pouvez maintenant lancer vos scripts:"
    echo "  - python train_simple_cnn.py"
    echo "  - python exploration.py"
    echo "  - sbatch train_job.sh"
    echo ""
    echo "Vos runs seront disponibles sur:"
    echo "  https://wandb.ai/leonard-zipper-utbm-utbm/Nodulocc_challenge"
else
    echo ""
    echo "✗ Une erreur s'est produite lors du test"
    echo "Vérifiez que wandb_utils.py est présent"
fi
