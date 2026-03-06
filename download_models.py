
import torchvision.models as models


# EfficientNet-B4
print("\n1. Téléchargement EfficientNet-B4 (IMAGENET1K_V1)...")
try:
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    print("   EfficientNet-B4 téléchargé avec succès!")
except Exception as e:
    print(f"   Erreur: {e}")

