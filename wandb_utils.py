
import os
import wandb
from pathlib import Path
from datetime import datetime


def load_wandb_key():
    key_file = Path(__file__).parent / ".wandb_key"
    
    if not key_file.exists():
        print("ATTENTION: Fichier .wandb_key non trouvé!")
        print(f"Créez le fichier: {key_file}")
        print("Et ajoutez votre clé API wandb dedans (une seule ligne)")
        return None
    
    try:
        with open(key_file, 'r') as f:
            # Lire ligne par ligne et trouver la première ligne non commentée
            for line in f:
                line = line.strip()
                # Ignorer les lignes vides et commentées
                if line and not line.startswith('#'):
                    return line
        
        # Si on arrive ici, toutes les lignes sont commentées ou vides
        print("ATTENTION: Fichier .wandb_key vide ou toutes les lignes sont commentées!")
        print("Ajoutez votre clé API sur une ligne sans # devant")
        return None
    
    except Exception as e:
        print(f"ERREUR lors de la lecture de .wandb_key: {e}")
        return None


def setup_wandb(project_name, run_name, config, job_type="train", offline_mode=True):
    
    # Charger la clé API
    api_key = load_wandb_key()
    
    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
            print(f"Connexion wandb réussie!")
        except Exception as e:
            print(f"ATTENTION: Échec de la connexion wandb: {e}")
            print("continuer en mode offline...")
    else:
        print("Mode offline, pas d'authentification wandb")
    
    # Configuration du mode
    if offline_mode:
        os.environ['WANDB_MODE'] = 'offline'
        print(f"Mode: OFFLINE (les logs seront synchronisés plus tard avec 'wandb sync')")
    else:
        os.environ['WANDB_MODE'] = 'online'
        print(f"Mode: ONLINE (synchronisation en temps réel)")
    
    # Vérifier si on partage une run unique (définie par train_job.sh)
    shared_run_id = os.environ.get('WANDB_SHARED_RUN_ID')
    if shared_run_id:
        print(f"Mode: RUN PARTAGEE (ID: {shared_run_id})")
        print(f"  Exploration + Entraînement → même run wandb")
    
    # Initialiser la run
    try:
        if '/' in project_name:
            entity, project = project_name.split('/', 1)
        else:
            entity, project = None, project_name

        init_kwargs = dict(
            project=project,
            config=config,
            job_type=job_type,
        )
        if entity:
            init_kwargs['entity'] = entity

        if shared_run_id:
            # Rejoindre la run partagée (résume si déjà démarrée)
            init_kwargs['id'] = shared_run_id
            init_kwargs['resume'] = 'allow'
            # Le nom de la run unique est défini à la première init
            init_kwargs['name'] = os.environ.get('WANDB_SHARED_RUN_NAME', run_name)
        else:
            init_kwargs['name'] = run_name
            init_kwargs['resume'] = 'allow'

        wandb.init(**init_kwargs)
        print(f"Run wandb initialisée: {wandb.run.name}")
        print(f"  Projet: {project_name}")
        return True
    
    except Exception as e:
        print(f"ERREUR lors de l'initialisation wandb: {e}")
        return False


def close_wandb():
    try:
        wandb.finish()
        print("\nRun wandb fermée")
        
        # Si en mode offline, rappeler la commande de sync
        if os.environ.get('WANDB_MODE') == 'offline':
            print("\nPour synchroniser les logs vers wandb.ai:")
            print("  wandb sync --sync-all")
    
    except Exception as e:
        print(f"ATTENTION lors de la fermeture wandb: {e}")


# Configuration par défaut pour le projet
DEFAULT_PROJECT = "leonard-zipper-utbm-utbm/Nodulocc_challenge"


def get_project_name():
    return DEFAULT_PROJECT
