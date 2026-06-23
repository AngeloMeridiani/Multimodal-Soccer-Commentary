"""
config.py
=========
Configurazione centralizzata dell'intera pipeline.

Tutti i percorsi sono RELATIVI alla root del progetto, calcolata dinamicamente
a partire dalla posizione di questo file. Questo permette al team di clonare la
repo e lanciare gli script da qualunque directory senza modificare i path.

Convenzione "a staffetta":
    Fase 1 (testo)   -> scrive features/text/text_features.npy   + features/manifest.csv
    Fase 2 (audio)   -> legge il manifest, scrive features/audio/audio_features.npy
    Fase 3 (vision)  -> legge il manifest, scrive features/vision/vision_features.npy
    Fase 4 (fusione) -> legge tutto, concatena (Early Fusion) e addestra il classificatore.
"""

from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Percorsi (PROJECT_ROOT = cartella che contiene src/, data/, features/, ...)  #
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Dati grezzi (input della pipeline)
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
VIDEO_DIR: Path = RAW_DIR / "videos"              # clip .mp4 di MUStARD
SARCASM_JSON: Path = RAW_DIR / "sarcasm_data.json"  # annotazioni ufficiali MUStARD

# Feature estratte (output intermedi, uno per fase)
FEATURES_DIR: Path = PROJECT_ROOT / "features"
TEXT_FEATURES_PATH: Path = FEATURES_DIR / "text" / "text_features.npy"
AUDIO_FEATURES_PATH: Path = FEATURES_DIR / "audio" / "audio_features.npy"
VISION_FEATURES_PATH: Path = FEATURES_DIR / "vision" / "vision_features.npy"

# Manifest: "fonte di verità" creata dalla Fase 1.
# Contiene l'elenco ordinato degli ID e la label (0/1). Le fasi 2-3-4 lo leggono.
MANIFEST_PATH: Path = FEATURES_DIR / "manifest.csv"

# Modelli addestrati
MODELS_DIR: Path = PROJECT_ROOT / "models"
CLASSIFIER_PATH: Path = MODELS_DIR / "fusion_mlp.pt"
SCALER_PATH: Path = MODELS_DIR / "scaler.joblib"

# --------------------------------------------------------------------------- #
# Hardware                                                                     #
# --------------------------------------------------------------------------- #
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Fase 1 - NLP (RoBERTa)                                                       #
# --------------------------------------------------------------------------- #
TEXT_MODEL_NAME: str = "roberta-base"
TEXT_MAX_LENGTH: int = 128
TEXT_BATCH_SIZE: int = 16
TEXT_EMBED_DIM: int = 768          # dimensione hidden di roberta-base
USE_CONTEXT: bool = False          # se True concatena il contesto all'utterance

# --------------------------------------------------------------------------- #
# Fase 2 - Audio (librosa / MFCC)                                              #
# --------------------------------------------------------------------------- #
AUDIO_SAMPLE_RATE: int = 22050
N_MFCC: int = 40
INCLUDE_DELTAS: bool = True        # aggiunge delta e delta-delta
# Dim risultante = N_MFCC * (1 + 2*INCLUDE_DELTAS) * 2  (pooling mean+std)
AUDIO_EMBED_DIM: int = N_MFCC * (3 if INCLUDE_DELTAS else 1) * 2

# --------------------------------------------------------------------------- #
# Fase 3 - Computer Vision (MediaPipe + ResNet50)                              #
# --------------------------------------------------------------------------- #
FRAMES_PER_SECOND: float = 1.0     # campionamento temporale (1-2 fps consigliato)
FACE_DETECTION_CONFIDENCE: float = 0.5
IMAGE_SIZE: int = 224              # input ResNet50
VISION_EMBED_DIM: int = 2048       # output ResNet50 senza il layer fc
# Se True, in assenza di volti usa il frame intero come fallback invece di scartare
VISION_FULLFRAME_FALLBACK: bool = True

# --------------------------------------------------------------------------- #
# Fase 4 - Fusione & Classificazione                                          #
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.2
RANDOM_SEED: int = 42
MLP_HIDDEN_DIMS: tuple[int, ...] = (512, 128)
MLP_DROPOUT: float = 0.3
MLP_EPOCHS: int = 120
MLP_LR: float = 1e-3
MLP_WEIGHT_DECAY: float = 1e-4
MLP_BATCH_SIZE: int = 32
EARLY_STOPPING_PATIENCE: int = 15