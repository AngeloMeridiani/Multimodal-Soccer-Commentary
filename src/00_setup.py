"""
00_setup.py
===========
Initial setup: creates the directories, checks the dependencies and generates
example files to start the pipeline right away.

Usage:
    python 00_setup.py
"""

from __future__ import annotations

import csv
import importlib

import config
from utils import ensure_dir, get_logger

logger = get_logger("setup")

REQUIRED_DIRS = [
    config.GAMEPLAY_DIR,
    config.COMMENTARY_DIR,
    config.NORMALIZED_DIR,
    config.EVENTS_DIR,
    config.SCRIPTS_DIR,
    config.PROSODY_DIR,
    config.MODELS_DIR,
    config.AUDIO_OUT_DIR,
    config.STUDY_DIR,
]

# (module, description, required)
DEPENDENCIES = [
    ("cv2", "OpenCV (opencv-python)", True),
    ("numpy", "NumPy", True),
    ("easyocr", "EasyOCR (OCR deep learning)", True),
    ("librosa", "Librosa (audio)", True),
    ("soundfile", "SoundFile (audio I/O)", True),
    ("torch", "PyTorch", True),
    ("pandas", "Pandas", True),
    ("sklearn", "scikit-learn", True),
    ("scipy", "SciPy", True),
    ("tqdm", "tqdm", True),
    ("ultralytics", "Ultralytics YOLOv8 (opzionale)", False),
    ("whisper", "OpenAI Whisper (opzionale)", False),
    ("TTS", "Coqui TTS / XTTS v2 (Fase 4)", True),
]


def create_directories() -> None:
    """
    Create all the necessary directories for the project, such as data and models.
    Also creates a .gitkeep file to ensure directories are tracked by git.
    """
    logger.info("=== Creazione directory ===")
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch(exist_ok=True)
        logger.info("  ok %s", d.relative_to(config.PROJECT_ROOT))


def check_dependencies():
    """
    Check if all the required and optional python packages are installed.
    Logs warnings for any missing packages.
    """
    logger.info("=== Verifica dipendenze ===")
    ok, missing = [], []
    for module, desc, required in DEPENDENCIES:
        try:
            importlib.import_module(module)
            logger.info("  ok  %-35s installato", desc)
            ok.append(desc)
        except ImportError:
            tag = "OBBLIGATORIO" if required else "opzionale"
            logger.warning("  --  %-35s mancante (%s)", desc, tag)
            missing.append(f"{desc} ({tag})")
    return ok, missing


def create_example_annotations() -> None:
    """
    Create an example CSV file for prosody annotations if it does not already exist.
    Provides a template for manual prosody event tagging.
    """
    csv_path = config.PROSODY_ANNOTATIONS
    if csv_path.exists():
        logger.info("Annotazioni gia' presenti: %s", csv_path)
        return
    ensure_dir(csv_path)
    rows = [
        {"clip": "esempio.wav", "start": "0.0", "end": "3.5", "event_type": "goal"},
        {"clip": "esempio.wav", "start": "5.0", "end": "7.2", "event_type": "pass"},
        {"clip": "esempio.wav", "start": "10.1", "end": "12.0", "event_type": "turnover"},
        {"clip": "esempio.wav", "start": "15.0", "end": "17.5", "event_type": "idle"},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "start", "end", "event_type"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Annotazioni di esempio create: %s", csv_path)


def main() -> None:
    """
    Main entry point for the setup script.
    Executes directory creation, dependency checks, and annotation setup.
    """
    print("=" * 64)
    print("  SETUP - Telecronaca AI per EA FC / FIFA")
    print("=" * 64)
    create_directories()
    ok, missing = check_dependencies()
    create_example_annotations()
    if missing:
        logger.warning(
            "Dipendenze mancanti (%d): installa con  pip install -r requirements.txt", len(missing)
        )
    else:
        logger.info("Tutte le dipendenze sono installate.")
    print("\nProssimo passo:")
    print("  1) metti un video in  data/raw/gameplay/")
    print("  2) calibra l'HUD:     python 01_calibrate_hud.py --video <clip> --ocr")
    print("  3) estrai gli eventi: python 01_extract_events.py --video <clip>")


if __name__ == "__main__":
    main()
