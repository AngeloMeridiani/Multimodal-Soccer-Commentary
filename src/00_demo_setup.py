"""
00_demo_setup.py
================
Setup iniziale del progetto: crea le directory, verifica le dipendenze e genera
file di esempio per poter avviare la pipeline immediatamente.

Uso:
    python src/00_demo_setup.py
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

# Aggiungi src al path per importare config
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import ensure_dir, get_logger

logger = get_logger("setup")


# --------------------------------------------------------------------------- #
# Directory da creare                                                          #
# --------------------------------------------------------------------------- #
REQUIRED_DIRS: list[Path] = [
    config.GAMEPLAY_DIR,
    config.COMMENTARY_DIR,
    config.EVENTS_DIR,
    config.SCRIPTS_DIR,
    config.PROSODY_DIR,
    config.MODELS_DIR,
    config.AUDIO_OUT_DIR,
    config.STUDY_DIR,
]


# --------------------------------------------------------------------------- #
# Dipendenze da verificare                                                     #
# --------------------------------------------------------------------------- #
DEPENDENCIES: list[tuple[str, str, bool]] = [
    # (modulo_python, descrizione, obbligatorio)
    ("cv2",         "OpenCV (opencv-python)",           True),
    ("easyocr",     "EasyOCR (OCR deep learning)",      True),
    ("torch",       "PyTorch",                          True),
    ("librosa",     "Librosa (audio analysis)",         True),
    ("soundfile",   "SoundFile (audio I/O)",            True),
    ("pyttsx3",     "pyttsx3 (TTS offline)",            True),
    ("numpy",       "NumPy",                            True),
    ("pandas",      "Pandas",                           True),
    ("sklearn",     "scikit-learn",                     True),
    ("tqdm",        "tqdm (progress bars)",             True),
    # Nuovi moduli (Livello 2)
    ("ultralytics", "Ultralytics YOLOv8 (detection)",   False),
    ("whisper",     "OpenAI Whisper (speech-to-text)",   False),
    ("TTS",         "Coqui TTS (TTS espressivo)",       False),
]


def create_directories() -> None:
    """Crea tutte le directory necessarie alla pipeline."""
    logger.info("=== Creazione directory ===")
    for d in REQUIRED_DIRS:
        ensure_dir(d / "_placeholder")  # ensure_dir lavora su file, trucco per dir
        # Aggiungi .gitkeep per tenere le dir vuote sotto git
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
        logger.info("  ✓ %s", d.relative_to(config.PROJECT_ROOT))


def check_dependencies() -> tuple[list[str], list[str]]:
    """Verifica quali dipendenze sono installate."""
    logger.info("\n=== Verifica dipendenze ===")
    ok, missing = [], []
    for module, desc, required in DEPENDENCIES:
        try:
            importlib.import_module(module)
            logger.info("  ✓ %-35s installato", desc)
            ok.append(desc)
        except ImportError:
            tag = "OBBLIGATORIO" if required else "opzionale"
            logger.warning("  ✗ %-35s NON trovato (%s)", desc, tag)
            missing.append(f"{desc} ({tag})")
    return ok, missing


def create_example_annotations() -> None:
    """Genera un CSV di annotazioni prosodiche di esempio."""
    csv_path = config.PROSODY_ANNOTATIONS
    if csv_path.exists():
        logger.info("\n  Annotazioni già presenti: %s", csv_path)
        return

    ensure_dir(csv_path)
    rows = [
        {"clip": "esempio_telecronaca.wav", "start": "0.0", "end": "3.5", "event_type": "goal"},
        {"clip": "esempio_telecronaca.wav", "start": "5.0", "end": "7.2", "event_type": "pass"},
        {"clip": "esempio_telecronaca.wav", "start": "10.1", "end": "12.0", "event_type": "turnover"},
        {"clip": "esempio_telecronaca.wav", "start": "15.0", "end": "17.5", "event_type": "idle"},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "start", "end", "event_type"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("\n  ✓ Annotazioni di esempio create: %s", csv_path)


def create_example_roster() -> None:
    """Stampa istruzioni per compilare il roster in config.py."""
    if config.ROSTER:
        logger.info("\n  Roster già compilato (%d giocatori).", len(config.ROSTER))
        return
    logger.info("\n  ⚠  Il ROSTER in config.py è vuoto.")
    logger.info("     Per distinguere 'pass' da 'turnover', compilalo così:")
    logger.info('     ROSTER = {"RONALDO": "home", "MESSI": "away", ...}')


def print_quickstart() -> None:
    """Stampa le istruzioni per il quickstart."""
    print("\n" + "=" * 70)
    print("  QUICKSTART — Come partire")
    print("=" * 70)
    print("""
  1. METTI UN VIDEO FIFA nella directory:
     data/raw/gameplay/match1.mp4

  2. CALIBRA L'HUD: apri un frame del video e aggiorna le coordinate
     in config.py → HUD_REGIONS (punteggio e giocatore attivo).

  3. COMPILA IL ROSTER in config.py con i nomi dei giocatori.

  4. ESEGUI LA PIPELINE BASE (modalità flat, senza prosodia):
     python src/01_extract_events.py --video data/raw/gameplay/match1.mp4
     python src/02_generate_script.py --events features/events/match1.json
     python src/04_synthesize.py --script features/scripts/match1.json --mode flat

  5. AGGIUNGI I MODULI AVANZATI (Livello 2):
     python src/01b_visual_analysis.py --video data/raw/gameplay/match1.mp4
     python src/01c_audio_analysis.py --video data/raw/gameplay/match1.mp4
     python src/02b_generate_llm.py --events features/events/match1_enriched.json

  6. PER IL TRAINING DELLA PROSODIA (Livello 3):
     - Metti clip di telecronache reali in data/raw/commentary/
     - Annota i segmenti in data/raw/prosody_annotations.csv
     - Esegui:
       python src/03_train_prosody.py build-dataset
       python src/03_train_prosody.py train
""")


def main() -> None:
    print("=" * 70)
    print("  SETUP — Telecronaca AI per FIFA")
    print("=" * 70)

    create_directories()
    ok, missing = check_dependencies()
    create_example_annotations()
    create_example_roster()

    if missing:
        logger.warning("\n⚠  Dipendenze mancanti (%d):", len(missing))
        for m in missing:
            logger.warning("   - %s", m)
        logger.info("   Installa con: pip install -r requirements.txt")
    else:
        logger.info("\n✓ Tutte le dipendenze sono installate!")

    print_quickstart()


if __name__ == "__main__":
    main()
