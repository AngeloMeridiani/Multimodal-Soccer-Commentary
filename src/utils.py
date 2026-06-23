"""
utils.py
========
Funzioni di utilità condivise da tutte le fasi della pipeline.

Contiene:
- get_logger:        logging uniforme su console.
- set_seed:          riproducibilità.
- ensure_dir:        crea le cartelle di output se mancanti.
- save/load_feature_dict: contratto di I/O tra le fasi (dict {id: vettore}).
- load_manifest:     legge l'elenco ordinato di ID e label prodotto dalla Fase 1.
- iter_sampled_frames: generatore di frame campionati a N fps da un video.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Restituisce un logger configurato con formato uniforme."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # evita handler duplicati su import multipli
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# Riproducibilità                                                             #
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Fissa i seed di random e numpy (e torch se disponibile)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Filesystem                                                                   #
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> None:
    """Crea la directory (anche genitori) se non esiste. Accetta file o dir."""
    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Contratto di I/O delle feature                                              #
# --------------------------------------------------------------------------- #
def save_feature_dict(features: Dict[str, np.ndarray], path: Path) -> None:
    """
    Salva un dizionario {utterance_id: vettore_feature} su disco.

    Usiamo un dict (non un array stacked) perché alcune fasi - tipicamente la
    Computer Vision - possono fallire su singole clip: salvare per-ID permette
    alla Fase 4 di allineare le modalita' e gestire i buchi.
    """
    ensure_dir(path)
    np.save(path, features, allow_pickle=True)


def load_feature_dict(path: Path) -> Dict[str, np.ndarray]:
    """Carica un dizionario {id: vettore} salvato con save_feature_dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature non trovate: {path}. "
            f"Hai eseguito la fase precedente della pipeline?"
        )
    return np.load(path, allow_pickle=True).item()


def load_manifest(path: Path) -> Tuple[list[str], np.ndarray]:
    """
    Legge il manifest CSV (id,label) prodotto dalla Fase 1.

    Returns:
        ids:    lista ordinata degli utterance_id.
        labels: array numpy (int) delle label allineate agli id.
    """
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest non trovato: {path}. Esegui prima 01_extract_text.py."
        )
    df = pd.read_csv(path, dtype={"id": str})
    return df["id"].tolist(), df["label"].to_numpy(dtype=np.int64)


# --------------------------------------------------------------------------- #
# Video                                                                        #
# --------------------------------------------------------------------------- #
def iter_sampled_frames(
    video_path: Path, frames_per_second: float
) -> Iterator[np.ndarray]:
    """
    Generatore che restituisce frame BGR (numpy) campionati a `frames_per_second`.

    Apre il video con OpenCV, calcola lo step in base agli FPS nativi e cede solo
    i frame agli istanti voluti. Solleva RuntimeError se il video non e' apribile.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback se metadato assente
    step = max(int(round(native_fps / max(frames_per_second, 1e-6))), 1)

    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield frame
            idx += 1
    finally:
        cap.release()