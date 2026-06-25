"""
utils.py
========
Utilita' condivise da tutte le fasi della pipeline.

- get_logger / set_seed / ensure_dir : infrastruttura di base.
- save_json / load_json              : I/O del log eventi e degli script.
- iter_sampled_frames                : generatore di frame campionati da un video.
- event_to_features                  : trasforma un evento nel vettore di input
                                       del modello di prosodia (importanza + one-hot).
- apply_prosody                      : applica i parametri prosodici a un'onda
                                       audio neutra (time-stretch, pitch, gain).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterator

import numpy as np

import config


# --------------------------------------------------------------------------- #
# Infrastruttura                                                               #
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: Path) -> None:
    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# I/O JSON                                                                     #
# --------------------------------------------------------------------------- #
def save_json(obj, path: Path) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Video                                                                        #
# --------------------------------------------------------------------------- #
def iter_sampled_frames(
    video_path: Path, frames_per_second: float
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Genera coppie (timestamp_secondi, frame_BGR) campionando a `frames_per_second`.
    Solleva RuntimeError se il video non e' apribile.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native_fps / max(frames_per_second, 1e-6))), 1)

    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx / native_fps, frame
            idx += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Eventi -> feature per il modello di prosodia                                 #
# --------------------------------------------------------------------------- #
def event_to_features(event_type: str, importance: float) -> np.ndarray:
    """
    Vettore di input del modello: [importanza, one-hot del tipo di evento].
    Mantiene un formato unico e coerente tra training (Fase 3) e sintesi (Fase 4).
    """
    one_hot = [1.0 if event_type == t else 0.0 for t in config.EVENT_TYPES]
    return np.asarray([importance, *one_hot], dtype=np.float32)


FEATURE_DIM: int = 1 + len(config.EVENT_TYPES)  # importanza + one-hot


# --------------------------------------------------------------------------- #
# Feature ESTESE per il modello arricchito (Livello 3)                         #
# --------------------------------------------------------------------------- #
# Mappature categoriche -> numeriche per le feature estese
CROWD_LEVEL_MAP: dict[str, float] = {
    "low": 0.0, "medium": 0.33, "high": 0.66, "peak": 1.0,
}
BALL_SPEED_MAP: dict[str, float] = {
    "stopped": 0.0, "low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 1.0,
}
EMOTION_LEVEL_MAP: dict[str, float] = {
    "low": 0.0, "medium": 0.33, "high": 0.66, "very_high": 1.0,
}
BALL_ZONE_LIST: list[str] = [
    "penalty_area_home", "penalty_area_away", "midfield", "wing_left", "wing_right",
]


def event_to_features_extended(event: dict) -> np.ndarray:
    """
    Vettore esteso: aggiunge crowd_excitement, ball_zone, ball_speed e
    emotion_intensity alle feature base. Usato quando
    config.PROSODY_USE_EXTENDED_FEATURES e' True.

    Formato: [importanza, one-hot_evento(11), crowd_score, ball_speed,
              emotion_intensity, one-hot_ball_zone(5)]
    """
    event_type = event.get("type", "idle")
    importance = event.get("importance", config.EVENT_IMPORTANCE.get(event_type, 0.1))

    # One-hot esteso (usa EVENT_TYPES_EXTENDED)
    one_hot_event = [
        1.0 if event_type == t else 0.0
        for t in config.EVENT_TYPES_EXTENDED
    ]

    # Crowd excitement (numerico)
    crowd_level = event.get("crowd_excitement", "low")
    crowd_score = event.get("crowd_score", CROWD_LEVEL_MAP.get(crowd_level, 0.0))

    # Ball speed (numerico)
    ball_speed_str = event.get("ball_speed", "medium")
    ball_speed = BALL_SPEED_MAP.get(ball_speed_str, 0.5)

    # Emotion intensity (numerico)
    emotion_str = event.get("emotion_intensity", "medium")
    emotion = EMOTION_LEVEL_MAP.get(emotion_str, 0.33)

    # Ball zone (one-hot)
    ball_zone = event.get("ball_zone", "midfield")
    one_hot_zone = [1.0 if ball_zone == z else 0.0 for z in BALL_ZONE_LIST]

    features = [importance, *one_hot_event, crowd_score, ball_speed, emotion, *one_hot_zone]
    return np.asarray(features, dtype=np.float32)


FEATURE_DIM_EXTENDED: int = (
    1                                    # importanza
    + len(config.EVENT_TYPES_EXTENDED)   # one-hot evento esteso
    + 3                                  # crowd_score, ball_speed, emotion
    + len(BALL_ZONE_LIST)                # one-hot ball_zone
)



# --------------------------------------------------------------------------- #
# Applicazione della prosodia all'audio neutro (DSP)                           #
# --------------------------------------------------------------------------- #
def apply_prosody(
    waveform: np.ndarray,
    sample_rate: int,
    rate_factor: float,
    pitch_semitones: float,
    energy_gain: float,
) -> np.ndarray:
    """
    Trasforma un'onda audio NEUTRA secondo i parametri prosodici.

    - rate_factor      > 1 accelera il parlato (telecronaca concitata).
    - pitch_semitones  > 0 alza il tono (eccitazione).
    - energy_gain      > 1 aumenta il volume.

    Disaccoppia il CONTRIBUTO (la mappatura evento->prosodia, appresa) dal motore
    TTS: qualunque TTS produca l'audio neutro, qui lo si rende espressivo. Questo
    rende l'esperimento riproducibile e indipendente dal TTS scelto.
    """
    import librosa

    out = waveform.astype(np.float32)

    # 1) Velocita': time-stretch (non altera il pitch)
    if abs(rate_factor - 1.0) > 1e-3:
        out = librosa.effects.time_stretch(out, rate=float(rate_factor))

    # 2) Pitch: spostamento in semitoni
    if abs(pitch_semitones) > 1e-3:
        out = librosa.effects.pitch_shift(
            out, sr=sample_rate, n_steps=float(pitch_semitones)
        )

    # 3) Energia: guadagno con clipping di sicurezza
    out = out * float(energy_gain)
    peak = np.max(np.abs(out)) if out.size else 0.0
    if peak > 1.0:                       # evita distorsione/clipping
        out = out / peak

    return out