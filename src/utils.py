"""
utils.py
========
Utilita' condivise da tutte le fasi.

- get_logger / set_seed / ensure_dir / save_json / load_json : infrastruttura.
- VideoNormalizer + iter_sampled_frames : lettura video RADDRIZZATO e RITAGLIATO
  (risolve rotazione ignorata da OpenCV + bande nere laterali).
- event_to_features / FEATURE_DIM : evento -> vettore di input del modello
  prosodia (importanza + one-hot UNICO su config.EVENT_TYPES).
- ProsodyMLP : la rete (qui, cosi' Fase 3 e Fase 4 la importano senza accoppiarsi
  al nome-file).
- apply_prosody : applica i parametri prosodici a un'onda audio neutra (DSP).
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
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


logger = get_logger("utils")


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
    """Crea la cartella di `path` (o `path` stessa se non ha estensione)."""
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
# Normalizzazione video: rotazione + ritaglio letterbox                       #
# --------------------------------------------------------------------------- #
def _probe_rotation(video_path: Path) -> int:
    """
    Rotazione (gradi orari) da applicare per vedere il video dritto.
    Prova prima ffprobe (robusto), poi torna 0.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream_side_data=rotation:stream_tags=rotate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout
        vals = [int(float(v)) for v in out.split() if v.strip().lstrip("-").isdigit()]
        if vals:
            # ffprobe da' la rotazione del display (spesso negativa, es. -90).
            # La rotazione da APPLICARE e' l'opposto, normalizzata a [0,360).
            return (-vals[0]) % 360
    except Exception:
        pass
    return 0


def _detect_letterbox_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    """
    Trova il rettangolo di "contenuto" eliminando le bande nere.
    Restituisce (x1, y1, x2, y2) in pixel. Se non trova bande, l'intero frame.
    """
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    thr = config.LETTERBOX_BLACK_THRESHOLD
    min_fill = config.LETTERBOX_MIN_FILL

    col_fill = (gray > thr).mean(axis=0)   # frazione di pixel "accesi" per colonna
    row_fill = (gray > thr).mean(axis=1)   # ... per riga

    cols = np.where(col_fill > min_fill)[0]
    rows = np.where(row_fill > min_fill)[0]
    h, w = gray.shape[:2]
    if cols.size == 0 or rows.size == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def open_capture(video_path: Path):
    """
    Apre un VideoCapture chiedendo a OpenCV di applicare la rotazione dei
    metadati (auto-orientamento). Restituisce (cap, auto_applied), dove
    auto_applied dice se OpenCV gestira' la rotazione da solo.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    auto_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    auto_applied = False
    if auto_prop is not None:
        cap.set(auto_prop, 1)   # OpenCV moderno: raddrizza lui i frame
        auto_applied = True
    return cap, auto_applied


class VideoNormalizer:
    """
    Raddrizza (rotazione) e ritaglia (letterbox) i frame, una sola volta per
    video. Se OpenCV ha gia' applicato l'orientamento (auto_applied=True) NON
    ruota di nuovo; altrimenti (build vecchie) applica la rotazione dai metadati.
    """

    def __init__(self, video_path: Path, auto_applied: bool = True) -> None:
        import cv2
        self.cv2 = cv2

        # Rotazione manuale da applicare NOI (oltre a quella eventuale di OpenCV).
        if config.VIDEO_ROTATION == "auto":
            # OpenCV moderno raddrizza da solo -> 0; build vecchie -> dai metadati.
            self.rotation = 0 if auto_applied else _probe_rotation(video_path)
        else:
            self.rotation = int(config.VIDEO_ROTATION) % 360

        self._rot_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(self.rotation)

        self._crop: tuple[int, int, int, int] | None = None
        self._crop_ready = False

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        return self.cv2.rotate(frame, self._rot_code) if self._rot_code else frame

    def _ensure_crop(self, rotated: np.ndarray) -> None:
        if self._crop_ready:
            return
        mode = config.LETTERBOX_CROP
        h, w = rotated.shape[:2]
        if mode is None:
            self._crop = (0, 0, w, h)
        elif isinstance(mode, tuple):                       # coordinate normalizzate fisse
            x1, y1, x2, y2 = mode
            self._crop = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        else:                                               # "auto"
            self._crop = _detect_letterbox_bbox(rotated)
        self._crop_ready = True
        x1, y1, x2, y2 = self._crop
        logger.info("Normalizzazione: rotazione=%d°, crop=(%d,%d,%d,%d) da frame %dx%d",
                    self.rotation, x1, y1, x2, y2, w, h)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        rotated = self._rotate(frame)
        self._ensure_crop(rotated)
        x1, y1, x2, y2 = self._crop  # type: ignore[misc]
        return rotated[y1:y2, x1:x2]


def iter_sampled_frames(
    video_path: Path, frames_per_second: float, normalize: bool = True,
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Genera (timestamp_s, frame_BGR) campionando a `frames_per_second`.
    Se `normalize` e' True, raddrizza e ritaglia ogni frame.
    """
    import cv2

    cap, auto_applied = open_capture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native_fps / max(frames_per_second, 1e-6))), 1)
    normalizer = VideoNormalizer(video_path, auto_applied) if normalize else None

    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx / native_fps, (normalizer.apply(frame) if normalizer else frame)
            idx += 1
    finally:
        cap.release()


def get_frame_at(video_path: Path, timestamp_s: float, normalize: bool = True) -> np.ndarray:
    """Restituisce un singolo frame al tempo dato (normalizzato di default)."""
    import cv2

    cap, auto_applied = open_capture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Nessun frame a {timestamp_s}s in {video_path}")
        if normalize:
            frame = VideoNormalizer(video_path, auto_applied).apply(frame)
        return frame
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Eventi -> feature per il modello di prosodia (UNA sola codifica)             #
# --------------------------------------------------------------------------- #
def event_to_features(event_type: str, importance: float) -> np.ndarray:
    """
    Vettore di input del modello: [importanza, one-hot su config.EVENT_TYPES].
    Identico in training (Fase 3) e sintesi (Fase 4): ogni tipo ha la sua colonna.
    """
    one_hot = [1.0 if event_type == t else 0.0 for t in config.EVENT_TYPES]
    return np.asarray([importance, *one_hot], dtype=np.float32)


FEATURE_DIM: int = 1 + len(config.EVENT_TYPES)  # importanza + one-hot


# --------------------------------------------------------------------------- #
# Modello di prosodia (qui, per disaccoppiare Fase 3 e Fase 4)                 #
# --------------------------------------------------------------------------- #
def build_prosody_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int):
    """Crea l'MLP. Import locale di torch: pesante, solo quando serve."""
    import torch.nn as nn

    class ProsodyMLP(nn.Module):
        """MLP di regressione: feature evento -> [rate, pitch, energy]."""

        def __init__(self) -> None:
            super().__init__()
            layers: list = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return ProsodyMLP()


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
    Trasforma un'onda NEUTRA secondo i parametri prosodici.
    - rate_factor     > 1 accelera (telecronaca concitata).
    - pitch_semitones > 0 alza il tono.
    - energy_gain     > 1 aumenta il volume.
    Disaccoppia il CONTRIBUTO (mappatura evento->prosodia) dal motore TTS.
    """
    import pyworld as pw
    import scipy.interpolate

    # pyworld requires 64-bit float array
    x = waveform.astype(np.float64)

    # 1. Estrarre le caratteristiche vocali
    _f0, t = pw.dio(x, sample_rate)
    f0 = pw.stonemask(x, _f0, t, sample_rate)
    sp = pw.cheaptrick(x, f0, t, sample_rate)
    ap = pw.d4c(x, f0, t, sample_rate)

    # 2. Modifica del pitch (tono)
    if abs(pitch_semitones) > 1e-3:
        pitch_ratio = 2.0 ** (float(pitch_semitones) / 12.0)
        f0 = f0 * pitch_ratio

    # 3. Modifica della velocità (rate)
    if abs(rate_factor - 1.0) > 1e-3:
        # rate_factor > 1 accelera (durata minore)
        new_length = int(len(f0) / float(rate_factor))
        old_indices = np.arange(len(f0))
        new_indices = np.linspace(0, len(f0) - 1, new_length)
        
        f0_interp = scipy.interpolate.interp1d(old_indices, f0)(new_indices)
        sp_interp = scipy.interpolate.interp1d(old_indices, sp, axis=0)(new_indices)
        ap_interp = scipy.interpolate.interp1d(old_indices, ap, axis=0)(new_indices)
        
        f0 = np.ascontiguousarray(f0_interp)
        sp = np.ascontiguousarray(sp_interp)
        ap = np.ascontiguousarray(ap_interp)

    # 4. Risintetizzare l'audio
    out = pw.synthesize(f0, sp, ap, sample_rate, pw.default_frame_period)
    out = out.astype(np.float32)

    # 5. Modifica del volume
    out = out * float(energy_gain)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:                       # evita clipping
        out = out / peak
    return out
