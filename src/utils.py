"""
utils.py
========
Utilities shared by all phases.

- get_logger / set_seed / ensure_dir / save_json / load_json : infrastructure.
- VideoNormalizer + iter_sampled_frames : reads the video STRAIGHTENED and CROPPED
  (fixes rotation ignored by OpenCV + black side bars).
- event_to_features / FEATURE_DIM : event -> input vector of the prosody model
  (importance + SINGLE one-hot over config.EVENT_TYPES).
- ProsodyMLP : the network (here, so Phase 3 and Phase 4 import it without
  coupling to the file name).
- apply_prosody : applies the prosody parameters to a neutral audio wave (DSP).
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
# Infrastructure                                                              #
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """
    Initialize and return a logger with the given name.
    Sets up a stream handler with a standard timestamped format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("utils")


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across random, numpy, and torch.
    """
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
    """Create the folder of `path` (or `path` itself if it has no extension)."""
    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# JSON I/O                                                                     #
# --------------------------------------------------------------------------- #
def save_json(obj, path: Path) -> None:
    """
    Save an object as a JSON file, ensuring the target directory exists.
    """
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    """
    Load and return the content of a JSON file.
    Raises FileNotFoundError if the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Video normalization: rotation + letterbox crop                              #
# --------------------------------------------------------------------------- #
def _probe_rotation(video_path: Path) -> int:
    """
    Rotation (clockwise degrees) to apply to see the video upright.
    Tries ffprobe first (robust), then falls back to 0.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_side_data=rotation:stream_tags=rotate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        vals = [int(float(v)) for v in out.split() if v.strip().lstrip("-").isdigit()]
        if vals:
            # ffprobe gives the display rotation (often negative, e.g. -90).
            # The rotation to APPLY is the opposite, normalized to [0,360).
            return (-vals[0]) % 360
    except Exception:
        pass
    return 0


def _detect_letterbox_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the "content" rectangle by removing the black bars.
    Returns (x1, y1, x2, y2) in pixels. If it finds no bars, the whole frame.
    """
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    thr = config.LETTERBOX_BLACK_THRESHOLD
    min_fill = config.LETTERBOX_MIN_FILL

    col_fill = (gray > thr).mean(axis=0)  # fraction of "lit" pixels per column
    row_fill = (gray > thr).mean(axis=1)  # ... per row

    cols = np.where(col_fill > min_fill)[0]
    rows = np.where(row_fill > min_fill)[0]
    h, w = gray.shape[:2]
    if cols.size == 0 or rows.size == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def open_capture(video_path: Path):
    """
    Open a VideoCapture asking OpenCV to apply the metadata rotation
    (auto-orientation). Returns (cap, auto_applied), where auto_applied says
    whether OpenCV will handle the rotation by itself.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    auto_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    auto_applied = False
    if auto_prop is not None:
        cap.set(auto_prop, 1)  # modern OpenCV: it straightens the frames
        auto_applied = True
    return cap, auto_applied


class VideoNormalizer:
    """
    Straightens (rotation) and crops (letterbox) the frames, once per video.
    If OpenCV already applied the orientation (auto_applied=True) it does NOT
    rotate again; otherwise (old builds) it applies the rotation from the metadata.
    """

    def __init__(self, video_path: Path, auto_applied: bool = True) -> None:
        """
        Initialize the normalizer. Calculates the required rotation 
        from video metadata if OpenCV has not already applied it.
        """
        import cv2

        self.cv2 = cv2

        # Manual rotation for US to apply (on top of OpenCV's, if any).
        if config.VIDEO_ROTATION == "auto":
            # modern OpenCV straightens by itself -> 0; old builds -> from metadata.
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
        """
        Apply rotation to the frame if a rotation code was computed.
        """
        return self.cv2.rotate(frame, self._rot_code) if self._rot_code else frame

    def _ensure_crop(self, rotated: np.ndarray) -> None:
        """
        Determine the crop boundaries on the first frame and store them for reuse.
        """
        if self._crop_ready:
            return
        mode = config.LETTERBOX_CROP
        h, w = rotated.shape[:2]
        if mode is None:
            self._crop = (0, 0, w, h)
        elif isinstance(mode, tuple):  # fixed normalized coordinates
            x1, y1, x2, y2 = mode
            self._crop = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        else:  # "auto"
            self._crop = _detect_letterbox_bbox(rotated)
        self._crop_ready = True
        x1, y1, x2, y2 = self._crop
        logger.info(
            "Normalizzazione: rotazione=%d°, crop=(%d,%d,%d,%d) da frame %dx%d",
            self.rotation,
            x1,
            y1,
            x2,
            y2,
            w,
            h,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate and crop the frame to remove black letterboxes.
        """
        rotated = self._rotate(frame)
        self._ensure_crop(rotated)
        x1, y1, x2, y2 = self._crop  # type: ignore[misc]
        return rotated[y1:y2, x1:x2]


def iter_sampled_frames(
    video_path: Path,
    frames_per_second: float,
    normalize: bool = True,
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Yields (timestamp_s, frame_BGR) sampling at `frames_per_second`.
    If `normalize` is True, straightens and crops every frame.
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
    """Return a single frame at the given time (normalized by default)."""
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
# Events -> features for the prosody model (ONE single encoding)              #
# --------------------------------------------------------------------------- #
def event_to_features(event_type: str, importance: float, audio_energy: float = 0.5) -> np.ndarray:
    """
    Model input vector: [importance, audio_energy, one-hot over
    config.EVENT_TYPES]. Identical in training (Phase 3) and synthesis (Phase 4):
    each type has its own column.

    audio_energy is the REAL crowd excitement measured by Phase 1c
    (AudioEnergyAnalyzer: RMS + spectral centroid + onset, in [0,1]) in the
    event's audio window. Before, it was computed and then ignored by the
    prosody model (it only ended up in Phase 2b's text prompt): now it is a
    real feature, so the prosody responds to HOW MUCH the crowd reacted at that
    specific moment, not only to the abstract event type. Default 0.5 (neutral)
    when Phase 1c was not run.
    """
    one_hot = [1.0 if event_type == t else 0.0 for t in config.EVENT_TYPES]
    return np.asarray([importance, audio_energy, *one_hot], dtype=np.float32)


FEATURE_DIM: int = 2 + len(config.EVENT_TYPES)  # importance + audio_energy + one-hot


# --------------------------------------------------------------------------- #
# Prosody model (here, to decouple Phase 3 and Phase 4)                       #
# --------------------------------------------------------------------------- #
def build_prosody_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int):
    """Create the MLP. Local torch import: heavy, only when needed."""
    import torch.nn as nn

    class ProsodyMLP(nn.Module):
        """Regression MLP: event feature -> [rate, pitch, energy]."""

        def __init__(self) -> None:
            """
            Initialize the MLP with the given hidden dimensions.
            """
            super().__init__()
            layers: list = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            """
            Forward pass of the MLP to predict prosody parameters.
            """
            return self.net(x)

    return ProsodyMLP()


# --------------------------------------------------------------------------- #
# Applying the prosody to the neutral audio (DSP)                             #
# --------------------------------------------------------------------------- #
def apply_prosody(
    waveform: np.ndarray,
    sample_rate: int,
    rate_factor: float,
    pitch_semitones: float,
    energy_gain: float,
) -> np.ndarray:
    """
    Transform a NEUTRAL wave according to the prosody parameters.
    - rate_factor     > 1 speeds up (excited commentary).
    - pitch_semitones > 0 raises the pitch.
    - energy_gain     > 1 increases the volume.
    Decouples the CONTRIBUTION (event->prosody mapping) from the TTS engine.
    """
    import pyworld as pw
    import scipy.interpolate

    # pyworld requires 64-bit float array
    x = waveform.astype(np.float64)

    # 1. Extract the vocal features
    _f0, t = pw.dio(x, sample_rate)
    f0 = pw.stonemask(x, _f0, t, sample_rate)
    sp = pw.cheaptrick(x, f0, t, sample_rate)
    ap = pw.d4c(x, f0, t, sample_rate)

    # 2. Pitch modification
    if abs(pitch_semitones) > 1e-3:
        pitch_ratio = 2.0 ** (float(pitch_semitones) / 12.0)
        f0 = f0 * pitch_ratio

    # 3. Rate (speed) modification
    if abs(rate_factor - 1.0) > 1e-3:
        # rate_factor > 1 speeds up (shorter duration)
        new_length = int(len(f0) / float(rate_factor))
        old_indices = np.arange(len(f0))
        new_indices = np.linspace(0, len(f0) - 1, new_length)

        f0_interp = scipy.interpolate.interp1d(old_indices, f0)(new_indices)
        sp_interp = scipy.interpolate.interp1d(old_indices, sp, axis=0)(new_indices)
        ap_interp = scipy.interpolate.interp1d(old_indices, ap, axis=0)(new_indices)

        f0 = np.ascontiguousarray(f0_interp)
        sp = np.ascontiguousarray(sp_interp)
        ap = np.ascontiguousarray(ap_interp)

    # 4. Re-synthesize the audio
    out = pw.synthesize(f0, sp, ap, sample_rate, pw.default_frame_period)
    out = out.astype(np.float32)

    # 5. Volume modification
    out = out * float(energy_gain)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:  # avoid clipping
        out = out / peak
    return out
