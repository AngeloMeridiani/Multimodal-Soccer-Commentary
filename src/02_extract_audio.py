"""
02_extract_audio.py
===================
FASE 2 della pipeline - Audio.

Per ogni clip elencata nel manifest estrae la traccia audio dal video, calcola
gli MFCC (con delta e delta-delta opzionali) tramite librosa e li riassume in un
vettore a lunghezza fissa con pooling statistico (media + deviazione standard
lungo l'asse temporale).

Input : features/manifest.csv  (id, label)  + data/raw/videos/<id>.mp4
Output: features/audio/audio_features.npy   -> dict {id: vettore}

Nota: librosa apre i .mp4 tramite il backend audioread, che richiede FFmpeg
installato nel sistema (vedi README).

Uso:
    python src/02_extract_audio.py
    python src/02_extract_audio.py --limit 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
from tqdm import tqdm

import config
from utils import get_logger, load_manifest, save_feature_dict, set_seed

logger = get_logger("fase2_audio")


class AudioFeatureExtractor:
    """Estrae un vettore MFCC a lunghezza fissa dalla traccia audio di un video."""

    def __init__(
        self, sample_rate: int, n_mfcc: int, include_deltas: bool
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.include_deltas = include_deltas

    def _load_audio(self, video_path: Path) -> np.ndarray:
        """Carica la forma d'onda mono dal video. Solleva eccezione se vuota."""
        waveform, _ = librosa.load(str(video_path), sr=self.sample_rate, mono=True)
        if waveform.size == 0:
            raise ValueError("Traccia audio vuota.")
        return waveform

    def extract(self, video_path: Path) -> np.ndarray:
        """
        Restituisce il vettore di feature acustiche per una clip.

        Pipeline: waveform -> MFCC (+ delta, delta2) -> pooling mean & std.
        Dimensione finale = n_mfcc * (1 o 3) * 2.
        """
        waveform = self._load_audio(video_path)

        mfcc = librosa.feature.mfcc(
            y=waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc
        )  # shape (n_mfcc, T)

        feature_stack = [mfcc]
        if self.include_deltas:
            feature_stack.append(librosa.feature.delta(mfcc))
            feature_stack.append(librosa.feature.delta(mfcc, order=2))

        stacked = np.vstack(feature_stack)  # (n_mfcc * k, T)

        # Pooling statistico sull'asse temporale -> vettore a lunghezza fissa
        pooled = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])
        return pooled.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2 - Feature audio MFCC")
    parser.add_argument("--limit", type=int, default=None,
                        help="Processa solo i primi N campioni (debug).")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    ids, _ = load_manifest(config.MANIFEST_PATH)
    if args.limit:
        ids = ids[: args.limit]
        logger.warning("Modalita' debug: limito a %d campioni.", len(ids))

    extractor = AudioFeatureExtractor(
        sample_rate=config.AUDIO_SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        include_deltas=config.INCLUDE_DELTAS,
    )

    features: Dict[str, np.ndarray] = {}
    failures = 0

    for uid in tqdm(ids, desc="Feature audio"):
        video_path = config.VIDEO_DIR / f"{uid}.mp4"
        try:
            if not video_path.exists():
                raise FileNotFoundError(f"File mancante: {video_path}")
            features[uid] = extractor.extract(video_path)
        except Exception as exc:
            # Un audio illeggibile non deve bloccare l'intera fase: log e si prosegue.
            failures += 1
            logger.warning("Audio fallito per '%s': %s", uid, exc)
            continue

    save_feature_dict(features, config.AUDIO_FEATURES_PATH)
    logger.info(
        "Salvati %d/%d vettori audio (%d falliti) in %s",
        len(features), len(ids), failures, config.AUDIO_FEATURES_PATH,
    )
    logger.info("Dimensione vettore audio: %d", config.AUDIO_EMBED_DIM)
    logger.info("Fase 2 completata.")


if __name__ == "__main__":
    main()