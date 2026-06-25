"""
03_train_prosody.py
===================
FASE 3 - Modello evento -> prosodia. QUESTO E' IL CONTRIBUTO DI RICERCA.

Idea: un TTS generico legge tutto con tono piatto. Un vero telecronista modula la
voce in base all'IMPORTANZA dell'evento (esplode sul gol, calmo a centrocampo).
Qui APPRENDIAMO quella mappatura da telecronache reali, invece di codificarla a
mano.

Il modello e' un piccolo MLP che, dato l'evento (importanza + tipo), predice i
parametri prosodici [rate_factor, pitch_semitones, energy_gain] da applicare poi
all'audio (Fase 4).

Due passi:
    A) build-dataset : estrae i target prosodici da clip di telecronache reali
                       annotate (CSV), misurandoli con librosa rispetto a un
                       riferimento neutro.
    B) train         : addestra l'MLP sul dataset estratto.

Formato del CSV di annotazione (data/raw/prosody_annotations.csv):
    clip,start,end,event_type
    telecronaca1.wav,12.3,14.1,goal
    telecronaca1.wav,30.0,31.2,pass
    ...
('clip' e' un file in data/raw/commentary/; start/end in secondi.)

Uso:
    python src/03_train_prosody.py build-dataset
    python src/03_train_prosody.py train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from utils import (ensure_dir, event_to_features, get_logger, set_seed, FEATURE_DIM)

logger = get_logger("fase3_prosodia")


# --------------------------------------------------------------------------- #
# A) Costruzione del dataset prosodico dalle clip reali                        #
# --------------------------------------------------------------------------- #
class ProsodyMeasurer:
    """Misura i parametri prosodici di un segmento audio (rate, pitch, energia)."""

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        # Riferimenti "neutri" stimati dall'insieme dei segmenti (vedi calibrate()).
        self.ref_pitch_hz: float | None = None
        self.ref_energy: float | None = None
        self.ref_rate: float | None = None

    def _raw_measures(self, y: np.ndarray) -> tuple[float, float, float]:
        """Restituisce (pitch_hz_medio, energia_rms, tasso_sillabico_proxy)."""
        import librosa

        # Pitch medio sui frame sonori
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=self.sr
        )
        pitch = float(np.nanmean(f0)) if np.any(voiced) else 150.0

        # Energia RMS media
        energy = float(np.mean(librosa.feature.rms(y=y)))

        # Proxy della velocita' di eloquio: densita' di onset / durata
        onsets = librosa.onset.onset_detect(y=y, sr=self.sr, units="time")
        duration = max(len(y) / self.sr, 1e-6)
        rate = len(onsets) / duration

        return pitch, energy, rate

    def calibrate(self, all_measures: list[tuple[float, float, float]]) -> None:
        """Fissa i riferimenti neutri come mediana di tutti i segmenti."""
        arr = np.asarray(all_measures, dtype=np.float32)
        self.ref_pitch_hz = float(np.median(arr[:, 0]))
        self.ref_energy = float(np.median(arr[:, 1]))
        self.ref_rate = float(np.median(arr[:, 2]))
        logger.info("Riferimenti neutri -> pitch=%.1fHz energy=%.4f rate=%.2f",
                    self.ref_pitch_hz, self.ref_energy, self.ref_rate)

    def to_targets(self, measures: tuple[float, float, float]) -> np.ndarray:
        """Converte misure assolute in target relativi al neutro."""
        pitch, energy, rate = measures
        rate_factor = rate / max(self.ref_rate, 1e-6)
        # Semitoni = 12*log2(f/f_ref)
        pitch_semitones = 12.0 * np.log2(max(pitch, 1e-6) / max(self.ref_pitch_hz, 1e-6))
        energy_gain = energy / max(self.ref_energy, 1e-6)
        return np.asarray([rate_factor, pitch_semitones, energy_gain], dtype=np.float32)


def build_dataset() -> None:
    """Legge il CSV di annotazioni, misura la prosodia di ogni segmento e salva (X, y)."""
    import librosa
    import pandas as pd

    if not config.PROSODY_ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"Annotazioni mancanti: {config.PROSODY_ANNOTATIONS}\n"
            f"Crea il CSV (clip,start,end,event_type) descritto nell'header dello script."
        )

    df = pd.read_csv(config.PROSODY_ANNOTATIONS)
    measurer = ProsodyMeasurer(config.SAMPLE_RATE)

    # Pass 1: misura grezza di ogni segmento (serve per calibrare i riferimenti)
    raw, rows = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Misuro prosodia"):
        clip_path = config.COMMENTARY_DIR / r["clip"]
        try:
            y, _ = librosa.load(clip_path, sr=config.SAMPLE_RATE,
                                offset=float(r["start"]),
                                duration=float(r["end"]) - float(r["start"]))
            if y.size < config.SAMPLE_RATE // 10:  # < 0.1s: troppo corto
                raise ValueError("segmento troppo corto")
            m = measurer._raw_measures(y)
        except Exception as exc:
            logger.warning("Segmento saltato (%s @%.1f): %s", r["clip"], r["start"], exc)
            continue
        raw.append(m)
        rows.append(r["event_type"])

    if not raw:
        raise RuntimeError("Nessun segmento valido: controlla clip e annotazioni.")

    # Pass 2: calibra i riferimenti e converte in target relativi
    measurer.calibrate(raw)
    X = np.vstack([
        event_to_features(et, config.EVENT_IMPORTANCE.get(et, 0.1))
        for et in rows
    ]).astype(np.float32)
    y = np.vstack([measurer.to_targets(m) for m in raw]).astype(np.float32)

    ensure_dir(config.PROSODY_DATASET)
    np.savez(config.PROSODY_DATASET, X=X, y=y,
             ref_pitch=measurer.ref_pitch_hz, ref_energy=measurer.ref_energy,
             ref_rate=measurer.ref_rate)
    logger.info("Dataset prosodico salvato: X=%s y=%s -> %s",
                X.shape, y.shape, config.PROSODY_DATASET)


# --------------------------------------------------------------------------- #
# B) Modello e training                                                        #
# --------------------------------------------------------------------------- #
class ProsodyMLP(nn.Module):
    """MLP di regressione: feature evento -> [rate_factor, pitch_semitones, energy_gain]."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train() -> None:
    """Addestra l'MLP sul dataset prosodico estratto e salva modello + scaler."""
    if not config.PROSODY_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {config.PROSODY_DATASET}. "
            f"Esegui prima: python src/03_train_prosody.py build-dataset"
        )

    set_seed(config.RANDOM_SEED)
    data = np.load(config.PROSODY_DATASET)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
    logger.info("Carico dataset: X=%s y=%s", X.shape, y.shape)

    # Standardizziamo i target (scale diverse) per stabilizzare il training
    scaler = StandardScaler().fit(y)
    y_scaled = scaler.transform(y).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y_scaled)),
        batch_size=config.PROSODY_BATCH_SIZE, shuffle=True,
    )

    model = ProsodyMLP(FEATURE_DIM, config.PROSODY_HIDDEN_DIMS, len(config.PROSODY_TARGETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.PROSODY_LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, config.PROSODY_EPOCHS + 1):
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        if epoch % 25 == 0 or epoch == 1:
            logger.info("Epoch %3d | loss=%.4f", epoch, running / len(X))

    ensure_dir(config.PROSODY_MODEL_PATH)
    torch.save({"state_dict": model.state_dict(), "in_dim": FEATURE_DIM,
                "out_dim": len(config.PROSODY_TARGETS)}, config.PROSODY_MODEL_PATH)
    joblib.dump(scaler, config.PROSODY_SCALER_PATH)
    logger.info("Modello salvato -> %s", config.PROSODY_MODEL_PATH)
    logger.info("Scaler target salvato -> %s", config.PROSODY_SCALER_PATH)
    logger.info("Fase 3 completata.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3 - Modello evento->prosodia")
    parser.add_argument("command", choices=["build-dataset", "train"],
                        help="build-dataset: estrae i target dalle clip; train: addestra l'MLP.")
    args = parser.parse_args()

    if args.command == "build-dataset":
        build_dataset()
    else:
        train()


if __name__ == "__main__":
    main()
 