"""
03_train_prosody.py
===================
FASE 3 - Addestramento del modello di PROSODIA (IL CONTRIBUTO di ricerca).

Impara la mappa  feature_evento -> (rate_factor, pitch_semitones, energy_gain),
cioe' COME deve suonare la voce per ogni tipo di evento. E' l'unica parte con un
modello addestrato (un piccolo MLP); il resto della pipeline e' impalcatura.

Costruzione del dataset:
  1) Se ci sono clip di telecronaca reale annotate (config.PROSODY_ANNOTATIONS +
     audio in config.COMMENTARY_DIR), si MISURANO i target prosodici da quei
     segmenti (rate da densita' di onset, pitch da f0, energy da RMS), normalizzati
     rispetto alla mediana del dataset.
  2) Se non ci sono dati reali utilizzabili, si SINTETIZZA un dataset dai valori
     a regole (config.RULE_BASED_PROSODY) con un po' di rumore, cosi' la pipeline
     gira comunque end-to-end. (In tesi, sostituire con dati reali.)

Input  : feature da utils.event_to_features (UNICA codifica, coerente con la Fase 4).
Output : models/prosody_mlp.pt , models/prosody_scaler.joblib , dataset .npz

Uso:
    python 03_train_prosody.py
    python 03_train_prosody.py --synthetic     # forza il dataset sintetico
    python 03_train_prosody.py --epochs 300
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import config
from utils import (
    FEATURE_DIM,
    build_prosody_mlp,
    ensure_dir,
    event_to_features,
    get_logger,
    set_seed,
)

logger = get_logger("fase3_prosodia")


# --------------------------------------------------------------------------- #
# Misura dei target prosodici da audio reale                                  #
# --------------------------------------------------------------------------- #
def _measure_segment(audio: np.ndarray, sr: int) -> dict[str, float] | None:
    """Misure grezze (non normalizzate) di un segmento: rate, f0, energy."""
    import librosa

    if audio.size < sr // 5:
        return None
    rms = float(np.sqrt(np.mean(audio**2)))
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = len(audio) / sr
    rate = len(onsets) / duration if duration > 0 else 0.0
    try:
        f0, voiced, _ = librosa.pyin(audio, fmin=70, fmax=400, sr=sr)
        f0_med = float(np.nanmedian(f0[voiced])) if np.any(voiced) else np.nan
    except Exception:
        f0_med = np.nan
    return {"rate": rate, "f0": f0_med, "energy": rms}


def build_dataset_from_audio() -> tuple[np.ndarray, np.ndarray] | None:
    """Costruisce (X, Y) misurando i target dai segmenti annotati reali."""
    import librosa

    csv_path = config.PROSODY_ANNOTATIONS
    if not csv_path.exists():
        logger.warning("Annotazioni non trovate: %s", csv_path)
        return None

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    measures: list[dict] = []
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    for row in rows:
        clip_path = config.COMMENTARY_DIR / row["clip"]
        if not clip_path.exists():
            continue
        if row["clip"] not in audio_cache:
            audio_cache[row["clip"]] = librosa.load(
                str(clip_path), sr=config.SAMPLE_RATE, mono=True
            )
        audio, sr = audio_cache[row["clip"]]
        seg = audio[int(float(row["start"]) * sr) : int(float(row["end"]) * sr)]
        m = _measure_segment(seg, sr)
        if m is None:
            continue
        m["event_type"] = row["event_type"]
        measures.append(m)

    measures = [m for m in measures if not np.isnan(m.get("f0", np.nan))]
    if len(measures) < 4:
        logger.warning("Solo %d segmenti reali utilizzabili: insufficiente.", len(measures))
        return None

    med_rate = np.median([m["rate"] for m in measures]) + 1e-6
    med_f0 = np.median([m["f0"] for m in measures]) + 1e-6
    med_energy = np.median([m["energy"] for m in measures]) + 1e-6

    X, Y = [], []
    for m in measures:
        et = m["event_type"]
        rate_factor = float(np.clip(m["rate"] / med_rate, *config.PROSODY_CLAMP["rate_factor"]))
        pitch_semi = float(
            np.clip(12.0 * np.log2(m["f0"] / med_f0), *config.PROSODY_CLAMP["pitch_semitones"])
        )
        energy_gain = float(np.clip(m["energy"] / med_energy, *config.PROSODY_CLAMP["energy_gain"]))
        X.append(event_to_features(et, config.EVENT_IMPORTANCE.get(et, 0.1)))
        Y.append([rate_factor, pitch_semi, energy_gain])
    logger.info("Dataset da audio reale: %d campioni.", len(X))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def build_dataset_synthetic(n_per_type: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Dataset sintetico dai valori a regole + rumore (per far girare la pipeline)."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    X, Y = [], []
    for et, params in config.RULE_BASED_PROSODY.items():
        base = np.array(
            [params["rate_factor"], params["pitch_semitones"], params["energy_gain"]], np.float32
        )
        importance = config.EVENT_IMPORTANCE.get(et, 0.1)
        for _ in range(n_per_type):
            noise = rng.normal(0.0, [0.04, 0.4, 0.06]).astype(np.float32)
            target = base + noise
            for i, key in enumerate(config.PROSODY_TARGETS):
                target[i] = np.clip(target[i], *config.PROSODY_CLAMP[key])
            imp = float(np.clip(importance + rng.normal(0, 0.03), 0, 1))
            X.append(event_to_features(et, imp))
            Y.append(target)
    logger.info("Dataset sintetico: %d campioni (%d tipi).", len(X), len(config.RULE_BASED_PROSODY))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


# --------------------------------------------------------------------------- #
# Addestramento                                                               #
# --------------------------------------------------------------------------- #
def train(X: np.ndarray, Y: np.ndarray, epochs: int) -> None:
    import joblib
    import torch
    from sklearn.preprocessing import StandardScaler

    set_seed(config.RANDOM_SEED)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    n = len(Xs)
    idx = np.random.permutation(n)
    split = max(int(n * 0.85), 1)
    tr, va = idx[:split], idx[split:] if n > 1 else idx[:1]

    Xt = torch.tensor(Xs[tr])
    Yt = torch.tensor(Y[tr])
    Xv = torch.tensor(Xs[va])
    Yv = torch.tensor(Y[va])

    model = build_prosody_mlp(FEATURE_DIM, config.PROSODY_HIDDEN_DIMS, len(config.PROSODY_TARGETS))
    opt = torch.optim.Adam(model.parameters(), lr=config.PROSODY_LR)
    loss_fn = torch.nn.MSELoss()

    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), config.PROSODY_BATCH_SIZE):
            b = perm[i : i + config.PROSODY_BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xv), Yv))
        if val < best_val:
            best_val, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 50 == 0:
            logger.info("Epoca %3d/%d | val_loss=%.4f", epoch + 1, epochs, val)

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Miglior val_loss: %.4f", best_val)

    ensure_dir(config.PROSODY_MODEL_PATH)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": FEATURE_DIM,
            "hidden": list(config.PROSODY_HIDDEN_DIMS),
            "targets": config.PROSODY_TARGETS,
        },
        config.PROSODY_MODEL_PATH,
    )
    joblib.dump(scaler, config.PROSODY_SCALER_PATH)
    logger.info(
        "Modello -> %s | scaler -> %s", config.PROSODY_MODEL_PATH, config.PROSODY_SCALER_PATH
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3 - Addestramento prosodia")
    parser.add_argument("--synthetic", action="store_true", help="Forza dataset sintetico.")
    parser.add_argument("--epochs", type=int, default=config.PROSODY_EPOCHS)
    args = parser.parse_args()

    dataset = None if args.synthetic else build_dataset_from_audio()
    if dataset is None:
        logger.info("Uso dataset SINTETICO (dai valori a regole).")
        dataset = build_dataset_synthetic()
    X, Y = dataset

    ensure_dir(config.PROSODY_DATASET)
    np.savez(config.PROSODY_DATASET, X=X, Y=Y)
    logger.info("Dataset salvato (%s) -> %s", X.shape, config.PROSODY_DATASET)

    train(X, Y, args.epochs)
    logger.info("Fase 3 completata.")


if __name__ == "__main__":
    main()
