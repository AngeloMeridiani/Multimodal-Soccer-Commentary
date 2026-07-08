"""
03_train_prosody.py
===================
PHASE 3 - Training of the PROSODY model (THE research contribution).

Learns the map  event_feature -> (rate_factor, pitch_semitones, energy_gain),
i.e. HOW the voice should sound for each event type. It is the only part with a
trained model (a small MLP); the rest of the pipeline is scaffolding.

Dataset construction:
  1) If there are annotated real commentary clips (config.PROSODY_ANNOTATIONS +
     audio in config.COMMENTARY_DIR), the prosody targets are MEASURED from those
     segments (rate from onset density, pitch from f0, energy from RMS), normalized
     against the dataset median.
  2) If there is no usable real data, a dataset is SYNTHESIZED from the rule-based
     values (config.RULE_BASED_PROSODY) with a bit of noise, so the pipeline runs
     end-to-end anyway. (In the thesis, replace with real data.)

Input  : features from utils.event_to_features (SINGLE encoding, consistent with Phase 4).
Output : models/prosody_mlp.pt , models/prosody_scaler.joblib , dataset .npz

Usage:
    python 03_train_prosody.py
    python 03_train_prosody.py --synthetic     # force the synthetic dataset
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
# Measuring the prosody targets from real audio                               #
# --------------------------------------------------------------------------- #
def _measure_segment(audio: np.ndarray, sr: int) -> dict[str, float] | None:
    """Raw (non-normalized) measures of a segment: rate, f0, energy."""
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
    """Build (X, Y) by measuring the targets from the real annotated segments."""
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
        # NB: the annotated clips are COMMENTARY segments (the commentator's voice),
        # not gameplay audio with the crowd: we do not have a real audio_energy
        # measure for these samples. We use the neutral value (0.5); the real crowd
        # signal only comes from the synthetic dataset until GAMEPLAY clips (with
        # crowd) aligned to the same events are annotated.
        X.append(event_to_features(et, config.EVENT_IMPORTANCE.get(et, 0.1), audio_energy=0.5))
        Y.append([rate_factor, pitch_semi, energy_gain])
    logger.info("Dataset da audio reale: %d campioni.", len(X))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def build_dataset_synthetic(n_per_type: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset from the rule-based values + noise (to run the pipeline).

    audio_energy is a synthetic PROXY of importance with noise: it models the
    idea that the crowd usually reacts in proportion to the event, but not
    always (sometimes an "important" event goes unnoticed, or a minor event
    triggers a roar). The target is nudged by the GAP between real energy and
    expected importance, so the model learns to use audio_energy as an
    independent signal (not just to ignore it as noise). It is a stated
    limitation: the true relation must be learned from real annotated data
    (gameplay audio + commentary), not from this synthetic correlation.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    X, Y = [], []
    for et, params in config.RULE_BASED_PROSODY.items():
        base = np.array(
            [params["rate_factor"], params["pitch_semitones"], params["energy_gain"]], np.float32
        )
        importance = config.EVENT_IMPORTANCE.get(et, 0.1)
        for _ in range(n_per_type):
            imp = float(np.clip(importance + rng.normal(0, 0.03), 0, 1))
            energy = float(np.clip(importance + rng.normal(0, 0.15), 0, 1))
            delta = energy - importance  # how much the crowd reacted MORE/LESS than expected

            noise = rng.normal(0.0, [0.04, 0.4, 0.06]).astype(np.float32)
            target = base + noise
            target[0] += 0.06 * delta  # rate_factor: more hyped crowd -> quicker pace
            target[2] += 0.20 * delta  # energy_gain: more hyped crowd -> more intense voice
            for i, key in enumerate(config.PROSODY_TARGETS):
                target[i] = np.clip(target[i], *config.PROSODY_CLAMP[key])

            X.append(event_to_features(et, imp, audio_energy=energy))
            Y.append(target)
    logger.info("Dataset sintetico: %d campioni (%d tipi).", len(X), len(config.RULE_BASED_PROSODY))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def train(X: np.ndarray, Y: np.ndarray, epochs: int) -> None:
    """
    Train the multi-layer perceptron (MLP) mapping event features to prosody targets.
    Scales the input data, performs a train/val split, and saves the best model.
    """
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
    """
    Main entry point for Phase 3.
    Constructs the dataset (from real audio or synthetically), trains the prosody MLP, and saves the artifacts.
    """
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
