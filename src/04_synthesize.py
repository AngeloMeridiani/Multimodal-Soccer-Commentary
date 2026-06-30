"""
04_synthesize.py
================
FASE 4 - Sintesi audio: testo + prosodia -> voce espressiva.

Per ogni battuta:
  1) PROSODIA: il modello (Fase 3) predice (rate, pitch, energy) dalle feature
     dell'evento. Se il modello manca, fallback ai valori a regole.
  2) TTS: si genera una voce NEUTRA (pyttsx3 offline, o Coqui).
  3) DSP: utils.apply_prosody applica rate/pitch/energy all'onda neutra.
  4) Si concatena tutto con piccole pause -> traccia unica.

Il modello prosodico importa ProsodyMLP da utils (nessun accoppiamento al
nome-file della Fase 3).

Output: outputs/audio/<nome>.wav

Uso:
    python 04_synthesize.py --script features/scripts/match1.json
    python 04_synthesize.py --script features/scripts/match1.json --rule-based   # baseline A/B
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import config
from utils import (FEATURE_DIM, apply_prosody, build_prosody_mlp, ensure_dir,
                   event_to_features, get_logger, load_json)

logger = get_logger("fase4_sintesi")


# --------------------------------------------------------------------------- #
# Predizione prosodia                                                         #
# --------------------------------------------------------------------------- #
class ProsodyPredictor:
    """Predice (rate, pitch, energy) per evento: modello appreso o regole."""

    def __init__(self, force_rule_based: bool = False) -> None:
        self.model = None
        self.scaler = None
        if not force_rule_based:
            self._try_load_model()
        self.mode = "rule_based" if self.model is None else "learned"
        logger.info("Predittore prosodia: modalita' = %s", self.mode)

    def _try_load_model(self) -> None:
        if not config.PROSODY_MODEL_PATH.exists():
            logger.warning("Modello non trovato (%s): uso i valori a regole.",
                           config.PROSODY_MODEL_PATH)
            return
        try:
            import joblib
            import torch
            ckpt = torch.load(config.PROSODY_MODEL_PATH, map_location="cpu")
            self.model = build_prosody_mlp(ckpt.get("feature_dim", FEATURE_DIM),
                                           tuple(ckpt.get("hidden", config.PROSODY_HIDDEN_DIMS)),
                                           len(config.PROSODY_TARGETS))
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.scaler = joblib.load(config.PROSODY_SCALER_PATH)
        except Exception as exc:
            logger.warning("Caricamento modello fallito (%s): uso le regole.", exc)
            self.model = None

    def predict(self, event_type: str, importance: float) -> dict[str, float]:
        if self.model is not None and self.scaler is not None:
            import torch
            feat = event_to_features(event_type, importance).reshape(1, -1)
            feat = self.scaler.transform(feat).astype(np.float32)
            with torch.no_grad():
                out = self.model(torch.tensor(feat)).numpy().ravel()
            values = {k: float(out[i]) for i, k in enumerate(config.PROSODY_TARGETS)}
        else:
            values = dict(config.RULE_BASED_PROSODY.get(
                event_type, config.RULE_BASED_PROSODY["idle"]))
        # Clamp di sicurezza (vale per entrambe le modalita').
        for key in config.PROSODY_TARGETS:
            lo, hi = config.PROSODY_CLAMP[key]
            values[key] = float(np.clip(values[key], lo, hi))
        return values


# --------------------------------------------------------------------------- #
# Motori TTS (voce neutra)                                                     #
# --------------------------------------------------------------------------- #
class TTSEngine:
    def synth_neutral(self, text: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class GttsEngine(TTSEngine):
    """TTS tramite Google Translate API (online). Ottima qualità, 100% Python."""

    def __init__(self) -> None:
        logger.info("TTS: gTTS (online).")

    def synth_neutral(self, text: str) -> tuple[np.ndarray, int]:
        from gtts import gTTS
        import librosa
        tmp = Path(tempfile.mktemp(suffix=".mp3"))
        
        # Effettua la richiesta a Google
        tts = gTTS(text, lang='it')
        tts.save(str(tmp))

        try:
            # Librosa usa ffmpeg (o audioread) sotto al cofano per leggere l'MP3
            audio, sr = librosa.load(str(tmp), sr=config.SAMPLE_RATE, mono=True)
        finally:
            tmp.unlink(missing_ok=True)
        return audio.astype(np.float32), sr

class Pyttsx3Engine(TTSEngine):
    """TTS offline. Su Windows il motore SAPI5 si pianta se riusato in un
    ciclo: lo ricreiamo a ogni battuta (piu' lento ma robusto)."""

    def __init__(self) -> None:
        import pyttsx3
        self._pyttsx3 = pyttsx3
        logger.info("TTS: pyttsx3 (offline).")

    def synth_neutral(self, text: str) -> tuple[np.ndarray, int]:
        import librosa
        import subprocess
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        
        # Bypass pyttsx3 buggy bindings su Linux e usa espeak CLI direttamente
        subprocess.run(["espeak-ng", "-w", str(tmp), "-v", "it", text], check=True)

        try:
            audio, sr = librosa.load(str(tmp), sr=config.SAMPLE_RATE, mono=True)
        finally:
            tmp.unlink(missing_ok=True)
        return audio.astype(np.float32), sr

class CoquiEngine(TTSEngine):
    """TTS espressivo (Coqui XTTS). Voce piu' naturale, piu' pesante."""

    def __init__(self) -> None:
        from TTS.api import TTS
        logger.info("TTS: Coqui (%s).", config.COQUI_MODEL)
        self.tts = TTS(config.COQUI_MODEL)

    def synth_neutral(self, text: str) -> tuple[np.ndarray, int]:
        kwargs = {"text": text, "language": config.COQUI_LANGUAGE}
        if config.COQUI_SPEAKER_WAV:
            kwargs["speaker_wav"] = config.COQUI_SPEAKER_WAV
        wav = np.asarray(self.tts.tts(**kwargs), dtype=np.float32)
        sr = getattr(self.tts.synthesizer, "output_sample_rate", config.SAMPLE_RATE)
        if sr != config.SAMPLE_RATE:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        return wav, config.SAMPLE_RATE


def make_tts_engine(name: str) -> TTSEngine:
    if name == "coqui":
        return CoquiEngine()
    if name == "gtts":
        return GttsEngine()
    return Pyttsx3Engine()


# --------------------------------------------------------------------------- #
# Sintesi della traccia                                                       #
# --------------------------------------------------------------------------- #
def synthesize(script: list[dict], predictor: ProsodyPredictor, tts: TTSEngine) -> np.ndarray:
    sr = config.SAMPLE_RATE
    gap = np.zeros(int(config.GAP_BETWEEN_UTTERANCES_S * sr), np.float32)
    pieces: list[np.ndarray] = []

    for i, line in enumerate(script):
        text = line.get("text", "").strip()
        if not text:
            continue
        prosody = predictor.predict(line.get("event_type", "idle"),
                                    line.get("importance", 0.1))
        try:
            neutral, _ = tts.synth_neutral(text)
            voiced = apply_prosody(neutral, sr,
                                   prosody["rate_factor"], prosody["pitch_semitones"],
                                   prosody["energy_gain"])
            pieces.append(voiced)
            pieces.append(gap)
        except Exception as exc:
            logger.warning("Battuta %d saltata (%s): %s", i + 1, text[:30], exc)
        if (i + 1) % 10 == 0:
            logger.info("Sintetizzate %d/%d battute.", i + 1, len(script))

    if not pieces:
        return np.zeros(sr, np.float32)
    track = np.concatenate(pieces).astype(np.float32)
    peak = float(np.max(np.abs(track))) or 1.0
    return track / peak * 0.95   # normalizzazione finale


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 4 - Sintesi audio")
    parser.add_argument("--script", required=True, help="JSON dello script (Fase 2/2b).")
    parser.add_argument("--rule-based", action="store_true",
                        help="Forza la prosodia a regole (condizione baseline dello studio).")
    parser.add_argument("--engine", choices=["gtts", "pyttsx3", "coqui"], default=config.TTS_ENGINE)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    import soundfile as sf
    script_path = Path(args.script)
    script = load_json(script_path)
    logger.info("Caricate %d battute da %s", len(script), script_path)

    predictor = ProsodyPredictor(force_rule_based=args.rule_based)
    tts = make_tts_engine(args.engine)
    track = synthesize(script, predictor, tts)

    suffix = "_rulebased" if args.rule_based else "_model"
    out_path = Path(args.out) if args.out else config.AUDIO_OUT_DIR / f"{script_path.stem}{suffix}.wav"
    ensure_dir(out_path)
    sf.write(str(out_path), track, config.SAMPLE_RATE)
    logger.info("Traccia salvata (%.1fs) -> %s", len(track) / config.SAMPLE_RATE, out_path)
    logger.info("Fase 4 completata.")


if __name__ == "__main__":
    main()
