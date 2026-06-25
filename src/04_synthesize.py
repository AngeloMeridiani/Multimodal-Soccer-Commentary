"""
04_synthesize.py
================
FASE 4 - Sintesi della telecronaca audio.

Per ogni battuta dello script (Fase 2):
    1. genera l'audio NEUTRO con un TTS;
    2. predice i parametri prosodici in base all'evento;
    3. applica la prosodia all'audio (time-stretch / pitch / gain);
    4. concatena tutte le battute in un'unica traccia.

Tre MODALITA' (selezionabili con --mode), che sono anche le condizioni dello
studio sugli ascoltatori (Fase 5):
    - flat    : nessuna prosodia (baseline piatta).
    - rule    : prosodia da regole scritte a mano (config.RULE_BASED_PROSODY).
    - learned : prosodia predetta dal modello addestrato (IL CONTRIBUTO).

Output: outputs/audio/<nome>_<mode>.wav

Uso:
    python src/04_synthesize.py --script features/scripts/match1.json --mode learned
    python src/04_synthesize.py --script features/scripts/match1.json --mode flat
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import config
from utils import (apply_prosody, ensure_dir, event_to_features, get_logger,
                   load_json, FEATURE_DIM)

logger = get_logger("fase4_sintesi")


# --------------------------------------------------------------------------- #
# Motore TTS (astratto: facile da sostituire)                                 #
# --------------------------------------------------------------------------- #
class TtsEngine:
    """
    Interfaccia base per i motori TTS. Produce audio NEUTRO da testo.

    La qualita'/lo stile vocale ("dark hero", ecc.) si ottengono sostituendo
    il motore con un TTS espressivo (es. Coqui XTTS) senza toccare il resto
    della pipeline: il contributo (la prosodia) e' indipendente dal TTS.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        raise NotImplementedError


class Pyttsx3Engine(TtsEngine):
    """TTS offline basato su pyttsx3. Semplice e leggero, voce robotica."""

    def synthesize(self, text: str) -> np.ndarray:
        import librosa
        import pyttsx3

        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            waveform, _ = librosa.load(tmp_path, sr=self.sr, mono=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return waveform.astype(np.float32)


class CoquiTtsEngine(TtsEngine):
    """
    TTS espressivo basato su Coqui XTTS v2.

    Offre voci molto piu' naturali di pyttsx3, con supporto multilingue
    (incluso italiano) e possibilita' di clonare lo stile vocale da un
    campione audio di riferimento (speaker_wav).

    Richiede: pip install TTS
    """

    def __init__(self, sample_rate: int) -> None:
        super().__init__(sample_rate)
        self._model = None

    def _load_model(self):
        """Lazy loading del modello (pesante, ~2GB)."""
        if self._model is None:
            from TTS.api import TTS as CoquiTTS
            logger.info("Carico Coqui TTS '%s'...", config.COQUI_MODEL)
            self._model = CoquiTTS(model_name=config.COQUI_MODEL)
        return self._model

    def synthesize(self, text: str) -> np.ndarray:
        import librosa

        model = self._load_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # XTTS v2: supporta speaker_wav per voice cloning dello stile
            kwargs = {
                "text": text,
                "file_path": tmp_path,
                "language": config.COQUI_LANGUAGE,
            }
            if config.COQUI_SPEAKER_WAV and Path(config.COQUI_SPEAKER_WAV).exists():
                kwargs["speaker_wav"] = config.COQUI_SPEAKER_WAV

            model.tts_to_file(**kwargs)
            waveform, _ = librosa.load(tmp_path, sr=self.sr, mono=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return waveform.astype(np.float32)


def create_tts_engine(engine_name: str | None = None) -> TtsEngine:
    """Factory: crea il motore TTS in base alla configurazione."""
    engine = engine_name or config.TTS_ENGINE
    sr = config.SAMPLE_RATE

    if engine == "coqui":
        try:
            import TTS  # noqa: F401
            logger.info("Motore TTS: Coqui XTTS (espressivo)")
            return CoquiTtsEngine(sr)
        except ImportError:
            logger.warning("Coqui TTS non installato. Fallback a pyttsx3. "
                           "Installa con: pip install TTS")
            return Pyttsx3Engine(sr)

    logger.info("Motore TTS: pyttsx3 (offline)")
    return Pyttsx3Engine(sr)


# --------------------------------------------------------------------------- #
# Predittori di prosodia                                                       #
# --------------------------------------------------------------------------- #
class ProsodyPredictor:
    """Restituisce i parametri prosodici per un evento, secondo la modalita' scelta."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.model = None
        self.scaler = None
        if mode == "learned":
            self._load_model()

    def _load_model(self) -> None:
        import joblib
        import torch
        from importlib import import_module

        if not config.PROSODY_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modello prosodia non trovato: {config.PROSODY_MODEL_PATH}. "
                f"Addestralo (Fase 3) o usa --mode rule."
            )
        ProsodyMLP = getattr(import_module("03_train_prosody"), "ProsodyMLP")
        ckpt = torch.load(config.PROSODY_MODEL_PATH, map_location="cpu")
        self.model = ProsodyMLP(ckpt["in_dim"], config.PROSODY_HIDDEN_DIMS, ckpt["out_dim"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.scaler = joblib.load(config.PROSODY_SCALER_PATH)

    @staticmethod
    def _clamp(params: dict[str, float]) -> dict[str, float]:
        """Limita i parametri al range di sicurezza (evita audio innaturale)."""
        out = {}
        for key, val in params.items():
            lo, hi = config.PROSODY_CLAMP[key]
            out[key] = float(np.clip(val, lo, hi))
        return out

    def predict(self, event_type: str, importance: float) -> dict[str, float]:
        if self.mode == "flat":
            return {"rate_factor": 1.0, "pitch_semitones": 0.0, "energy_gain": 1.0}

        if self.mode == "rule":
            return self._clamp(dict(config.RULE_BASED_PROSODY[event_type]))

        # learned
        import torch
        feats = event_to_features(event_type, importance).reshape(1, -1)
        with torch.no_grad():
            scaled = self.model(torch.from_numpy(feats)).numpy()
        values = self.scaler.inverse_transform(scaled)[0]
        params = dict(zip(config.PROSODY_TARGETS, values))
        return self._clamp(params)


# --------------------------------------------------------------------------- #
# Sintesi completa                                                            #
# --------------------------------------------------------------------------- #
def synthesize_commentary(script: list[dict], mode: str, tts_engine: str | None = None) -> np.ndarray:
    tts = create_tts_engine(tts_engine)
    predictor = ProsodyPredictor(mode)
    gap = np.zeros(int(config.GAP_BETWEEN_UTTERANCES_S * config.SAMPLE_RATE), dtype=np.float32)

    from tqdm import tqdm
    segments: list[np.ndarray] = []
    for utt in tqdm(script, desc=f"Sintesi [{mode}]"):
        try:
            neutral = tts.synthesize(utt["text"])
            params = predictor.predict(utt["event_type"], utt["importance"])
            expressive = apply_prosody(neutral, config.SAMPLE_RATE, **params)
            segments.append(expressive)
            segments.append(gap)
        except Exception as exc:
            logger.warning("Battuta saltata (\"%s\"): %s", utt.get("text", ""), exc)
            continue

    return np.concatenate(segments) if segments else np.zeros(1, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 4 - Sintesi telecronaca audio")
    parser.add_argument("--script", required=True, help="JSON script della Fase 2/2b.")
    parser.add_argument("--mode", choices=["flat", "rule", "learned"], default="learned")
    parser.add_argument("--tts", choices=["pyttsx3", "coqui"], default=None,
                        help="Motore TTS (default: da config.TTS_ENGINE).")
    args = parser.parse_args()

    import soundfile as sf

    script_path = Path(args.script)
    script = load_json(script_path)

    audio = synthesize_commentary(script, args.mode, tts_engine=args.tts)

    out_path = config.AUDIO_OUT_DIR / f"{script_path.stem}_{args.mode}.wav"
    ensure_dir(out_path)
    sf.write(out_path, audio, config.SAMPLE_RATE)
    logger.info("Telecronaca [%s] salvata (%.1fs) -> %s",
                args.mode, len(audio) / config.SAMPLE_RATE, out_path)
    logger.info("Fase 4 completata.")


if __name__ == "__main__":
    main()
