"""
04_synthesize.py
================
PHASE 4 - Audio synthesis: text + prosody -> expressive voice.

Pipeline (Coqui XTTS v2, zero-shot voice cloning):
  1) PROSODY: prosody_mlp (Phase 3) predicts the emphasis per event; here we use
     rate_factor -> XTTS speed and energy_gain -> volume. (Fallback: rules.)
  2) BLOCKS: the lines are merged into blocks and synthesized in a single
     pass -> connected prosody, with the voice cloned from a reference audio.
  3) POLISH: silence trim + fade-in, then concatenation -> track.

The prosody model imports ProsodyMLP from utils (no coupling to the Phase 3
file name).

Output: outputs/audio/<name>.wav

Usage:
    python 04_synthesize.py --script features/scripts/match1.json
    python 04_synthesize.py --script features/scripts/match1.json --rule-based   # A/B baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config
from utils import (
    FEATURE_DIM,
    build_prosody_mlp,
    ensure_dir,
    event_to_features,
    get_logger,
    load_json,
)

logger = get_logger("fase4_sintesi")


# --------------------------------------------------------------------------- #
# Prosody prediction                                                          #
# --------------------------------------------------------------------------- #
class ProsodyPredictor:
    """Predicts (rate, pitch, energy) per event: learned model or rules."""

    def __init__(self, force_rule_based: bool = False) -> None:
        """
        Initialize the prosody predictor.
        Attempts to load the learned model unless rule-based is forced.
        """
        self.model = None
        self.scaler = None
        if not force_rule_based:
            self._try_load_model()
        self.mode = "rule_based" if self.model is None else "learned"
        logger.info("Predittore prosodia: modalita' = %s", self.mode)

    def _try_load_model(self) -> None:
        """
        Load the PyTorch model and the scikit-learn scaler from disk.
        Sets self.model to None on failure.
        """
        if not config.PROSODY_MODEL_PATH.exists():
            logger.warning(
                "Modello non trovato (%s): uso i valori a regole.", config.PROSODY_MODEL_PATH
            )
            return
        try:
            import joblib
            import torch

            ckpt = torch.load(config.PROSODY_MODEL_PATH, map_location="cpu")
            self.model = build_prosody_mlp(
                ckpt.get("feature_dim", FEATURE_DIM),
                tuple(ckpt.get("hidden", config.PROSODY_HIDDEN_DIMS)),
                len(config.PROSODY_TARGETS),
            )
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.scaler = joblib.load(config.PROSODY_SCALER_PATH)
        except Exception as exc:
            logger.warning("Caricamento modello fallito (%s): uso le regole.", exc)
            self.model = None

    def predict(
        self, event_type: str, importance: float, audio_energy: float = 0.5
    ) -> dict[str, float]:
        """
        Predict the target prosody values (rate, pitch, energy) for an event.
        Uses the MLP if loaded, otherwise falls back to configured rule-based values.
        """
        if self.model is not None and self.scaler is not None:
            import torch

            feat = event_to_features(event_type, importance, audio_energy).reshape(1, -1)
            feat = self.scaler.transform(feat).astype(np.float32)
            with torch.no_grad():
                out = self.model(torch.tensor(feat)).numpy().ravel()
            values = {k: float(out[i]) for i, k in enumerate(config.PROSODY_TARGETS)}
        else:
            values = dict(
                config.RULE_BASED_PROSODY.get(event_type, config.RULE_BASED_PROSODY["idle"])
            )
        # Safety clamp (applies to both modes).
        for key in config.PROSODY_TARGETS:
            lo, hi = config.PROSODY_CLAMP[key]
            values[key] = float(np.clip(values[key], lo, hi))
        return values


# --------------------------------------------------------------------------- #
# TTS engines (neutral voice)                                                 #
# --------------------------------------------------------------------------- #
class TTSEngine:
    """TTS engine interface. Single implementation: CoquiEngine (XTTS v2)."""

    output_sample_rate: int = config.SAMPLE_RATE  # rate of the produced audio

    def synth_neutral(
        self, text: str, speed: float | None = None, excited: bool = False
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize text into audio using the specified speed and emotion.
        Returns the audio array and its sample rate.
        """
        raise NotImplementedError


class CoquiEngine(TTSEngine):
    """Expressive TTS (Coqui XTTS v2). Natively multilingual (Italian included)
    with zero-shot voice cloning from speaker_wav. Model loaded ONCE."""

    def __init__(self) -> None:
        """
        Initialize the Coqui XTTS v2 engine and resolve reference speaker audio paths.
        """
        import os

        # XTTS v2 is under the CPML license: without this, on the first download
        # the loader asks for interactive confirmation and hangs in batch mode.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS

        logger.info("TTS: Coqui (%s).", config.COQUI_MODEL)
        self.tts = TTS(config.COQUI_MODEL)
        # Rate of the produced audio: XTTS's native 24 kHz (config), NOT the 22050
        # of the audio analysis -> no downsampling that "muffles" the speech.
        self.output_sample_rate = config.COQUI_OUTPUT_SAMPLE_RATE
        # Resolve the references against the project root (robust to the cwd).
        # XTTS v2 REQUIRES a speaker_wav (it has no built-in voices). If the
        # current language has no dedicated wav, the Italian one is used as
        # fallback (the voice will have a slight accent but will speak correctly
        # in the target language thanks to the language parameter).
        raw_wav = config.COQUI_SPEAKER_WAV
        if not raw_wav:
            # Fallback: first wav available among the configured languages.
            for wav in config.COQUI_SPEAKER_WAVS.values():
                if wav:
                    raw_wav = wav
                    logger.info("Nessun speaker_wav per '%s': uso fallback '%s'.", config.LANGUAGE, wav)
                    break
        self.speaker_wav = str(config.PROJECT_ROOT / raw_wav) if raw_wav else None

        raw_excited = config.COQUI_SPEAKER_WAV_EXCITED
        if not raw_excited:
            for wav in config.COQUI_SPEAKER_WAVS_EXCITED.values():
                if wav:
                    raw_excited = wav
                    break
        self.speaker_wav_excited = str(config.PROJECT_ROOT / raw_excited) if raw_excited else None

        # ORIGINAL / neutral voice: no cloning. It uses a built-in "studio" XTTS
        # voice (`speaker=` parameter), at a regular pace. Overrides the cloned
        # speaker_wav when COMMENTARY_SPEAKER=builtin.
        self.builtin_speaker = config.COQUI_BUILTIN_SPEAKER if config.USE_BUILTIN_SPEAKER else None
        if self.builtin_speaker:
            logger.info("Voce ORIGINALE (no cloning): speaker built-in '%s'.", self.builtin_speaker)
            self.speaker_wav = None
            self.speaker_wav_excited = None

    def synth_neutral(
        self, text: str, speed: float | None = None, excited: bool = False
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech using Coqui XTTS v2, maintaining continuous prosody for blocks.
        """
        # XTTS v2 in ITALIAN speaks out the final period ("punto") on short
        # sentences: we strip it. In other languages the behavior does not
        # happen, and the period can guide the sentence-final intonation.
        # Internal commas, ! and ? stay because they guide the prosody.
        if config.LANGUAGE == "it":
            text = text.rstrip().rstrip(".").rstrip()
        # split_sentences=False: the block (multiple sentences) is generated in
        # ONE single pass -> continuous/connected prosody across the sentences.
        kwargs = {
            "text": text,
            "language": config.COQUI_LANGUAGE,
            "speed": config.COQUI_SPEED if speed is None else speed,
            "split_sentences": False,
        }
        # Built-in voice (original/neutral) -> `speaker=`; otherwise cloning
        # from the reference (excited on important events, if available).
        if self.builtin_speaker:
            kwargs["speaker"] = self.builtin_speaker
        else:
            ref = (
                self.speaker_wav_excited
                if (excited and self.speaker_wav_excited)
                else self.speaker_wav
            )
            if ref:
                kwargs["speaker_wav"] = ref
        wav = np.asarray(self.tts.tts(**kwargs), dtype=np.float32)
        # XTTS's real rate; we resample ONLY if it does not match the synthesis
        # target (usually both 24 kHz -> no resampling, max quality).
        sr = getattr(self.tts.synthesizer, "output_sample_rate", self.output_sample_rate)
        if sr != self.output_sample_rate:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.output_sample_rate)
        return wav, self.output_sample_rate


def make_tts_engine(name: str = "coqui") -> TTSEngine:
    """Only supported engine: Coqui XTTS v2 (voice cloning, native Italian)."""
    if name != "coqui":
        logger.warning("Motore '%s' non piu' supportato: uso XTTS (coqui).", name)
    return CoquiEngine()


# --------------------------------------------------------------------------- #
# Track synthesis                                                             #
# --------------------------------------------------------------------------- #
def _trim_silence(
    wav: np.ndarray, sr: int, thresh: float = 0.02, margin_ms: int = 30
) -> np.ndarray:
    """Trim the silence at head/tail (XTTS adds padding to every line).
    Amplitude threshold + small margin so the attacks are not clipped."""
    active = np.where(np.abs(wav) > thresh)[0]
    if active.size == 0:
        return wav
    pad = int(sr * margin_ms / 1000)
    start = max(0, active[0] - pad)
    end = min(len(wav), active[-1] + 1 + pad)
    return wav[start:end]


def _fade_in(wav: np.ndarray, sr: int, ms: int) -> np.ndarray:
    """Linear fade-in ramp: softens the 'raspy' startup attack of XTTS."""
    n = min(len(wav), int(sr * ms / 1000))
    if n <= 0:
        return wav
    wav = wav.copy()
    wav[:n] *= np.linspace(0.0, 1.0, n, dtype=wav.dtype)
    return wav


def _is_excited(importance: float) -> bool:
    """
    Check if the event importance crosses the threshold for excited delivery.
    """
    return importance >= config.EMPHASIS_IMPORTANCE_THRESHOLD


def _group_utterances(
    items: list[tuple[str, str, float, float, float]], max_chars: int, max_gap_s: float
) -> list[tuple[str, str, float, float, float]]:
    """Merge consecutive lines into blocks <= max_chars keeping the internal
    punctuation. A block is split when: the emotional level changes
    (calm<->excited), so an important event (e.g. goal) ends up in its own block
    and can get its own emphasis; OR two lines are farther apart in time than
    max_gap_s (merging them would break the sync).
    Each item is (text, event_type, importance, t, audio_energy); the block
    inherits event_type/importance/audio_energy of the MOST important line
    (its "character") and the timestamp of the FIRST line (when it starts on
    the timeline)."""
    chunks: list[tuple[str, str, float, float, float]] = []
    cur, cur_evt, cur_imp, cur_t, cur_energy = "", "idle", -1.0, 0.0, 0.5
    prev_t: float | None = None
    for text, evt, imp, t, energy in items:
        text = text.strip()
        if not text:
            continue
        candidate = f"{cur} {text}".strip()
        gap_too_big = prev_t is not None and (t - prev_t) > max_gap_s
        if cur and (
            _is_excited(imp) != _is_excited(cur_imp) or len(candidate) > max_chars or gap_too_big
        ):
            chunks.append((cur, cur_evt, cur_imp, cur_t, cur_energy))
            cur, cur_evt, cur_imp, cur_t, cur_energy = text, evt, imp, t, energy
        else:
            if not cur:  # first line of the block: sets its timestamp
                cur_t = t
            cur = candidate
            if imp > cur_imp:  # block representative = most important
                cur_evt, cur_imp, cur_energy = evt, imp, energy
        prev_t = t
    if cur:
        chunks.append((cur, cur_evt, cur_imp, cur_t, cur_energy))
    return chunks


def synthesize(script: list[dict], predictor: ProsodyPredictor, tts: TTSEngine) -> np.ndarray:
    """
    Synthesize an entire script into a single continuous audio track.
    Groups lines into blocks and positions them sequentially on the timeline to avoid overlap.
    """
    sr = tts.output_sample_rate  # XTTS native 24 kHz (not the 22050 of the analysis)
    min_gap = int(config.GAP_BETWEEN_UTTERANCES_S * sr)

    # NOTE: the prosody is NO LONGER applied via DSP (WORLD/pyworld). Passing an
    # already-synthetic neural voice through a second vocoder "double-vocodes"
    # it and introduces the metallic artifact. We use the TTS voice as is: the
    # expressiveness comes from the reference audio (voice cloning). The
    # predictor stays for compatibility (log/suffix).
    #
    # The lines are merged into BLOCKS and synthesized in one pass, so XTTS
    # connects the prosody across the sentences (more natural speech) instead
    # of generating isolated sentences one after the other.
    items = [
        (
            line.get("text", ""),
            line.get("event_type", "idle"),
            float(line.get("importance", 0.0)),
            float(line.get("t", 0.0)),
            # audio_energy: REAL crowd excitement (Phase 1c), if the line has
            # it; 0.5 (neutral) if Phase 1c was not run.
            float(line.get("audio_energy", 0.5)),
        )
        for line in script
    ]
    chunks = _group_utterances(items, config.COQUI_CHUNK_MAX_CHARS, config.SYNC_MERGE_MAX_GAP_S)
    lo, hi = config.COQUI_SPEED_CLAMP

    # Synthesis + placement on the TIMELINE: each block starts at its event's
    # timestamp; if the previous one is not finished yet, it queues (cursor) so
    # the voices do not overlap. The track ends with the last line.
    placed: list[tuple[int, np.ndarray]] = []  # (start sample, audio)
    cursor = 0  # end (in samples) of the last placed block
    for i, (chunk, evt, imp, t, energy) in enumerate(chunks):
        # EMPHASIS FROM THE MODEL (Phase 3): rate_factor -> speed, energy_gain -> volume.
        # energy (Phase 1c) is an EXTRA feature: for the same event, the prosody
        # now also reacts to how much the crowd actually got excited at that
        # moment, not only to the event type/importance.
        prosody = predictor.predict(evt, imp, energy)
        speed = float(np.clip(config.COQUI_SPEED * prosody["rate_factor"], lo, hi))
        excited = _is_excited(imp)
        try:
            voiced, _ = tts.synth_neutral(chunk, speed=speed, excited=excited)
            voiced = _trim_silence(voiced, sr)
            voiced = _fade_in(voiced, sr, config.ONSET_FADE_MS)
            voiced = voiced * float(prosody["energy_gain"])
        except Exception as exc:
            logger.warning("Blocco %d saltato (%s): %s", i + 1, chunk[:30], exc)
            continue
        start = max(int(round(t * sr)), cursor)
        placed.append((start, voiced))
        cursor = start + len(voiced) + min_gap
        # Log ONLY on a successful block: before, "synthesized" was printed
        # even for blocks skipped by the except, confusing the count.
        logger.info(
            "Sintetizzati %d/%d blocchi a t=%.2fs (inizio reale %.2fs, %s, speed=%.2f, gain=%.2f).",
            i + 1,
            len(chunks),
            t,
            start / sr,
            evt,
            speed,
            prosody["energy_gain"],
        )

    if not placed:
        return np.zeros(sr, np.float32)
    total = max(start + len(voiced) for start, voiced in placed)
    track = np.zeros(total, np.float32)
    for start, voiced in placed:
        track[start : start + len(voiced)] += voiced
    peak = float(np.max(np.abs(track))) or 1.0
    return track / peak * 0.95  # final normalization


def main() -> None:
    """
    Main entry point for Phase 4.
    Synthesizes the given JSON script to an output WAV audio track.
    """
    parser = argparse.ArgumentParser(description="Fase 4 - Sintesi audio")
    parser.add_argument("--script", required=True, help="JSON dello script (Fase 2/2b).")
    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Forza la prosodia a regole (condizione baseline dello studio).",
    )
    parser.add_argument("--engine", choices=["coqui"], default="coqui")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    import soundfile as sf

    script_path = Path(args.script)
    script = load_json(script_path)
    logger.info("Caricate %d battute da %s", len(script), script_path)

    predictor = ProsodyPredictor(force_rule_based=args.rule_based)
    tts = make_tts_engine(args.engine)
    track = synthesize(script, predictor, tts)

    sr = tts.output_sample_rate  # real track rate (24 kHz), not 22050
    suffix = "_rulebased" if args.rule_based else "_model"
    out_path = (
        Path(args.out) if args.out else config.AUDIO_OUT_DIR / f"{script_path.stem}{suffix}.wav"
    )
    ensure_dir(out_path)
    sf.write(str(out_path), track, sr)
    logger.info("Traccia salvata (%.1fs) -> %s", len(track) / sr, out_path)
    logger.info("Fase 4 completata.")


if __name__ == "__main__":
    main()
