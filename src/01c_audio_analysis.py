"""
01c_audio_analysis.py
====================
PHASE 1c - Auditory module: Audio Energy + Whisper transcription.

A) AUDIO ENERGY: measures the ENERGY/ACTIVITY of the audio (RMS, spectral
   centroid, onset density) -> low/medium/high/peak level over time.
   NB: it is called "energy", not "crowd", on purpose. Tests in hand (AST on
   AudioSet, dimensional SER) showed that this audio does NOT contain
   detectable crowd noise: it is voice (commentary) + effects. So the descriptor
   measures the audio energy - which correlates with the hot moments - not the
   crowd. It is a stated acoustic proxy, not a "crowd excitement" measure.
B) TRANSCRIPTION: transcribes the game's original commentary with Whisper
   (useful for player names the OCR does not see, e.g. the goalkeeper).

Output: features/events/<video_name>_audio.json  (or _enriched.json if --events)

Usage:
    python 01c_audio_analysis.py --video data/raw/gameplay/match1.mp4
    python 01c_audio_analysis.py --video data/raw/gameplay/match1.mp4 --no-whisper
    python 01c_audio_analysis.py --video data/raw/gameplay/match1.mp4 --events features/events/match1.json
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import config
from utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("fase1c_audio")


class AudioEnergyAnalyzer:
    """Measures the energy/activity of the audio over time windows (NOT the crowd:
    see the module docstring)."""

    def __init__(self, sample_rate: int = config.SAMPLE_RATE) -> None:
        """
        Initialize the analyzer with the specified sample rate and configuration thresholds.
        """
        self.sr = sample_rate
        self.thresholds = config.AUDIO_ENERGY_THRESHOLDS
        self.window_s = config.AUDIO_ANALYSIS_WINDOW_S

    def analyze_full(self, audio: np.ndarray) -> list[dict]:
        """
        Analyze the full audio array by computing RMS energy, spectral centroid, and onset density.
        Returns a timeline of normalized energy scores and discrete levels.
        """
        import librosa

        total = len(audio) / self.sr
        win = int(self.window_s * self.sr)
        hop = max(win // 2, 1)

        global_rms = float(np.sqrt(np.mean(audio**2))) + 1e-8
        global_centroid = (
            float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sr))) + 1e-8
        )

        # 1) RAW SCORE per window: energy (RMS) + brightness (centroid)
        #    + onset density (attacks), weighted. Normalized against the clip
        #    average, so an "average" value lands around ~0.33.
        windows: list[tuple[float, float, float]] = []  # (t_start, t_end, raw)
        offset = 0
        while offset < len(audio):
            chunk = audio[offset : offset + win]
            if len(chunk) < self.sr // 10:
                break
            t_start = offset / self.sr
            t_end = min((offset + len(chunk)) / self.sr, total)

            rms = float(np.sqrt(np.mean(chunk**2)))
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=self.sr)))
            onset = float(np.mean(librosa.onset.onset_strength(y=chunk, sr=self.sr)))

            rms_n = min(rms / global_rms, 3.0) / 3.0
            cen_n = min(centroid / global_centroid, 3.0) / 3.0
            ons_n = min(onset / 5.0, 1.0)
            raw = float(np.clip(0.50 * rms_n + 0.30 * cen_n + 0.20 * ons_n, 0.0, 1.0))
            windows.append((t_start, t_end, raw))
            offset += hop

        # 2) RE-SCALING to the clip percentiles: maps [p5, p95] -> [0,1], so the
        #    score covers the whole range and the levels (low/medium/high/peak)
        #    actually partition it (before, the score stayed ~0.33 and "peak"
        #    never fired). The AUDIO_ENERGY_MIN_RANGE guard prevents that on a
        #    FLAT clip the background noise gets stretched to fake "peak"s.
        results: list[dict] = []
        if windows:
            raws = np.array([w[2] for w in windows])
            p5, p95 = np.percentile(raws, [5, 95])
            span = max(float(p95 - p5), config.AUDIO_ENERGY_MIN_RANGE)
            for t_start, t_end, raw in windows:
                score = float(np.clip((raw - p5) / span, 0.0, 1.0))
                results.append(
                    {
                        "t_start": round(t_start, 2),
                        "t_end": round(t_end, 2),
                        "energy_score": round(score, 3),
                        "energy_level": self._level(score),
                    }
                )
        return results

    def get_at(self, data: list[dict], timestamp: float) -> dict:
        """
        Fetch the audio energy metrics at a specific timestamp.
        Falls back to a default neutral value if no data is found.
        """
        for e in data:
            if e["t_start"] <= timestamp <= e["t_end"]:
                return e
        if data:
            return min(data, key=lambda e: abs(e["t_start"] - timestamp))
        # No audio data: neutral (0.5), CONSISTENT with the default of the
        # prosody feature (event_to_features). A 0.0 would say "ice-cold crowd",
        # which is FALSE information, not a "don't know".
        return {"energy_score": 0.5, "energy_level": "medium"}

    def _level(self, score: float) -> str:
        """
        Convert a continuous normalized score into a discrete category level.
        """
        if score >= self.thresholds["high"]:
            return "peak"
        if score >= self.thresholds["medium"]:
            return "high"
        if score >= self.thresholds["low"]:
            return "medium"
        return "low"


class WhisperTranscriber:
    """Transcribes the audio with OpenAI Whisper -> segments {start, end, text}."""

    def __init__(
        self, model_size: str = config.WHISPER_MODEL_SIZE, language: str = config.WHISPER_LANGUAGE
    ) -> None:
        """
        Initialize the Whisper model for audio transcription.
        """
        import whisper

        logger.info("Carico Whisper '%s'...", model_size)
        self.model = whisper.load_model(model_size)
        self.language = language

    def transcribe(self, audio_path) -> list[dict]:
        """
        Transcribe the audio file using Whisper, returning time-aligned text segments.
        """
        logger.info("Trascrizione (lingua=%s)...", self.language)
        result = self.model.transcribe(
            str(audio_path), language=self.language, task="transcribe", verbose=False
        )
        segments = [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        logger.info("Trascritti %d segmenti.", len(segments))
        return segments


def extract_audio(video_path: Path) -> Path:
    """Extracts the video's audio track to a temporary mono wav (via ffmpeg).
    The file is deleted by the caller (analyze_audio) at the end of the analysis."""
    # NamedTemporaryFile instead of the deprecated tempfile.mktemp: reserves the
    # file at once (no name collisions); ffmpeg then overwrites it (-y).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(config.SAMPLE_RATE),
        "-ac",
        "1",
        str(out),
    ]
    logger.info("Estraggo audio: %s", video_path.name)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        out.unlink(missing_ok=True)  # no orphan files if ffmpeg is missing
        raise RuntimeError("ffmpeg non trovato (Ubuntu: sudo apt-get install ffmpeg).")
    except subprocess.CalledProcessError:
        out.unlink(missing_ok=True)  # same if ffmpeg fails on the video
        raise
    return out


def analyze_audio(video_path: Path, use_whisper: bool = True) -> dict:
    """
    Orchestrate the extraction, energy analysis, and transcription of a video's audio track.
    Returns a dictionary with timelines and a summary.
    """
    import librosa

    audio_path = extract_audio(video_path)
    try:
        audio, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        logger.info("Audio: %.1fs @ %dHz", duration, sr)

        energy = AudioEnergyAnalyzer(sr).analyze_full(audio)
        scores = [e["energy_score"] for e in energy]
        levels = [e["energy_level"] for e in energy]

        transcription: list[dict] = []
        if use_whisper:
            try:
                transcription = WhisperTranscriber().transcribe(audio_path)
            except ImportError:
                logger.warning("Whisper non installato (pip install openai-whisper).")
            except Exception as exc:
                logger.warning("Errore trascrizione: %s", exc)

        return {
            "energy": energy,
            "transcription": transcription,
            "summary": {
                "duration_s": round(duration, 2),
                "avg_energy": round(float(np.mean(scores)), 3) if scores else 0.0,
                "max_energy": round(float(np.max(scores)), 3) if scores else 0.0,
                "level_distribution": {lv: levels.count(lv) for lv in set(levels)},
                "n_transcribed_segments": len(transcription),
            },
        }
    finally:
        Path(audio_path).unlink(missing_ok=True)


def enrich_events_with_audio(events: list[dict], audio_data: dict) -> list[dict]:
    """
    Merge the audio analysis and transcription data into existing game events.
    Matches audio characteristics and spoken context to event timestamps.
    """
    analyzer = AudioEnergyAnalyzer()
    energy = audio_data.get("energy", [])
    transcription = audio_data.get("transcription", [])
    enriched: list[dict] = []
    for ev in events:
        e = dict(ev)
        t = e.get("t", 0.0)
        exc = analyzer.get_at(energy, t)
        # 0.5 (neutral) as default, consistent with event_to_features: an event
        # without an excitement value must not read as "ice-cold crowd" (0.0).
        e["audio_energy_level"] = exc.get("energy_level", "medium")
        e["audio_energy"] = exc.get("energy_score", 0.5)
        context = " ".join(
            s["text"]
            for s in transcription
            if abs(s["start"] - t) <= 3.0 or abs(s["end"] - t) <= 3.0
        )
        e["original_commentary"] = context.strip()
        enriched.append(e)
    return enriched


def main() -> None:
    """
    Main entry point for Phase 1c.
    Parses arguments, executes audio analysis, and outputs JSON metrics.
    """
    parser = argparse.ArgumentParser(description="Fase 1c - Analisi audio (Tifo + Whisper)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--no-whisper", action="store_true")
    parser.add_argument(
        "--events", type=str, default=None, help="JSON eventi da arricchire con i dati audio."
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    audio_data = analyze_audio(video_path, use_whisper=not args.no_whisper)

    if args.events:
        events = enrich_events_with_audio(load_json(Path(args.events)), audio_data)
        out_path = config.EVENTS_DIR / f"{video_path.stem}_enriched.json"
    else:
        events = audio_data
        out_path = config.EVENTS_DIR / f"{video_path.stem}_audio.json"

    ensure_dir(out_path)
    save_json(events, out_path)
    logger.info("Sommario: %s", audio_data["summary"])
    logger.info("Output -> %s", out_path)
    logger.info("Fase 1c completata.")


if __name__ == "__main__":
    main()
