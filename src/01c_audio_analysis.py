"""
01c_audio_analysis.py
====================
FASE 1c - Modulo Uditivo: Crowd Excitement + Trascrizione Whisper.

A) CROWD EXCITEMENT: misura l'eccitazione del pubblico (RMS, centroide
   spettrale, densita' di onset) -> livello low/medium/high/peak nel tempo.
B) TRASCRIZIONE: trascrive la telecronaca originale del gioco con Whisper
   (utile per i nomi dei giocatori che l'OCR non vede, es. il portiere).

Output: features/events/<nome_video>_audio.json  (oppure _enriched.json se --events)

Uso:
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


class CrowdAnalyzer:
    """Misura l'eccitazione del pubblico su finestre temporali."""

    def __init__(self, sample_rate: int = config.SAMPLE_RATE) -> None:
        self.sr = sample_rate
        self.thresholds = config.CROWD_EXCITEMENT_THRESHOLDS
        self.window_s = config.AUDIO_ANALYSIS_WINDOW_S

    def analyze_full(self, audio: np.ndarray) -> list[dict]:
        import librosa
        total = len(audio) / self.sr
        win = int(self.window_s * self.sr)
        hop = max(win // 2, 1)

        global_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
        global_centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self.sr))) + 1e-8

        results: list[dict] = []
        offset = 0
        while offset < len(audio):
            chunk = audio[offset:offset + win]
            if len(chunk) < self.sr // 10:
                break
            t_start = offset / self.sr
            t_end = min((offset + len(chunk)) / self.sr, total)

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            centroid = float(np.mean(
                librosa.feature.spectral_centroid(y=chunk, sr=self.sr)))
            onset = float(np.mean(librosa.onset.onset_strength(y=chunk, sr=self.sr)))

            rms_n = min(rms / global_rms, 3.0) / 3.0
            cen_n = min(centroid / global_centroid, 3.0) / 3.0
            ons_n = min(onset / 5.0, 1.0)
            score = float(np.clip(0.50 * rms_n + 0.30 * cen_n + 0.20 * ons_n, 0.0, 1.0))

            results.append({
                "t_start": round(t_start, 2), "t_end": round(t_end, 2),
                "excitement_score": round(score, 3),
                "excitement_level": self._level(score),
            })
            offset += hop
        return results

    def get_at(self, data: list[dict], timestamp: float) -> dict:
        for e in data:
            if e["t_start"] <= timestamp <= e["t_end"]:
                return e
        if data:
            return min(data, key=lambda e: abs(e["t_start"] - timestamp))
        return {"excitement_score": 0.0, "excitement_level": "low"}

    def _level(self, score: float) -> str:
        if score >= self.thresholds["high"]:
            return "peak"
        if score >= self.thresholds["medium"]:
            return "high"
        if score >= self.thresholds["low"]:
            return "medium"
        return "low"


class WhisperTranscriber:
    """Trascrive l'audio con OpenAI Whisper -> segmenti {start, end, text}."""

    def __init__(self, model_size: str = config.WHISPER_MODEL_SIZE,
                 language: str = config.WHISPER_LANGUAGE) -> None:
        import whisper
        logger.info("Carico Whisper '%s'...", model_size)
        self.model = whisper.load_model(model_size)
        self.language = language

    def transcribe(self, audio_path) -> list[dict]:
        logger.info("Trascrizione (lingua=%s)...", self.language)
        result = self.model.transcribe(str(audio_path), language=self.language,
                                       task="transcribe", verbose=False)
        segments = [{"start": round(s["start"], 2), "end": round(s["end"], 2),
                     "text": s["text"].strip()} for s in result.get("segments", [])]
        logger.info("Trascritti %d segmenti.", len(segments))
        return segments


def extract_audio(video_path: Path) -> Path:
    out = Path(tempfile.mktemp(suffix=".wav"))
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
           "-ar", str(config.SAMPLE_RATE), "-ac", "1", str(out)]
    logger.info("Estraggo audio: %s", video_path.name)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg non trovato (Ubuntu: sudo apt-get install ffmpeg).")
    return out


def analyze_audio(video_path: Path, use_whisper: bool = True) -> dict:
    import librosa
    audio_path = extract_audio(video_path)
    try:
        audio, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        logger.info("Audio: %.1fs @ %dHz", duration, sr)

        excitement = CrowdAnalyzer(sr).analyze_full(audio)
        scores = [e["excitement_score"] for e in excitement]
        levels = [e["excitement_level"] for e in excitement]

        transcription: list[dict] = []
        if use_whisper:
            try:
                transcription = WhisperTranscriber().transcribe(audio_path)
            except ImportError:
                logger.warning("Whisper non installato (pip install openai-whisper).")
            except Exception as exc:
                logger.warning("Errore trascrizione: %s", exc)

        return {
            "excitement": excitement,
            "transcription": transcription,
            "summary": {
                "duration_s": round(duration, 2),
                "avg_excitement": round(float(np.mean(scores)), 3) if scores else 0.0,
                "max_excitement": round(float(np.max(scores)), 3) if scores else 0.0,
                "level_distribution": {lv: levels.count(lv) for lv in set(levels)},
                "n_transcribed_segments": len(transcription),
            },
        }
    finally:
        Path(audio_path).unlink(missing_ok=True)


def enrich_events_with_audio(events: list[dict], audio_data: dict) -> list[dict]:
    analyzer = CrowdAnalyzer()
    excitement = audio_data.get("excitement", [])
    transcription = audio_data.get("transcription", [])
    enriched: list[dict] = []
    for ev in events:
        e = dict(ev)
        t = e.get("t", 0.0)
        exc = analyzer.get_at(excitement, t)
        e["crowd_excitement"] = exc.get("excitement_level", "low")
        e["crowd_score"] = exc.get("excitement_score", 0.0)
        context = " ".join(s["text"] for s in transcription
                           if abs(s["start"] - t) <= 3.0 or abs(s["end"] - t) <= 3.0)
        e["original_commentary"] = context.strip()
        enriched.append(e)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1c - Analisi audio (Tifo + Whisper)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--no-whisper", action="store_true")
    parser.add_argument("--events", type=str, default=None,
                        help="JSON eventi da arricchire con i dati audio.")
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
