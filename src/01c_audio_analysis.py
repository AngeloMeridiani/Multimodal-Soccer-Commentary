"""
01c_audio_analysis.py
====================
FASE 1c - Modulo Uditivo: Crowd Excitement + Trascrizione Whisper.

Analizza la traccia audio del video di gameplay per estrarre due tipi di
informazione:

A) CROWD EXCITEMENT (Analisi del Tifo):
   Misura l'intensita' sonora del pubblico usando Librosa. Picchi di volume
   e frequenze alte indicano momenti di eccitazione (quasi-gol, azione pericolosa).
   Output: un livello di eccitazione (low/medium/high/peak) allineato nel tempo.

B) TRASCRIZIONE TELECRONACA (Whisper):
   Trascrive il commento vocale originale del gioco (se presente) usando
   OpenAI Whisper. Estrae nomi di giocatori, tattiche, contesto.

Entrambi gli output vengono aggiunti al JSON eventi per arricchire il contesto
disponibile per la generazione della telecronaca (Fase 2b).

Output: features/events/<nome_video>_audio.json

Uso:
    python src/01c_audio_analysis.py --video data/raw/gameplay/match1.mp4
    python src/01c_audio_analysis.py --video data/raw/gameplay/match1.mp4 --no-whisper
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

import config
from utils import ensure_dir, get_logger, save_json

logger = get_logger("fase1c_audio")


# --------------------------------------------------------------------------- #
# A) Analisi Crowd Excitement                                                  #
# --------------------------------------------------------------------------- #
class CrowdAnalyzer:
    """
    Analizza l'audio del gameplay per misurare l'eccitazione del pubblico.

    Metriche calcolate su finestre temporali:
    - RMS Energy: volume complessivo
    - Spectral Centroid: frequenza media (pubblico che urla -> frequenze alte)
    - Onset Strength: densita' di transitori (applausi, boati)

    Il livello di eccitazione (low/medium/high/peak) si basa sulla combinazione
    normalizzata di queste metriche.
    """

    def __init__(self, sample_rate: int = config.SAMPLE_RATE) -> None:
        self.sr = sample_rate
        self.thresholds = config.CROWD_EXCITEMENT_THRESHOLDS
        self.window_s = config.AUDIO_ANALYSIS_WINDOW_S

    def analyze_full(self, audio: np.ndarray) -> list[dict]:
        """
        Analizza l'intero audio in finestre e restituisce la serie temporale
        di eccitazione. Ogni entry: {t_start, t_end, excitement_level, excitement_score}.
        """
        total_duration = len(audio) / self.sr
        window_samples = int(self.window_s * self.sr)
        hop_samples = window_samples // 2  # overlap del 50%

        # Calcola le metriche globali per la normalizzazione
        global_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
        global_centroid = float(np.mean(librosa.feature.spectral_centroid(
            y=audio, sr=self.sr
        ))) + 1e-8

        results: list[dict] = []
        offset = 0

        while offset < len(audio):
            chunk = audio[offset:offset + window_samples]
            if len(chunk) < self.sr // 10:  # troppo corto
                break

            t_start = offset / self.sr
            t_end = min((offset + len(chunk)) / self.sr, total_duration)

            # Metriche locali
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            centroid = float(np.mean(librosa.feature.spectral_centroid(
                y=chunk, sr=self.sr
            )))
            onset_env = librosa.onset.onset_strength(y=chunk, sr=self.sr)
            onset_density = float(np.mean(onset_env))

            # Score normalizzato (0..1)
            rms_norm = min(rms / global_rms, 3.0) / 3.0
            centroid_norm = min(centroid / global_centroid, 3.0) / 3.0
            onset_norm = min(onset_density / 5.0, 1.0)  # onset density tipicamente 0-5

            # Score combinato (peso maggiore a RMS)
            score = 0.50 * rms_norm + 0.30 * centroid_norm + 0.20 * onset_norm
            score = float(np.clip(score, 0.0, 1.0))

            level = self._score_to_level(score)

            results.append({
                "t_start": round(t_start, 2),
                "t_end": round(t_end, 2),
                "excitement_score": round(score, 3),
                "excitement_level": level,
                "rms": round(rms, 5),
                "spectral_centroid": round(centroid, 1),
            })

            offset += hop_samples

        return results

    def get_excitement_at(self, excitement_data: list[dict], timestamp: float) -> dict:
        """Restituisce il livello di eccitazione al timestamp dato."""
        for entry in excitement_data:
            if entry["t_start"] <= timestamp <= entry["t_end"]:
                return entry
        # Fallback: piu' vicino
        if excitement_data:
            closest = min(excitement_data, key=lambda e: abs(e["t_start"] - timestamp))
            return closest
        return {"excitement_score": 0.0, "excitement_level": "low"}

    def _score_to_level(self, score: float) -> str:
        """Converte lo score numerico in un livello categorico."""
        if score >= self.thresholds["high"]:
            return "peak"
        elif score >= self.thresholds["medium"]:
            return "high"
        elif score >= self.thresholds["low"]:
            return "medium"
        return "low"


# --------------------------------------------------------------------------- #
# B) Trascrizione con Whisper                                                  #
# --------------------------------------------------------------------------- #
class WhisperTranscriber:
    """
    Trascrive l'audio del gameplay usando OpenAI Whisper.

    Restituisce segmenti con timestamp, utili per allineare la trascrizione
    agli eventi visivi.
    """

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL_SIZE,
        language: str = config.WHISPER_LANGUAGE,
    ) -> None:
        import whisper  # import locale: pesante

        logger.info("Carico Whisper '%s'...", model_size)
        self.model = whisper.load_model(model_size)
        self.language = language

    def transcribe(self, audio_path: str | Path) -> list[dict]:
        """
        Trascrive l'audio e restituisce una lista di segmenti:
        [{start, end, text}, ...]
        """
        logger.info("Trascrizione in corso (lingua=%s)...", self.language)

        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            task="transcribe",
            verbose=False,
        )

        segments: list[dict] = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            })

        logger.info("Trascritti %d segmenti (%.0f secondi di audio).",
                     len(segments),
                     segments[-1]["end"] if segments else 0)
        return segments

    def get_context_at(self, segments: list[dict], timestamp: float, window: float = 3.0) -> str:
        """
        Restituisce il testo trascritto intorno a un dato timestamp.
        Utile per dare contesto all'LLM.
        """
        relevant = [
            s["text"] for s in segments
            if abs(s["start"] - timestamp) <= window or abs(s["end"] - timestamp) <= window
        ]
        return " ".join(relevant) if relevant else ""


# --------------------------------------------------------------------------- #
# Utilita': estrazione audio da video                                          #
# --------------------------------------------------------------------------- #
def extract_audio_from_video(video_path: Path, output_path: Path | None = None) -> Path:
    """Estrae la traccia audio da un video usando ffmpeg. Restituisce il path del .wav."""
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav"))

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",                          # niente video
        "-acodec", "pcm_s16le",         # WAV 16-bit
        "-ar", str(config.SAMPLE_RATE), # sample rate
        "-ac", "1",                     # mono
        str(output_path),
    ]

    logger.info("Estraggo audio: %s -> %s", video_path.name, output_path.name)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg non trovato. Installalo:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt-get install ffmpeg"
        )

    return output_path


# --------------------------------------------------------------------------- #
# Pipeline principale                                                          #
# --------------------------------------------------------------------------- #
def analyze_audio(
    video_path: Path,
    use_whisper: bool = True,
) -> dict:
    """
    Analisi completa dell'audio: crowd excitement + trascrizione opzionale.

    Restituisce un dict con:
    - "excitement": lista di entry temporali con livelli di eccitazione
    - "transcription": lista di segmenti trascritti (se use_whisper=True)
    - "summary": statistiche aggregate
    """
    # 1. Estrai l'audio dal video
    audio_path = extract_audio_from_video(video_path)

    try:
        # Carica l'audio per l'analisi
        audio, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        logger.info("Audio caricato: %.1f secondi @ %d Hz", duration, sr)

        # 2. Analisi crowd excitement
        analyzer = CrowdAnalyzer(sample_rate=sr)
        excitement = analyzer.analyze_full(audio)
        logger.info("Analizzate %d finestre di eccitazione.", len(excitement))

        # Statistiche
        scores = [e["excitement_score"] for e in excitement]
        levels = [e["excitement_level"] for e in excitement]
        level_counts = {lev: levels.count(lev) for lev in set(levels)}

        # 3. Trascrizione (opzionale)
        transcription: list[dict] = []
        if use_whisper:
            try:
                transcriber = WhisperTranscriber()
                transcription = transcriber.transcribe(audio_path)
            except ImportError:
                logger.warning("Whisper non installato. Salto la trascrizione. "
                               "Installa con: pip install openai-whisper")
            except Exception as exc:
                logger.warning("Errore durante la trascrizione: %s", exc)

        return {
            "excitement": excitement,
            "transcription": transcription,
            "summary": {
                "duration_s": round(duration, 2),
                "n_windows": len(excitement),
                "avg_excitement": round(float(np.mean(scores)), 3) if scores else 0.0,
                "max_excitement": round(float(np.max(scores)), 3) if scores else 0.0,
                "level_distribution": level_counts,
                "n_transcribed_segments": len(transcription),
            },
        }
    finally:
        # Pulisci il file audio temporaneo
        Path(audio_path).unlink(missing_ok=True)


def enrich_events_with_audio(
    events: list[dict],
    audio_data: dict,
) -> list[dict]:
    """
    Arricchisce una lista di eventi con i dati audio (eccitazione + trascrizione).
    """
    analyzer = CrowdAnalyzer()
    excitement = audio_data.get("excitement", [])
    transcription = audio_data.get("transcription", [])

    enriched: list[dict] = []
    for event in events:
        ev = dict(event)
        t = ev.get("t", 0.0)

        # Aggiungi eccitazione del tifo
        exc = analyzer.get_excitement_at(excitement, t)
        ev["crowd_excitement"] = exc.get("excitement_level", "low")
        ev["crowd_score"] = exc.get("excitement_score", 0.0)

        # Aggiungi contesto dalla telecronaca originale
        if transcription:
            transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
            context = ""
            for s in transcription:
                if abs(s["start"] - t) <= 3.0 or abs(s["end"] - t) <= 3.0:
                    context += " " + s["text"]
            ev["original_commentary"] = context.strip()
        else:
            ev["original_commentary"] = ""

        enriched.append(ev)

    return enriched


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1c - Analisi audio (Tifo + Whisper)")
    parser.add_argument("--video", required=True, help="Percorso del video di gameplay.")
    parser.add_argument("--no-whisper", action="store_true",
                        help="Salta la trascrizione Whisper (solo crowd excitement).")
    parser.add_argument("--events", type=str, default=None,
                        help="JSON eventi (Fase 1/1b) da arricchire con i dati audio.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    # Analisi audio completa
    audio_data = analyze_audio(video_path, use_whisper=not args.no_whisper)

    # Se abbiamo eventi, arricchiscili
    if args.events:
        from utils import load_json
        events = load_json(Path(args.events))
        events = enrich_events_with_audio(events, audio_data)
        out_path = config.EVENTS_DIR / f"{video_path.stem}_enriched.json"
    else:
        # Salva solo i dati audio grezzi
        events = audio_data  # type: ignore[assignment]
        out_path = config.EVENTS_DIR / f"{video_path.stem}_audio.json"

    ensure_dir(out_path)
    save_json(events, out_path)

    logger.info("Sommario audio: %s", audio_data["summary"])
    logger.info("Output -> %s", out_path)
    logger.info("Fase 1c completata.")


if __name__ == "__main__":
    main()
 