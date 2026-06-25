"""
01_extract_events.py
====================
FASE 1 - Estrazione eventi dall'HUD (Computer Vision / OCR).

Legge un video di gameplay, campiona i frame, ritaglia le regioni dell'HUD
(punteggio e nome del giocatore con la palla), le passa all'OCR e RICOSTRUISCE
gli eventi dai *cambiamenti*:

    - punteggio incrementa            -> "goal"
    - nome attivo cambia, stessa squadra -> "pass"
    - nome attivo passa all'avversario   -> "turnover"
    - nessun cambiamento               -> "idle"

Output: features/events/<nome_video>.json  (lista ordinata di eventi).

Nota: e' la parte di "ingegneria" della pipeline (non deep learning). Le regioni
HUD vanno calibrate sul proprio video (vedi config.HUD_REGIONS e il README).

Uso:
    python src/01_extract_events.py --video data/raw/gameplay/match1.mp4
    python src/01_extract_events.py --video data/raw/gameplay/match1.mp4 --limit 200
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from utils import ensure_dir, get_logger, iter_sampled_frames, save_json

logger = get_logger("fase1_eventi")


class HudReader:
    """Esegue l'OCR sulle regioni dell'HUD di un singolo frame."""

    def __init__(self, languages: list[str], min_confidence: float) -> None:
        import easyocr  # import locale: pesante, si carica solo quando serve

        logger.info("Inizializzo EasyOCR (lingue=%s)...", languages)
        self.reader = easyocr.Reader(languages, gpu=True)
        self.min_confidence = min_confidence

    @staticmethod
    def _crop(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        """Ritaglia una regione da coordinate normalizzate [0,1]."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    def read_region(self, frame: np.ndarray, region) -> str:
        """Restituisce il testo letto in una regione (stringa vuota se nulla)."""
        crop = self._crop(frame, region)
        if crop.size == 0:
            return ""
        results = self.reader.readtext(crop)
        # Concatena i frammenti sopra la soglia di confidenza
        tokens = [text for _, text, conf in results if conf >= self.min_confidence]
        return " ".join(tokens).strip()


def parse_score(text: str) -> tuple[int, int] | None:
    """Estrae (home, away) da testi tipo '2 - 1', '2:1', '2 1'."""
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None


def team_of(player: str) -> str | None:
    """Squadra del giocatore secondo il roster in config (None se sconosciuto)."""
    return config.ROSTER.get(player.upper())


def classify_event(prev: dict, curr: dict) -> str:
    """
    Determina il tipo di evento confrontando lo stato precedente con quello attuale.
    `prev`/`curr` contengono: score (tupla|None), player (str).
    """
    # Gol: il punteggio e' aumentato
    if prev["score"] and curr["score"] and sum(curr["score"]) > sum(prev["score"]):
        return "goal"

    prev_p, curr_p = prev["player"], curr["player"]
    if not curr_p:
        return "idle"
    if curr_p == prev_p:
        return "idle"  # stesso giocatore: nessun nuovo evento

    # Cambio di giocatore: passaggio o palla persa, in base alla squadra
    prev_team, curr_team = team_of(prev_p), team_of(curr_p)
    if prev_team and curr_team and prev_team != curr_team:
        return "turnover"
    return "pass"


def extract_events(video_path: Path, reader: HudReader, limit: int | None) -> list[dict]:
    """Scorre il video e produce la lista di eventi."""
    events: list[dict] = []
    prev = {"score": None, "player": ""}
    processed = 0

    for timestamp, frame in tqdm(
        iter_sampled_frames(video_path, config.FRAMES_PER_SECOND), desc="OCR HUD"
    ):
        if limit and processed >= limit:
            break
        processed += 1

        try:
            score_txt = reader.read_region(frame, config.HUD_REGIONS["score"])
            player_txt = reader.read_region(frame, config.HUD_REGIONS["active_player"])
        except Exception as exc:
            # Un frame illeggibile non deve fermare l'estrazione.
            logger.debug("Frame a %.1fs saltato: %s", timestamp, exc)
            continue

        curr = {
            "score": parse_score(score_txt) or prev["score"],
            "player": player_txt.upper(),
        }
        event_type = classify_event(prev, curr)

        # Registra solo gli eventi "interessanti" (salta gli idle ripetuti)
        if event_type != "idle":
            events.append({
                "t": round(timestamp, 2),
                "type": event_type,
                "player": curr["player"] or "il giocatore",
                "importance": config.EVENT_IMPORTANCE[event_type],
                "score": list(curr["score"]) if curr["score"] else None,
            })
        prev = curr

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1 - Estrazione eventi da HUD")
    parser.add_argument("--video", required=True, help="Percorso del video di gameplay.")
    parser.add_argument("--limit", type=int, default=None, help="Max frame (debug).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    if not config.ROSTER:
        logger.warning("config.ROSTER e' vuoto: 'pass' e 'turnover' non saranno "
                       "distinguibili. Compila le rose per risultati migliori.")

    reader = HudReader(config.OCR_LANGUAGES, config.OCR_MIN_CONFIDENCE)
    events = extract_events(video_path, reader, args.limit)

    out_path = config.EVENTS_DIR / f"{video_path.stem}.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    by_type = {t: sum(1 for e in events if e["type"] == t) for t in config.EVENT_TYPES}
    logger.info("Estratti %d eventi (%s) -> %s", len(events), by_type, out_path)
    logger.info("Fase 1 completata.")


if __name__ == "__main__":
    main()
