"""
01_extract_events.py
====================
FASE 1 - Estrazione eventi dall'HUD (Computer Vision / OCR).

Su frame GIA' normalizzati (rotazione + crop, vedi utils.VideoNormalizer),
legge quattro regioni dell'HUD: punteggio, cronometro e le DUE targhette nomi
(giocatore controllato di casa e ospite). Ricostruisce gli eventi dai cambiamenti.

CORREZIONI rispetto alla prima versione:
  - Il punteggio e' il PRIMO e l'ULTIMO numero letti: l'icona centrale (timer/
    pallone) che l'OCR scambia per una cifra (es. "BRA 1 [8] 1 HAI") viene ignorata.
  - Punteggio stabilizzato e MONOTONO: non puo' diminuire ne' saltare di colpo,
    quindi un errore momentaneo dell'OCR non genera un gol falso.
  - Nomi normalizzati: "VINI JR." e "VINI JR" sono lo stesso giocatore.
  - Il gol e' attribuito alla squadra che ha incrementato il punteggio (casa o
    ospite). Durante l'esultanza le targhette sono vuote: il gol viene rilevato
    comunque, perche' dipende dal punteggio, non dai nomi.

Nota: dalla sola HUD il "turnover" non e' affidabile; per tiri/parate/turnover
usa la Fase 1b (modulo visivo) e fondi i risultati.

Output: features/events/<nome_video>.json

Uso:
    python 01_extract_events.py --video data/raw/gameplay/match1.mp4
    python 01_extract_events.py --video data/raw/gameplay/match1.mp4 --limit 200
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


# --------------------------------------------------------------------------- #
# OCR delle regioni HUD                                                       #
# --------------------------------------------------------------------------- #
class HudReader:
    """OCR sulle regioni dell'HUD di un singolo frame normalizzato."""

    def __init__(self, languages: list[str], min_confidence: float) -> None:
        import easyocr   # import locale: pesante
        logger.info("Inizializzo EasyOCR (lingue=%s)...", languages)
        self.reader = easyocr.Reader(languages, gpu=True)
        self.min_confidence = min_confidence

    @staticmethod
    def _crop(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    def read_region(self, frame: np.ndarray, region) -> str:
        crop = self._crop(frame, region)
        if crop.size == 0:
            return ""
        results = self.reader.readtext(crop)
        tokens = [text for _, text, conf in results if conf >= self.min_confidence]
        return " ".join(tokens).strip()


# --------------------------------------------------------------------------- #
# Parsing del testo letto                                                     #
# --------------------------------------------------------------------------- #
def parse_score(text: str) -> tuple[int, int] | None:
    """
    Estrae (home, away) prendendo il PRIMO e l'ULTIMO numero della stringa.
    Ignora l'icona centrale (timer/pallone) che l'OCR legge come una cifra:
      "BRA 1 8 0 HAI" -> (1, 0)     "BRA 1 8 1 HAI" -> (1, 1)
    """
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[-1])
    if len(nums) == 1:           # una sola cifra letta: non affidabile
        return None
    return None


def parse_clock(text: str) -> str | None:
    """Estrae 'MM:SS' dal cronometro."""
    m = re.search(r"(\d{1,3})\s*[:\.]\s*(\d{2})", text)
    return f"{int(m.group(1))}:{m.group(2)}" if m else None


def clean_name(text: str) -> str:
    """
    Ripulisce una targhetta: toglie cifre e punteggiatura, normalizza in
    MAIUSCOLO. Cosi' "VINI JR." e "VINI JR" diventano identici.
    """
    name = text.upper()
    name = re.sub(r"[^A-Z ]", " ", name)     # via cifre, punti, simboli
    return re.sub(r"\s+", " ", name).strip()


def team_of(player: str) -> str | None:
    return config.ROSTER.get(player.upper())


# --------------------------------------------------------------------------- #
# Tracker del punteggio (stabile, monotono, anti-flicker)                     #
# --------------------------------------------------------------------------- #
class ScoreTracker:
    """
    Mantiene il punteggio 'confermato' e segnala i gol in modo robusto:
      - si inizializza alla prima lettura valida (nessun gol al primo frame);
      - accetta solo aumenti PLAUSIBILI: nessun lato puo' diminuire e
        l'incremento totale non puo' superare +2 (un OCR ballerino viene ignorato);
      - quando il punteggio confermato aumenta, restituisce il lato che ha segnato.
    """

    MAX_STEP = 2   # incremento massimo plausibile tra due frame campionati

    def __init__(self) -> None:
        self.confirmed: tuple[int, int] | None = None

    def update(self, raw: tuple[int, int] | None) -> str | None:
        """Restituisce 'home'/'away' se e' stato segnato un gol, altrimenti None."""
        if raw is None:
            return None
        if self.confirmed is None:
            self.confirmed = raw
            return None

        ch, ca = self.confirmed
        h, a = raw
        # Implausibile (lettura errata): un lato cala, o salto troppo grande.
        if h < ch or a < ca or (h - ch) + (a - ca) > self.MAX_STEP:
            return None
        if (h, a) == self.confirmed:
            return None

        side = "home" if h > ch else "away"
        self.confirmed = (h, a)
        return side


# --------------------------------------------------------------------------- #
# Estrazione                                                                   #
# --------------------------------------------------------------------------- #
class EventExtractor:
    """Stato della partita frame dopo frame; produce gli eventi."""

    def __init__(self) -> None:
        self.score = ScoreTracker()
        self.prev_home = ""
        self.last_home = ""     # ultimo nome di casa visto (per attribuire i gol)
        self.last_away = ""     # ultimo nome ospite visto

    def step(self, t: float, score_txt: str, clock_txt: str,
             home_txt: str, away_txt: str) -> dict | None:
        raw_score = parse_score(score_txt)
        clock = parse_clock(clock_txt)
        home = clean_name(home_txt)
        away = clean_name(away_txt)
        if home:
            self.last_home = home
        if away:
            self.last_away = away

        # 1) GOL: ha priorita' e non dipende dai nomi (targhette vuote in esultanza).
        scoring_side = self.score.update(raw_score)
        if scoring_side is not None:
            scorer = self.last_home if scoring_side == "home" else self.last_away
            team_code = config.TEAM_CODES.get(scoring_side, scoring_side)
            event = {
                "t": round(t, 2),
                "type": "goal",
                "player": scorer or team_code,   # se la targhetta era vuota, la squadra
                "scoring_team": scoring_side,
                "scoring_team_code": team_code,
                "player_home": home,
                "player_away": away,
                "importance": config.EVENT_IMPORTANCE["goal"],
                "score": list(self.score.confirmed),
                "clock": clock,
                "source": "ocr",
            }
            self.prev_home = home
            return event

        # 2) PASSAGGIO: il giocatore di casa controllato cambia davvero.
        event = None
        if home and home != self.prev_home and self.prev_home != "":
            event = {
                "t": round(t, 2),
                "type": "pass",
                "player": home,
                "player_home": home,
                "player_away": away,
                "importance": config.EVENT_IMPORTANCE["pass"],
                "score": list(self.score.confirmed) if self.score.confirmed else None,
                "clock": clock,
                "source": "ocr",
            }
        if home:
            self.prev_home = home
        return event


def extract_events(video_path: Path, reader: HudReader, limit: int | None) -> list[dict]:
    extractor = EventExtractor()
    events: list[dict] = []
    processed = 0

    for timestamp, frame in tqdm(
        iter_sampled_frames(video_path, config.FRAMES_PER_SECOND), desc="OCR HUD"
    ):
        if limit and processed >= limit:
            break
        processed += 1
        try:
            score_txt = reader.read_region(frame, config.HUD_REGIONS["score"])
            clock_txt = reader.read_region(frame, config.HUD_REGIONS["clock"])
            home_txt = reader.read_region(frame, config.HUD_REGIONS["active_player_home"])
            away_txt = reader.read_region(frame, config.HUD_REGIONS["active_player_away"])
        except Exception as exc:
            logger.debug("Frame a %.1fs saltato: %s", timestamp, exc)
            continue

        event = extractor.step(timestamp, score_txt, clock_txt, home_txt, away_txt)
        if event is not None:
            events.append(event)

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1 - Estrazione eventi da HUD")
    parser.add_argument("--video", required=True, help="Percorso del video di gameplay.")
    parser.add_argument("--limit", type=int, default=None, help="Max frame (debug).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    reader = HudReader(config.OCR_LANGUAGES, config.OCR_MIN_CONFIDENCE)
    events = extract_events(video_path, reader, args.limit)

    out_path = config.EVENTS_DIR / f"{video_path.stem}.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    by_type: dict[str, int] = {}
    for e in events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    n_goal = by_type.get("goal", 0)
    logger.info("Estratti %d eventi %s -> %s", len(events), by_type, out_path)
    if n_goal:
        for e in events:
            if e["type"] == "goal":
                logger.info("  GOL a %.1fs | punteggio %s | segna: %s (%s)",
                            e["t"], e["score"], e["player"], e["scoring_team_code"])
    else:
        logger.warning("Nessun gol rilevato: controlla che la clip arrivi fino al gol "
                       "e che il punteggio venga letto correttamente.")
    logger.info("Fase 1 completata.")


if __name__ == "__main__":
    main()