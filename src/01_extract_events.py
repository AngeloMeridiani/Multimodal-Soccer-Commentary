"""
01_extract_events.py  (versione corretta, self-contained)
=========================================================
FASE 1 - Estrazione eventi dall'HUD (Computer Vision / OCR).

Correzioni rispetto alla versione precedente:
  * GOL: non basta piu' che la somma del punteggio salga. Serve +1 ESATTO a una
    squadra, l'altra invariata, punteggi plausibili e variazione CONFERMATA su
    piu' frame (debounce). Uccide i "gol fantasma" da mislettura (es. 16->18).
  * NOMI: ogni targhetta viene agganciata alla rosa (config.ROSTER_HOME/AWAY)
    con match esatto/sottostringa/fuzzy. "INHOS"->MARQUINHOS, "NDRO"->ALEX SANDRO.
    I frammenti troppo corti restano marcati incerti (player_resolved=False).
  * SQUADRA: l'attore non e' piu' "sempre home". Si deriva dalla rosa del nome
    agganciato; se entrambe le targhette sono leggibili il possesso e' ambiguo
    dalla sola HUD e va confermato in Fase 1b (possession_certain=False).

Tutta la logica di validazione/aggancio e' QUI dentro: nessun file aggiuntivo.

Output: features/events/<nome_video>.json
"""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from utils import ensure_dir, get_logger, iter_sampled_frames, save_json

logger = get_logger("fase1_eventi")

# --- parametri (dalla config, con fallback se non definiti) ---------------- #
ROSTER_HOME = getattr(config, "ROSTER_HOME", [])
ROSTER_AWAY = getattr(config, "ROSTER_AWAY", [])
MAX_SCORE = getattr(config, "MAX_PLAUSIBLE_SCORE", 9)
CONFIRM_N = getattr(config, "GOAL_CONFIRM_FRAMES", 2)
NAME_MIN_CONF = getattr(config, "OCR_NAME_MIN_CONFIDENCE", 0.15)
NAME_MIN_FRAGMENT = 3       # sotto questa lunghezza NON si tenta il match ("RO","AA")
NAME_RATIO_THRESHOLD = 0.62  # similarita' minima per accettare un match "fuzzy"


# =========================================================================== #
# Aggancio dei nomi alla rosa (ex events_cleanup)                             #
# =========================================================================== #
def normalize_name(raw: str | None) -> str:
    """MAIUSCOLO, via numeri di maglia e simboli, spazi compattati."""
    if not raw:
        return ""
    name = re.sub(r"[^A-Za-zÀ-ÿ ]", " ", raw).upper()
    return re.sub(r"\s+", " ", name).strip()


def _compact(s: str) -> str:
    return s.replace(" ", "")


def _best_in_roster(name: str, roster: list[str]) -> tuple[str | None, float]:
    """Miglior candidato nella rosa e relativo punteggio [0,1]."""
    if not name:
        return None, 0.0
    if name in roster:
        return name, 1.0
    nc = _compact(name)
    best, best_score = None, 0.0
    for cand in roster:
        cc = _compact(cand)
        score = 0.0
        # (a) Contenimento (frammenti tronchi: "INHOS" in "MARQUINHOS").
        if len(name) >= NAME_MIN_FRAGMENT and (nc in cc or cc in nc):
            shorter, longer = sorted((len(nc), len(cc)))
            score = 0.6 + 0.4 * (shorter / max(longer, 1))
        # (b) Similarita' fuzzy sull'intera stringa e sulle singole parole.
        ratio = SequenceMatcher(None, nc, cc).ratio()
        for w in cand.split():
            ratio = max(ratio, SequenceMatcher(None, nc, w).ratio())
        score = max(score, ratio)
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


def snap_name(raw: str | None) -> dict:
    """
    Aggancia una lettura OCR alle rose. Ritorna {name, team, confidence, matched}.
    Se incerto: name = nome normalizzato grezzo, matched=False.
    """
    name = normalize_name(raw)
    if not name:
        return {"name": "", "team": None, "confidence": 0.0, "matched": False}
    h_name, h_score = _best_in_roster(name, ROSTER_HOME)
    a_name, a_score = _best_in_roster(name, ROSTER_AWAY)
    if max(h_score, a_score) >= NAME_RATIO_THRESHOLD:
        if h_score >= a_score:
            return {"name": h_name, "team": "home", "confidence": round(h_score, 2), "matched": True}
        return {"name": a_name, "team": "away", "confidence": round(a_score, 2), "matched": True}
    return {"name": name, "team": None, "confidence": round(max(h_score, a_score), 2), "matched": False}


# =========================================================================== #
# Validazione punteggio / gol                                                 #
# =========================================================================== #
def _plausible(score) -> bool:
    return (
        isinstance(score, (list, tuple)) and len(score) == 2
        and all(isinstance(v, int) and 0 <= v <= MAX_SCORE for v in score)
    )


def is_real_goal(prev_score, curr_score) -> bool:
    """Gol vero solo se +1 ESATTO a una squadra, l'altra invariata, plausibili."""
    if not (_plausible(prev_score) and _plausible(curr_score)):
        return False
    dh = curr_score[0] - prev_score[0]
    da = curr_score[1] - prev_score[1]
    return (dh, da) in ((1, 0), (0, 1))


# =========================================================================== #
# OCR dell'HUD                                                                 #
# =========================================================================== #
class HudReader:
    """OCR sulle regioni dell'HUD di un singolo frame normalizzato."""

    def __init__(self, languages: list[str], min_confidence: float) -> None:
        import easyocr
        logger.info("Inizializzo EasyOCR (lingue=%s)...", languages)
        self.reader = easyocr.Reader(languages, gpu=True)
        self.min_confidence = min_confidence

    @staticmethod
    def _crop(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    def read_region(self, frame: np.ndarray, region, min_conf: float | None = None) -> str:
        crop = self._crop(frame, region)
        if crop.size == 0:
            return ""
        floor = self.min_confidence if min_conf is None else min_conf
        results = self.reader.readtext(crop)
        tokens = [text for _, text, conf in results if conf >= floor]
        return " ".join(tokens).strip()


def parse_score(text: str) -> tuple[int, int] | None:
    """Estrae (home, away) tenendo SOLO numeri plausibili (0..MAX_SCORE)."""
    nums = [int(n) for n in re.findall(r"\d+", text)]
    nums = [n for n in nums if 0 <= n <= MAX_SCORE]
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


def parse_clock(text: str) -> str | None:
    m = re.search(r"(\d{1,3})\s*[:\.]\s*(\d{2})", text)
    return f"{int(m.group(1))}:{m.group(2)}" if m else None


# =========================================================================== #
# Estrazione                                                                   #
# =========================================================================== #
def extract_events(video_path: Path, reader: HudReader, limit: int | None) -> list[dict]:
    events: list[dict] = []
    confirmed_score = None
    pending_score, pending_count = None, 0
    prev_active = ""
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
            home_txt = reader.read_region(frame, config.HUD_REGIONS["active_player_home"], NAME_MIN_CONF)
            away_txt = reader.read_region(frame, config.HUD_REGIONS["active_player_away"], NAME_MIN_CONF)
        except Exception as exc:
            logger.debug("Frame a %.1fs saltato: %s", timestamp, exc)
            continue

        # --- punteggio con debounce ---
        score = parse_score(score_txt)
        if score is not None:
            if list(score) == (list(pending_score) if pending_score else None):
                pending_count += 1
            else:
                pending_score, pending_count = score, 1
            became_official = pending_count >= CONFIRM_N
        else:
            became_official = False

        # --- nomi agganciati alla rosa ---
        snap_h = snap_name(home_txt)
        snap_a = snap_name(away_txt)
        h_ok, a_ok = snap_h["matched"], snap_a["matched"]
        if h_ok and not a_ok:
            active, possession_certain = snap_h, True
        elif a_ok and not h_ok:
            active, possession_certain = snap_a, True
        elif h_ok and a_ok:
            # Entrambe leggibili: dalla SOLA HUD il possesso e' ambiguo
            # (le targhette mostrano i CONTROLLATI, non chi ha la palla).
            # La conferma deve arrivare dalla Fase 1b (tracking visivo).
            active = snap_h if snap_h["confidence"] >= snap_a["confidence"] else snap_a
            possession_certain = False
        else:
            active, possession_certain = snap_h, False
        team = active["team"] or ("home" if active is snap_h else "away")
        active_name = active["name"]

        # --- classificazione ---
        event_type = "idle"
        if became_official and is_real_goal(confirmed_score, pending_score):
            event_type = "goal"
        elif active_name and active_name != prev_active:
            event_type = "pass"   # cambio del giocatore controllato

        if became_official:
            confirmed_score = pending_score

        if event_type != "idle":
            events.append({
                "t": round(timestamp, 2),
                "type": event_type,
                "player": active_name or "il giocatore",
                "player_team": team,
                "possession_certain": possession_certain,
                "player_resolved": active["matched"],
                "name_confidence": active["confidence"],
                "player_home": snap_h["name"],
                "player_away": snap_a["name"],
                "importance": config.EVENT_IMPORTANCE[event_type],
                "score": list(confirmed_score) if confirmed_score else None,
                "clock": parse_clock(clock_txt),
                "source": "ocr",
            })
        if active_name:
            prev_active = active_name

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1 - Estrazione eventi da HUD")
    parser.add_argument("--video", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    if not (ROSTER_HOME or ROSTER_AWAY):
        logger.warning("config.ROSTER_HOME/ROSTER_AWAY vuote: i nomi non verranno corretti.")

    reader = HudReader(config.OCR_LANGUAGES, config.OCR_MIN_CONFIDENCE)
    events = extract_events(video_path, reader, args.limit)

    out_path = config.EVENTS_DIR / f"{video_path.stem}.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    by_type = {t: sum(1 for e in events if e["type"] == t) for t in config.EVENT_TYPES}
    by_type = {t: n for t, n in by_type.items() if n}
    logger.info("Estratti %d eventi (%s) -> %s", len(events), by_type, out_path)
    logger.info("Fase 1 completata.")


if __name__ == "__main__":
    main()