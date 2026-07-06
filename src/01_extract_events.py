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
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from utils import ensure_dir, get_logger, iter_sampled_frames, save_json

logger = get_logger("fase1_eventi")

# --- parametri (dalla config, con fallback se non definiti) ---------------- #
MAX_SCORE = getattr(config, "MAX_PLAUSIBLE_SCORE", 9)
CONFIRM_N = getattr(config, "GOAL_CONFIRM_FRAMES", 2)
NAME_MIN_CONF = getattr(config, "OCR_NAME_MIN_CONFIDENCE", 0.15)
NAME_MIN_FRAGMENT = 3  # sotto questa lunghezza NON si tenta il match ("RO","AA")
NAME_RATIO_THRESHOLD = 0.62  # similarita' minima per accettare un match "fuzzy"
# Nota: rose, regioni HUD e lato attivo NON sono cablati qui: si leggono dal
# PROFILO HUD attivo (config.ROSTER_HOME/ROSTER_AWAY/HUD_REGIONS/HUD_ACTIVE_SIDE),
# scelto a runtime in base alla risoluzione o a --profile.


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
    h_name, h_score = _best_in_roster(name, config.ROSTER_HOME)
    a_name, a_score = _best_in_roster(name, config.ROSTER_AWAY)
    if max(h_score, a_score) >= NAME_RATIO_THRESHOLD:
        if h_score >= a_score:
            return {
                "name": h_name,
                "team": "home",
                "confidence": round(h_score, 2),
                "matched": True,
            }
        return {"name": a_name, "team": "away", "confidence": round(a_score, 2), "matched": True}
    return {
        "name": name,
        "team": None,
        "confidence": round(max(h_score, a_score), 2),
        "matched": False,
    }


# =========================================================================== #
# Validazione punteggio / gol                                                 #
# =========================================================================== #
def _plausible(score) -> bool:
    return (
        isinstance(score, (list, tuple))
        and len(score) == 2
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
# OCR dell'HUD                                                                #
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
        return frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]

    def read_region(
        self,
        frame: np.ndarray,
        region,
        min_conf: float | None = None,
        allowlist: str | None = None,
        paragraph: bool = False,
        resize: int = 1,
    ) -> str:
        crop = self._crop(frame, region)
        if crop.size == 0:
            return ""
        if resize > 1:
            import cv2

            crop = cv2.resize(crop, (0, 0), fx=resize, fy=resize)

        floor = self.min_confidence if min_conf is None else min_conf

        kwargs = {}
        if allowlist:
            kwargs["allowlist"] = allowlist
        if paragraph:
            kwargs["paragraph"] = paragraph

        results = self.reader.readtext(crop, **kwargs)

        if paragraph:
            # When paragraph=True, result format is [[bbox, text]]
            # Sometimes it might return multiple paragraphs or strings
            tokens = [res[1] for res in results]
        else:
            tokens = [text for _, text, conf in results if conf >= floor]

        return " ".join(tokens).strip()


def parse_score(text: str) -> tuple[int, int] | None:
    """
    Estrae (home, away) tenendo SOLO numeri plausibili (0..MAX_SCORE) e prendendo
    il PRIMO e l'ULTIMO: cosi' l'icona/trofeo in mezzo ai due punteggi (letta a
    volte come '8') viene scartata invece di entrare nel risultato.
    """
    nums = [int(n) for n in re.findall(r"\d+", text)]
    nums = [n for n in nums if 0 <= n <= MAX_SCORE]
    if len(nums) >= 2:
        return nums[0], nums[-1]
    return None


def parse_clock(text: str) -> str | None:
    m = re.search(r"(\d{1,3})\s*[:\.]\s*(\d{2})", text)
    return f"{int(m.group(1))}:{m.group(2)}" if m else None


# =========================================================================== #
# Euristica del possesso dalle targhette HUD                                  #
# =========================================================================== #
class PossessionHeuristic:
    """
    Deduce il possesso dalle SOLE targhette HUD, quando il lato attivo non e'
    forzato (HUD_ACTIVE_SIDE=None). Le targhette mostrano il giocatore
    SELEZIONATO di ciascuna squadra, quindi il possesso va inferito da COME
    cambiano nel tempo. Tre segnali, in ordine:

      1) AVVIO: il primo lato che cambia targhetta ha la palla (il gioco
         aggiorna il giocatore selezionato di chi sta manovrando).
      2) FREQUENZA: nella finestra scorrevole il lato che cambia targhetta
         molto piu' spesso dell'altro ha la palla (serve un margine di
         MIN_MARGIN cambi per evitare falsi sorpassi).
      3) SILENZIO: se ENTRAMBI i lati sono fermi da INACTIVITY_S e il possesso
         dura da almeno MIN_DURATION_S, probabile cambio di fronte che i
         segnali 1-2 non hanno visto: si inverte (fallback prudente).
    """

    WINDOW_S = 2.0  # ampiezza finestra dei cambi (2s = reattiva ai cambi veloci)
    MIN_MARGIN = 1  # cambi in piu' richiesti per il sorpasso (segnale 2)
    INACTIVITY_S = 2.0  # da quanti secondi un lato e' "fermo" (segnale 3)
    MIN_DURATION_S = 8.0  # durata minima del possesso prima del flip (segnale 3)

    def __init__(self) -> None:
        self.current: str | None = None
        self._changes: deque[tuple[float, str]] = deque()  # (t, lato) recenti
        self._last_change = {"home": 0.0, "away": 0.0}
        self._start_t: float | None = None  # inizio possesso corrente

    def update(self, t: float, home_changed: bool, away_changed: bool) -> str | None:
        """Registra i cambi targhetta al tempo t e ritorna il possesso corrente."""
        for side, changed in (("home", home_changed), ("away", away_changed)):
            if changed:
                self._changes.append((t, side))
                self._last_change[side] = t
        while self._changes and self._changes[0][0] < t - self.WINDOW_S:
            self._changes.popleft()

        prev = self.current

        # 1) Avvio: il primo lato che si muove prende il possesso.
        if self.current is None and (home_changed or away_changed):
            self.current = "home" if home_changed else "away"

        # 2) Frequenza dei cambi nella finestra scorrevole.
        n_home = sum(1 for _, side in self._changes if side == "home")
        n_away = sum(1 for _, side in self._changes if side == "away")
        if self.current == "home" and n_away > n_home + self.MIN_MARGIN:
            self.current = "away"
        elif self.current == "away" and n_home > n_away + self.MIN_MARGIN:
            self.current = "home"

        # Segnale 3 (silence flip) RIMOSSO: causava falsi cambi di possesso
        # ogni frame quando entrambe le targhette erano ferme, generando
        # cascate di eventi spuri. Il possesso ora cambia SOLO quando le
        # targhette mostrano attivita' reale (segnali 1 e 2).

        if self.current != prev:
            self._start_t = t
        return self.current


# =========================================================================== #
# Estrazione                                                                   #
# =========================================================================== #
def extract_events(
    video_path: Path, reader: HudReader, limit: int | None, profile_name: str = "auto"
) -> list[dict]:
    events: list[dict] = []
    confirmed_score = None
    pending_score, pending_count = None, 0
    prev_active = ""
    prev_home, prev_away = "", ""
    processed = 0
    profile_applied = False

    heuristic = PossessionHeuristic()
    current_possession: str | None = None
    last_possession: str | None = None

    for timestamp, frame in tqdm(
        iter_sampled_frames(video_path, config.FRAMES_PER_SECOND), desc="OCR HUD"
    ):
        if limit and processed >= limit:
            break
        processed += 1

        # Sceglie e applica il PROFILO HUD una volta sola, sul primo frame
        # normalizzato (in base alla risoluzione, o forzato da --profile).
        if not profile_applied:
            h, w = frame.shape[:2]
            pname, prof = config.select_profile(w, h, profile_name)
            config.apply_profile(prof)
            logger.info(
                "Profilo HUD: '%s' (frame %dx%d, lato attivo=%s)",
                pname,
                w,
                h,
                config.HUD_ACTIVE_SIDE,
            )
            profile_applied = True

        try:
            score_txt = reader.read_region(
                frame,
                config.HUD_REGIONS["score"],
                allowlist="0123456789 ",
                paragraph=True,
                resize=3,
            )
            clock_txt = reader.read_region(frame, config.HUD_REGIONS["clock"])
            home_txt = reader.read_region(
                frame, config.HUD_REGIONS["active_player_home"], NAME_MIN_CONF
            )
            away_txt = reader.read_region(
                frame, config.HUD_REGIONS["active_player_away"], NAME_MIN_CONF
            )
        except Exception as exc:
            logger.debug("Frame a %.1fs saltato: %s", timestamp, exc)
            continue

        # --- punteggio con debounce ---
        # Un punteggio diventa "ufficiale" solo se letto identico per CONFIRM_N
        # frame consecutivi: filtra il flicker dell'OCR (es. "1-0" letto "7-0").
        score = parse_score(score_txt)
        if score is not None:
            # parse_score ritorna sempre una tupla (home, away): il confronto
            # diretto con pending_score (tupla o None al primo giro) basta.
            if score == pending_score:
                pending_count += 1
            else:
                pending_score, pending_count = score, 1
            became_official = pending_count >= CONFIRM_N
        else:
            became_official = False

        # --- nomi agganciati alla rosa (entrambe le targhette sono SEMPRE a video) ---
        snap_h = snap_name(home_txt)
        snap_a = snap_name(away_txt)

        # --- possesso e giocatore attivo -------------------------------- #
        home_changed = bool(snap_h["name"]) and snap_h["name"] != prev_home
        away_changed = bool(snap_a["name"]) and snap_a["name"] != prev_away

        if config.HUD_ACTIVE_SIDE:
            current_possession = config.HUD_ACTIVE_SIDE
        else:
            current_possession = heuristic.update(timestamp, home_changed, away_changed)

        # Gli eventi si registrano SOLO sul lato in possesso.
        active = snap_h if current_possession == "home" else snap_a
        active_side_changed = home_changed if current_possession == "home" else away_changed

        team = active["team"] or ("home" if active is snap_h else "away")
        active_name = active["name"]

        # --- classificazione ---
        event_type = "idle"

        # Se il possesso e' appena cambiato, registriamo un turnover.
        if last_possession and current_possession != last_possession:
            event_type = "turnover"
        elif became_official and is_real_goal(confirmed_score, pending_score):
            event_type = "goal"
        elif active_name and active_name != prev_active and active_side_changed:
            event_type = "pass"  # cambio del giocatore controllato SUL LATO IN POSSESSO

        last_possession = current_possession

        if became_official:
            confirmed_score = pending_score

        if event_type != "idle":
            events.append(
                {
                    "t": round(timestamp, 2),
                    "type": event_type,
                    "player": active_name or "il giocatore",
                    "player_team": team,
                    "possession": current_possession,
                    "player_resolved": active["matched"],
                    "name_confidence": active["confidence"],
                    "player_home": snap_h["name"],
                    "player_away": snap_a["name"],
                    "importance": config.EVENT_IMPORTANCE[event_type],
                    "score": list(confirmed_score) if confirmed_score else None,
                    "clock": parse_clock(clock_txt),
                    "source": "ocr",
                }
            )
        if active_name:
            prev_active = active_name
        prev_home, prev_away = snap_h["name"], snap_a["name"]

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1 - Estrazione eventi da HUD")
    parser.add_argument("--video", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--profile",
        default="auto",
        help="Profilo HUD: 'auto' (per risoluzione) o un nome di config.HUD_PROFILES.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    reader = HudReader(config.OCR_LANGUAGES, config.OCR_MIN_CONFIDENCE)
    events = extract_events(video_path, reader, args.limit, args.profile)

    out_path = config.EVENTS_DIR / f"{video_path.stem}.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    by_type = {t: sum(1 for e in events if e["type"] == t) for t in config.EVENT_TYPES}
    by_type = {t: n for t, n in by_type.items() if n}
    logger.info("Estratti %d eventi (%s) -> %s", len(events), by_type, out_path)
    logger.info("Fase 1 completata.")


if __name__ == "__main__":
    main()
