"""
01_extract_events.py  (fixed, self-contained version)
=========================================================
PHASE 1 - Event extraction from the HUD (Computer Vision / OCR).

Fixes over the previous version:
  * GOAL: it is no longer enough that the score sum rises. It requires EXACTLY
    +1 to one team, the other unchanged, plausible scores and a CONFIRMED change
    over several frames (debounce). Kills the "phantom goals" from misreads (e.g. 16->18).
  * NAMES: every nameplate is snapped to the roster (config.ROSTER_HOME/AWAY)
    with exact/substring/fuzzy match. "INHOS"->MARQUINHOS, "NDRO"->ALEX SANDRO.
    Fragments that are too short stay marked uncertain (player_resolved=False).
  * TEAM: the actor is no longer "always home". It is derived from the roster of
    the snapped name; if both nameplates are readable the possession is ambiguous
    from the HUD alone and must be confirmed in Phase 1b (possession_certain=False).

All the validation/snapping logic is HERE inside: no extra file.

Output: features/events/<video_name>.json
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

# --- parameters (from the config, with fallback if undefined) -------------- #
MAX_SCORE = getattr(config, "MAX_PLAUSIBLE_SCORE", 9)
CONFIRM_N = getattr(config, "GOAL_CONFIRM_FRAMES", 2)
NAME_MIN_CONF = getattr(config, "OCR_NAME_MIN_CONFIDENCE", 0.15)
NAME_MIN_FRAGMENT = 3  # below this length the match is NOT attempted ("RO","AA")
NAME_RATIO_THRESHOLD = 0.62  # minimum similarity to accept a "fuzzy" match
# Note: rosters, HUD regions and active side are NOT hard-wired here: they are
# read from the active HUD PROFILE (config.ROSTER_HOME/ROSTER_AWAY/HUD_REGIONS/
# HUD_ACTIVE_SIDE), chosen at runtime based on the resolution or on --profile.


# =========================================================================== #
# Snapping the names to the roster (formerly events_cleanup)                  #
# =========================================================================== #
def normalize_name(raw: str | None) -> str:
    """UPPERCASE, remove jersey numbers and symbols, collapse spaces."""
    if not raw:
        return ""
    name = re.sub(r"[^A-Za-zÀ-ÿ ]", " ", raw).upper()
    return re.sub(r"\s+", " ", name).strip()


def _compact(s: str) -> str:
    """
    Remove all spaces from a string for more robust matching.
    """
    return s.replace(" ", "")


def _best_in_roster(name: str, roster: list[str]) -> tuple[str | None, float]:
    """Best candidate in the roster and its score [0,1]."""
    if not name:
        return None, 0.0
    if name in roster:
        return name, 1.0
    nc = _compact(name)
    best, best_score = None, 0.0
    for cand in roster:
        cc = _compact(cand)
        score = 0.0
        # (a) Containment (truncated fragments: "INHOS" in "MARQUINHOS").
        if len(name) >= NAME_MIN_FRAGMENT and (nc in cc or cc in nc):
            shorter, longer = sorted((len(nc), len(cc)))
            score = 0.6 + 0.4 * (shorter / max(longer, 1))
        # (b) Fuzzy similarity on the whole string and on the single words.
        ratio = SequenceMatcher(None, nc, cc).ratio()
        for w in cand.split():
            ratio = max(ratio, SequenceMatcher(None, nc, w).ratio())
        score = max(score, ratio)
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


def snap_name(raw: str | None) -> dict:
    """
    Snap an OCR reading to the rosters. Returns {name, team, confidence, matched}.
    If uncertain: name = raw normalized name, matched=False.
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
# Score / goal validation                                                     #
# =========================================================================== #
def _plausible(score) -> bool:
    """
    Check if the given score tuple is valid and within realistic bounds.
    """
    return (
        isinstance(score, (list, tuple))
        and len(score) == 2
        and all(isinstance(v, int) and 0 <= v <= MAX_SCORE for v in score)
    )


def is_real_goal(prev_score, curr_score) -> bool:
    """Real goal only if EXACTLY +1 to one team, the other unchanged, plausible."""
    if not (_plausible(prev_score) and _plausible(curr_score)):
        return False
    dh = curr_score[0] - prev_score[0]
    da = curr_score[1] - prev_score[1]
    return (dh, da) in ((1, 0), (0, 1))


# =========================================================================== #
# HUD OCR                                                                     #
# =========================================================================== #
class HudReader:
    """OCR on the HUD regions of a single normalized frame."""

    def __init__(self, languages: list[str], min_confidence: float) -> None:
        """
        Initialize the EasyOCR reader with the specified languages and confidence threshold.
        """
        import easyocr

        logger.info("Inizializzo EasyOCR (lingue=%s)...", languages)
        self.reader = easyocr.Reader(languages, gpu=True)
        self.min_confidence = min_confidence

    @staticmethod
    def _crop(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        """
        Crop a normalized frame using the given fractional region coordinates.
        """
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
        """
        Read text from a specific region of the frame using OCR.
        Applies resizing and optional allowlists to improve read accuracy.
        """
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
    Extract (home, away) keeping ONLY plausible numbers (0..MAX_SCORE) and taking
    the FIRST and the LAST: this way the icon/trophy between the two scores (read
    sometimes as '8') is discarded instead of entering the result.
    """
    nums = [int(n) for n in re.findall(r"\d+", text)]
    nums = [n for n in nums if 0 <= n <= MAX_SCORE]
    if len(nums) >= 2:
        return nums[0], nums[-1]
    return None


def parse_clock(text: str) -> str | None:
    """
    Extract a valid mm:ss clock time from the raw OCR text.
    """
    m = re.search(r"(\d{1,3})\s*[:\.]\s*(\d{2})", text)
    return f"{int(m.group(1))}:{m.group(2)}" if m else None


# =========================================================================== #
# Possession heuristic from the HUD nameplates                               #
# =========================================================================== #
class PossessionHeuristic:
    """
    Infers the possession from the HUD nameplates ALONE, when the active side is
    not forced (HUD_ACTIVE_SIDE=None). The nameplates show the SELECTED player of
    each team, so the possession must be inferred from HOW they change over time.
    Three signals, in order:

      1) START: the first side to change nameplate has the ball (the game
         updates the selected player of whoever is maneuvering).
      2) FREQUENCY: in the sliding window the side that changes nameplate much
         more often than the other has the ball (a margin of MIN_MARGIN changes
         is required to avoid false overtakes).
      3) SILENCE: if BOTH sides have been still for INACTIVITY_S and the
         possession has lasted at least MIN_DURATION_S, a likely change of side
         that signals 1-2 did not see: it flips (prudent fallback).
    """

    WINDOW_S = 2.0  # width of the changes window (2s = responsive to fast changes)
    MIN_MARGIN = 1  # extra changes required for the overtake (signal 2)
    INACTIVITY_S = 2.0  # how long a side has been "still" (signal 3)
    MIN_DURATION_S = 8.0  # minimum possession duration before the flip (signal 3)

    def __init__(self) -> None:
        """
        Initialize the heuristic tracker with empty change history.
        """
        self.current: str | None = None
        self._changes: deque[tuple[float, str]] = deque()  # recent (t, side)
        self._last_change = {"home": 0.0, "away": 0.0}
        self._start_t: float | None = None  # start of the current possession

    def update(self, t: float, home_changed: bool, away_changed: bool) -> str | None:
        """Record the nameplate changes at time t and return the current possession."""
        for side, changed in (("home", home_changed), ("away", away_changed)):
            if changed:
                self._changes.append((t, side))
                self._last_change[side] = t
        while self._changes and self._changes[0][0] < t - self.WINDOW_S:
            self._changes.popleft()

        prev = self.current

        # 1) Start: the first side to move takes the possession.
        if self.current is None and (home_changed or away_changed):
            self.current = "home" if home_changed else "away"

        # 2) Frequency of the changes in the sliding window.
        n_home = sum(1 for _, side in self._changes if side == "home")
        n_away = sum(1 for _, side in self._changes if side == "away")
        if self.current == "home" and n_away > n_home + self.MIN_MARGIN:
            self.current = "away"
        elif self.current == "away" and n_home > n_away + self.MIN_MARGIN:
            self.current = "home"

        # Signal 3 (silence flip) REMOVED: it caused false possession changes
        # every frame when both nameplates were still, generating cascades of
        # spurious events. The possession now changes ONLY when the nameplates
        # show real activity (signals 1 and 2).

        if self.current != prev:
            self._start_t = t
        return self.current


# =========================================================================== #
# Extraction                                                                  #
# =========================================================================== #
def extract_events(
    video_path: Path, reader: HudReader, limit: int | None, profile_name: str = "auto"
) -> list[dict]:
    """
    Process the video frame-by-frame and extract HUD events like passes, turnovers, and goals.
    Uses OCR to read the score and active player nameplates.
    """
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

        # Choose and apply the HUD PROFILE only once, on the first normalized
        # frame (based on the resolution, or forced by --profile).
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

        # --- score with debounce ---
        # A score becomes "official" only if read identical for CONFIRM_N
        # consecutive frames: filters the OCR flicker (e.g. "1-0" read as "7-0").
        score = parse_score(score_txt)
        if score is not None:
            # parse_score always returns a (home, away) tuple: the direct
            # comparison with pending_score (tuple or None on the first pass) is enough.
            if score == pending_score:
                pending_count += 1
            else:
                pending_score, pending_count = score, 1
            became_official = pending_count >= CONFIRM_N
        else:
            became_official = False

        # --- names snapped to the roster (both nameplates are ALWAYS on screen) ---
        snap_h = snap_name(home_txt)
        snap_a = snap_name(away_txt)

        # --- possession and active player ------------------------------- #
        home_changed = bool(snap_h["name"]) and snap_h["name"] != prev_home
        away_changed = bool(snap_a["name"]) and snap_a["name"] != prev_away

        if config.HUD_ACTIVE_SIDE:
            current_possession = config.HUD_ACTIVE_SIDE
        else:
            current_possession = heuristic.update(timestamp, home_changed, away_changed)

        # Events are recorded ONLY on the side in possession.
        active = snap_h if current_possession == "home" else snap_a
        active_side_changed = home_changed if current_possession == "home" else away_changed

        team = active["team"] or ("home" if active is snap_h else "away")
        active_name = active["name"]

        # --- classification ---
        event_type = "idle"

        # If the possession just changed, we record a turnover.
        if last_possession and current_possession != last_possession:
            event_type = "turnover"
        elif became_official and is_real_goal(confirmed_score, pending_score):
            event_type = "goal"
        elif active_name and active_name != prev_active and active_side_changed:
            event_type = "pass"  # change of the controlled player ON THE SIDE IN POSSESSION

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
    """
    Main entry point for Phase 1.
    Initializes OCR, processes the video, and saves the extracted events to JSON.
    """
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
