"""
config.py
=========
Centralized configuration of the video-game commentary pipeline.

All paths are RELATIVE to the project root. The only file to touch to change
behavior (video rotation/crop, HUD regions, thresholds, hyperparameters,
templates) is this one.

FLAT layout: this file and the "NN_*.py" scripts live in the same folder,
which is also the project root. The data folders are created HERE inside.

"Relay" pipeline:
    Phase 0  (norm.)    -> straightens the video (rotation) and crops the black bars.
    Phase 1  (events)   -> HUD OCR (score + nameplates) -> JSON event log.
    Phase 1b (visual)   -> YOLO + OpenCV: shots, saves, corners.
    Phase 1c (audio)    -> crowd excitement (Librosa) + transcription (Whisper).
    Phase 2  (script)   -> events -> commentary text (template).
    Phase 2b (LLM)      -> events -> commentary text (LLM, alternative).
    Phase 3  (prosody)  -> TRAINS the event->prosody model (the CONTRIBUTION).
    Phase 4  (synthesis)-> text + prosody -> expressive audio (Coqui XTTS v2).
    Phase 5  (eval)     -> A/B study on listeners + statistical test.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Commentary language                                                         #
# --------------------------------------------------------------------------- #
# ISO 639-1 code. Must be supported by XTTS v2:
#   en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hu, hi.
# All the components (templates, LLM prompt, TTS voice, Whisper) derive from here.
# The COMMENTARY_LANGUAGE environment variable overrides the default: it lets
# run_pipeline.py propagate the language to the subprocesses (which re-import
# config.py from scratch).
import os as _os
LANGUAGE: str = _os.environ.get("COMMENTARY_LANGUAGE", "it")
SUPPORTED_LANGUAGES: list[str] = ["it", "en"]

# Placeholder for "unknown player", per language.
UNKNOWN_PLAYER: dict[str, str] = {
    "it": "il giocatore",
    "en": "the player",
}


def get_unknown_player(lang: str | None = None) -> str:
    """
    Return the localized placeholder for the unknown player.
    It falls back to English if the language is not found.
    """
    return UNKNOWN_PLAYER.get(lang or LANGUAGE, UNKNOWN_PLAYER["en"])

# --------------------------------------------------------------------------- #
# Paths (FLAT layout: the root is this file's folder)                         #
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
GAMEPLAY_DIR: Path = DATA_DIR / "raw" / "gameplay"  # gameplay videos (input)
COMMENTARY_DIR: Path = DATA_DIR / "raw" / "commentary"  # real commentary clips
PROSODY_ANNOTATIONS: Path = DATA_DIR / "raw" / "prosody_annotations.csv"

FEATURES_DIR: Path = PROJECT_ROOT / "features"
NORMALIZED_DIR: Path = FEATURES_DIR / "normalized"  # normalized frames/previews
EVENTS_DIR: Path = FEATURES_DIR / "events"  # JSON event log (Phase 1)
SCRIPTS_DIR: Path = FEATURES_DIR / "scripts"  # commentary text (Phase 2)
PROSODY_DIR: Path = FEATURES_DIR / "prosody"  # prosody dataset (Phase 3)
PROSODY_DATASET: Path = PROSODY_DIR / "prosody_dataset.npz"

MODELS_DIR: Path = PROJECT_ROOT / "models"
PROSODY_MODEL_PATH: Path = MODELS_DIR / "prosody_mlp.pt"
PROSODY_SCALER_PATH: Path = MODELS_DIR / "prosody_scaler.joblib"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
AUDIO_OUT_DIR: Path = OUTPUTS_DIR / "audio"
STUDY_DIR: Path = OUTPUTS_DIR / "study"

# --------------------------------------------------------------------------- #
# Phase 0 - Video normalization (rotation + letterbox crop)                   #
# --------------------------------------------------------------------------- #
# Videos recorded from a phone may have a rotation flag that OpenCV does NOT
# apply, and black side bars. Here we straighten and crop, once, before
# everything else.
#
# VIDEO_ROTATION: "auto" reads the rotation from the metadata (cv2/ffprobe);
#   otherwise forces a clockwise value in degrees: 0, 90, 180, 270.
VIDEO_ROTATION: str | int = "auto"

# LETTERBOX_CROP: "auto" detects the black bars on the first frame and crops;
#   None disables the crop; or a normalized tuple (x1, y1, x2, y2) in [0,1]
#   to fix the play area manually (recommended after calibration).
LETTERBOX_CROP: str | None | tuple[float, float, float, float] = "auto"

# Below this intensity (0-255) a pixel is considered "black" (border).
LETTERBOX_BLACK_THRESHOLD: int = 18
# Minimum fraction of non-black pixels for a row/column to be "content".
LETTERBOX_MIN_FILL: float = 0.05

# --------------------------------------------------------------------------- #
# Phase 1 - Event extraction (HUD OCR)                                        #
# --------------------------------------------------------------------------- #
# The HUD changes slowly: 2 fps is enough and keeps the OCR light.
FRAMES_PER_SECOND: float = 2.0

# Phase 1b (visual) samples MORE densely than the OCR: a shot lasts a few tenths
# of a second and at 2 fps it "disappears" between two samples (the ball travels
# more than max_ball_jump_px and the motion is discarded as a glitch). At 8 fps
# the path between two samples stays below the anti-glitch threshold and the shot is seen.
VISUAL_FRAMES_PER_SECOND: float = 8.0

# HUD regions in NORMALIZED [0,1] coordinates (x1, y1, x2, y2), referred to the
# ALREADY normalized frame (straightened + cropped). They must be calibrated on
# YOUR video with: python 01_calibrate_hud.py --video <clip> --ocr
# (these are typical starting values for the EA FC HUD, score at the top left
#  and two name nameplates at the bottom on the two sides).
HUD_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # Calibrated on the REAL frames of this clip (1920x886, letterbox crop):
    "score": (0.115, 0.025, 0.205, 0.10),  # ONLY the two digits "1 .. 0"
    "clock": (0.27, 0.02, 0.37, 0.11),  # "16:55"
    "active_player_home": (0.035, 0.90, 0.22, 0.985),  # "11 RAPHINHA" (Brazil, left)
    "active_player_away": (0.74, 0.88, 0.95, 0.99),  # "BELLEGARDE 10" (Haiti, right)
}

# Nameplate side to use as "player in possession" when there is NO reliable
# possession source (Phase 1b/visual). The two nameplates are ALWAYS both on
# screen (they show the SELECTED player of each team), so from the HUD alone the
# possession is not deducible.
#   "home" / "away" -> force the side (use it when you KNOW who attacks in the clip).
#   None            -> not forced: follow the changing side, uncertain possession.
HUD_ACTIVE_SIDE: str | None = None  # Set to None to enable dynamic possession change

# Team codes shown in the score (to disambiguate home/away from the OCR).
TEAM_CODES: dict[str, str] = {"home": "BRA", "away": "HAI"}

# SEPARATE rosters per team. Used to snap the names read by the OCR to the known
# roster and to derive the actor's team. Names in UPPERCASE.
# Put here the REAL rosters of the clip's two teams.
ROSTER_HOME: list[str] = [  # Brazil (home) - COMPLETE with the clip's REAL roster
    "RAPHINHA",
    "VINICIUS",
    "RODRYGO",
    "CASEMIRO",
    "BRUNO GUIMARAES",
    "MARQUINHOS",
    "DANILO",
    "ALEX SANDRO",
    "BREMER",
    "ANDREAS PEREIRA",
    "LUIZ HENRIQUE",
    "BARELLA",
    "BASTONI",
    "VINI JR",
    "GABRIEL",
    "WESLEY",
    "EDERSON",
    "CUNHA",  # were missing from the OCR debug
]
ROSTER_AWAY: list[str] = [  # Haiti (away) - COMPLETE with the clip's REAL roster
    "BELLEGARDE",
    "JEAN JACQUES",
    "DELCROIX",
    "PIERRE",
    "PIERROT",
    "NAZON",
    "SAINTE",
    "PROVIDENCE",
    "CASIMIR",
    "PLACIDE",
    "MARCUS",  # were missing from the OCR debug
]

# Combined ROSTER player -> team ("home"/"away"), derived from the two rosters.
# Kept for backward compatibility with whoever still uses the single map.
ROSTER: dict[str, str] = {
    **{n: "home" for n in ROSTER_HOME},
    **{n: "away" for n in ROSTER_AWAY},
}

OCR_LANGUAGES: list[str] = ["en"]  # languages for EasyOCR
OCR_MIN_CONFIDENCE: float = 0.30  # below this confidence the reading is discarded
# Lower threshold for the NAMES: it is better to read the whole surname (even at
# low confidence) and then snap it to the roster, instead of discarding pieces
# and ending up with fragments like "NDRO"/"INHOS".
OCR_NAME_MIN_CONFIDENCE: float = 0.15

# A football score above this value is almost certainly broken OCR:
# out-of-range numbers are ignored in the score parsing.
MAX_PLAUSIBLE_SCORE: int = 9
# A score change must persist for N consecutive readings before counting as a
# goal (debounce against the OCR flicker).
GOAL_CONFIRM_FRAMES: int = 2

# --------------------------------------------------------------------------- #
# Event importance (0 = trivial, 1 = key). Guides the prosody.                 #
# --------------------------------------------------------------------------- #
EVENT_IMPORTANCE: dict[str, float] = {
    "goal": 1.00,
    "save": 0.80,  # goalkeeper save
    "shot_on_goal": 0.85,  # shot on target
    "shot_off": 0.60,  # shot off target / blocked
    "corner": 0.45,
    "free_kick": 0.50,
    "turnover": 0.55,  # ball lost / interception
    "foul": 0.40,
    "carry": 0.20,  # carrying / dribbling forward
    "pass": 0.25,
    "idle": 0.10,  # midfield play
}

# SINGLE list used for the one-hot, both in training (Phase 3) and synthesis
# (Phase 4). Consistency guaranteed: each event type has its own column.
EVENT_TYPES: list[str] = [
    "goal",
    "save",
    "shot_on_goal",
    "shot_off",
    "corner",
    "free_kick",
    "turnover",
    "foul",
    "carry",
    "pass",
    "idle",
]

# --------------------------------------------------------------------------- #
# Phase 1b - Visual module (YOLO + OpenCV)                                     #
# --------------------------------------------------------------------------- #
YOLO_MODEL: str = "models/best_finetuned.pt"  # fine-tuned FIFA model (new graphics, imgsz 960)
# The FIFA model uses different names than COCO. We map them to the project's internal names.
YOLO_CLASS_MAP: dict[str, str] = {
    "player": "person",  # players
    # The goalkeeper stays a SEPARATE class: it is needed to detect SAVES
    # (fast ball stopping near the keeper = parried/caught).
    "goalkeeper": "goalkeeper",
    "referee": "referee",  # referee
    "ball": "sports_ball",  # ball
    # NB: the model's "goalpost" class was tried as a "goal in frame" signal for
    # the shot and DISCARDED: false positives of the post at midfield brought
    # back the phantom shots (retest 4 clips, 4 Jul 2026).
}
YOLO_CONFIDENCE: float = 0.35
# The ball is small and often blurred by motion blur: with the players'
# threshold (0.35) YOLO sees it in <10% of frames and the speed cannot be
# computed. A dedicated lower threshold: a few more false positives are
# tolerable (the max_ball_jump_px filter discards implausible jumps).
YOLO_BALL_CONFIDENCE: float = 0.15
# YOLO inference resolution. MUST match the model's training resolution:
# best_finetuned.pt is at 960, and at 640 (default) it would lose the gain on
# the small ball (test: ball seen 47% at 960 vs less at 640).
YOLO_IMGSZ: int = 960

# --- Ball possession from the JERSEY COLOR of the player nearest the ball --- #
# HSV range (OpenCV: H 0-179, S 0-255, V 0-255), TUNED on the clip's real frames.
# home = Brazil (yellow); away = Haiti (red, which in HSV straddles 0/180).
TEAM_JERSEY_HSV: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "home": [((20, 90, 90), (38, 255, 255))],
    "away": [((0, 90, 70), (10, 255, 255)), ((168, 90, 70), (179, 255, 255))],
}
# Portion of the player's bounding box to sample (torso = upper-central),
# in box fractions: (x1, y1, x2, y2). Avoids legs/grass and the back number.
JERSEY_SAMPLE_BOX: tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.55)
# Minimum fraction of team-color pixels for the classification to count.
JERSEY_MIN_FILL: float = 0.12
# Margin: the winning color must beat the other by at least this factor.
JERSEY_MIN_MARGIN: float = 1.3
# Maximum radius (px on the normalized frame) within which a player "has" the ball.
POSSESSION_MAX_DIST_PX: float = 90.0
# The possession changes team only after N consistent frames (anti-flicker). At
# VISUAL_FRAMES_PER_SECOND=8, 4 frames = 0.5s: enough to filter the noise.
POSSESSION_CONFIRM_FRAMES: int = 4
# On the radar, during a possession phase, the ball is in a cluster with dots
# of BOTH teams right next to it. To steal the ball, the new dot must beat the
# other by this margin.
POSSESSION_MARGIN_PX: float = 12.0
# Bonus (px) to the team already in possession when comparing the distances:
# further anti-flicker brake in favor of continuity.
POSSESSION_HYSTERESIS_PX: float = 15.0
# PRIMARY possession from the white name above the carrier (direct signal: the
# radar is ambiguous in the clusters). The OCR of that name is costly, so it is
# done every CARRIER_OCR_INTERVAL_S seconds (between two readings the cached name
# holds). The name can be faded/semi-transparent and the reading is flaky
# frame by frame: only OCR on EVERY frame (~0.125s at 8 fps -> ocr_every=1)
# catches the carrier changes in time, keeping the possession transition as
# close as possible to the real one (cost: slower Phase 1b).
CARRIER_OCR_INTERVAL_S: float = 0.125
# Snapping the read name to the roster. TIGHT thresholds: a garbled reading
# ('NEXE'->NEUER at 0.67, 'NES' substring of 'NUNES') must not snap to the
# wrong roster -> better NO match (the cached name holds) than a wrong one,
# which would flip the possession and mess up the attribution.
CARRIER_NAME_MIN_FUZZY: float = 0.80  # minimum similarity for the fuzzy match
CARRIER_NAME_MIN_LEN: int = 4  # minimum fragment for the containment match

# Ball tracking thresholds on the normalized frame.
# The speeds are in PIXELS PER SECOND (px/s), NOT per frame: this way they stay
# valid whatever VISUAL_FRAMES_PER_SECOND is (before they were px/sample at
# 2 fps: 120 px/sample = 240 px/s, 40 = 80).
BALL_TRACKING: dict[str, float] = {
    "min_speed_shot": 240.0,  # px/s: above = possible shot
    "min_speed_pass": 80.0,  # px/s: above = pass, below = possession
    "save_drop_ratio": 0.35,  # save: the speed drops below this fraction
    "near_player_px": 100.0,  # radius (px) for "player near the ball"
    # Ball-goalkeeper radius for SHOT and SAVE. With the fine-tuned detector the
    # keeper localizes well (test: real shot+save with GK at 211-322px;
    # midfield play with GK at ~900px). This distance is the separator.
    "near_goalkeeper_px": 260.0,
    # A SHOT counts only with the keeper REALLY close (towards the goal):
    # distinguishes the real shot from a strong midfield pass (similar speeds
    # but GK far ~900px). Before it was 1200 on the wide camera to compensate
    # the old model's poor GK localization: with the fine-tuned one it can go to 350.
    "shot_goal_view_px": 350.0,
    # A SHOT is a CLEAR acceleration: current speed > prev * this factor. This
    # way it fires even if the ball was already moving in the build-up (real
    # shot: 344 -> 835 px/s), not only from a still ball.
    "shot_accel_ratio": 1.5,
    "max_ball_jump_px": 400.0,  # px BETWEEN TWO SAMPLES: above = tracking glitch
    # For how many seconds the classifier REMEMBERS the last ball state when
    # YOLO loses it: in a save the motion blur hides the ball for several
    # frames and without memory the "speed drop" is never seen.
    "ball_memory_s": 1.0,
    # Memory of the keeper's last POSITION, used ONLY by the save:
    # during the dive YOLO loses the GK right in the decisive frames, but its
    # on-screen position stays valid for a couple of seconds.
    "gk_memory_s": 2.0,
}

# PRISTINE copy of the defaults: apply_profile merges the profile overrides
# always starting from here, so switching profile does not inherit the values
# of the previously applied profile.
_BALL_TRACKING_DEFAULTS: dict[str, float] = dict(BALL_TRACKING)

# Colors of the MARKERS on the minimap/radar (not the jerseys: the radar uses
# its own stylized colors). OpenCV HSV range (H 0-179); each entry is a LIST
# of intervals because red straddles 0/180. Per-profile override with the
# "radar_hsv" key (see HUD_PROFILES).
# Default = bra_hai radar: orange ball, yellow home, red away.
RADAR_HSV: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "ball": [((10, 100, 150), (25, 255, 255))],
    "home": [((25, 30, 150), (45, 255, 255))],
    "away": [((0, 50, 100), (10, 255, 255)), ((160, 50, 100), (180, 255, 255))],
}
_RADAR_HSV_DEFAULTS = dict(RADAR_HSV)

# Two visual events of the SAME type within this window are the same action:
# at 8 fps a save satisfies the condition on several consecutive frames and
# without cooldown it would be commented twice.
VISUAL_EVENT_COOLDOWN_S: float = 3.0

# Field zones in normalized coordinates (to classify the ball position).
FIELD_ZONES: dict[str, tuple[float, float, float, float]] = {
    "penalty_area_home": (0.00, 0.15, 0.20, 0.85),
    "penalty_area_away": (0.80, 0.15, 1.00, 0.85),
    "midfield": (0.30, 0.00, 0.70, 1.00),
    "wing_left": (0.00, 0.00, 1.00, 0.25),
    "wing_right": (0.00, 0.75, 1.00, 1.00),
}

# --------------------------------------------------------------------------- #
# Phase 1c - Auditory module (Audio Energy + Whisper)                          #
# --------------------------------------------------------------------------- #
SAMPLE_RATE: int = 22050  # used by audio and synthesis
# 'small' instead of 'base': on gameplay audio (music + effects over the
# commentary) 'base' produces unreadable transcriptions; 'small' is much more
# accurate in Italian. Cost: ~5x slower on CPU (acceptable for short clips).
WHISPER_MODEL_SIZE: str = "small"  # tiny/base/small/medium/large
WHISPER_LANGUAGE: str = LANGUAGE  # derived from the commentary language
AUDIO_ANALYSIS_WINDOW_S: float = 2.0
# The thresholds partition the excitement score, which is now RE-SCALED to the
# clip percentiles (0 = calmest moment, 1 = most heated): this way the levels
# make sense and "peak" actually fires (before the score stayed ~0.33 and
# never reached 0.75). The neutral value is 0.5.
AUDIO_ENERGY_THRESHOLDS: dict[str, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,  # above 0.75 = "peak"
}
# Minimum dynamic range of the raw score for the percentile re-scaling to
# "stretch" the clip to [0,1]. Below this (FLAT clip, no hot moments), it does
# not stretch: it avoids promoting the background noise to fake "peak"s.
AUDIO_ENERGY_MIN_RANGE: float = 0.20

# --------------------------------------------------------------------------- #
# Phase 2 - Text generation (template)                                         #
# --------------------------------------------------------------------------- #
TEMPLATES_BY_LANG: dict[str, dict[str, list[str]]] = {
    "it": {
        "goal": [
            "GOOOL! Ha segnato {player}!",
            "Rete di {player}! Incredibile!",
            "{player} la mette dentro! GOL!",
        ],
        "save": [
            "Che parata! Il portiere dice di no!",
            "Risponde presente il portiere!",
            "Miracolo del portiere su {player}!",
        ],
        "shot_on_goal": [
            "Che tiro di {player}! Nello specchio!",
            "{player} calcia! Conclusione potente verso la porta!",
        ],
        "shot_off": ["{player} calcia... fuori di poco!", "Tiro di {player}, palla a lato!"],
        "corner": ["Calcio d'angolo. Si prepara {player}.", "Corner, batte {player}."],
        "free_kick": [
            "Punizione per {player}. Posizione interessante.",
            "Calcio di punizione, si incarica {player}.",
        ],
        "turnover": [
            "Palla persa, la recupera {player}.",
            "Se ne impossessa {player}.",
            "Cambio di fronte, ora {player}.",
        ],
        "foul": ["Fallo su {player}! Intervento duro.", "{player} viene atterrato."],
        "carry": [
            "{player} avanza con il pallone.",
            "{player} porta palla.",
            "Conduce {player}.",
        ],
        "pass": ["{player} controlla.", "Apertura per {player}.", "La gioca {player}."],
        "idle": ["Si fa girare il pallone.", "Fase di studio a centrocampo."],
    },
    "en": {
        "goal": [
            "GOAAL! {player} scores!",
            "What a goal by {player}!",
            "{player} puts it in the back of the net! GOAL!",
        ],
        "save": [
            "What a save! The keeper says no!",
            "Brilliant stop by the goalkeeper!",
            "The keeper denies {player}!",
        ],
        "shot_on_goal": [
            "Great shot by {player}! On target!",
            "{player} shoots! Powerful effort towards goal!",
        ],
        "shot_off": ["{player} shoots... just wide!", "Shot by {player}, off target!"],
        "corner": ["Corner kick. {player} to take it.", "Corner, taken by {player}."],
        "free_kick": [
            "Free kick for {player}. Interesting position.",
            "Free kick, {player} steps up.",
        ],
        "turnover": [
            "Ball lost, {player} picks it up.",
            "{player} wins possession.",
            "Change of play, now {player}.",
        ],
        "foul": ["Foul on {player}! Tough challenge.", "{player} is brought down."],
        "carry": [
            "{player} drives forward.",
            "{player} carries the ball.",
            "{player} on the move.",
        ],
        "pass": ["{player} on the ball.", "Played out to {player}.", "{player} with the pass."],
        "idle": ["Passing it around.", "Keeping possession in midfield."],
    },
}


def get_templates(lang: str | None = None) -> dict[str, list[str]]:
    """
    Return the templates for the requested language (default: LANGUAGE).
    Fallback: English if the language has no dedicated templates.
    """
    return TEMPLATES_BY_LANG.get(lang or LANGUAGE, TEMPLATES_BY_LANG["en"])


# Backward compatibility: whoever imports config.TEMPLATES gets the templates of
# the current language. For the full dictionary use TEMPLATES_BY_LANG.
TEMPLATES: dict[str, list[str]] = get_templates()

# --------------------------------------------------------------------------- #
# Phase 2b - Text generation with LLM                                          #
# --------------------------------------------------------------------------- #
LLM_PROVIDER: str = "ollama"  # "ollama" (local) | "openai" | "anthropic"
LLM_CONFIG: dict[str, dict] = {
    # gemma3:4b: model actually installed locally; fits comfortably in the
    # 7.6 GB of available RAM and stays fast on CPU for the lines.
    "ollama": {"model": "gemma3:4b", "base_url": "http://localhost:11434"},
    "openai": {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"model": "claude-haiku-4-5-20251001", "api_key_env": "ANTHROPIC_API_KEY"},
    # Groq: OpenAI-compatible endpoint, free tier. Llama 70B runs on their
    # servers (no local RAM), much more capable than the 3-4B that fits in the
    # machine's 7.6 GB -> fewer style/role hallucinations.
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
    },
}
LLM_SYSTEM_PROMPTS: dict[str, str] = {
    "it": (
        "Sei un telecronista sportivo italiano professionista, stile Pierluigi Pardo o Fabio Caressa. "
        "La lunghezza della battuta dipende dall'importanza dell'evento: "
        "azioni di routine (passaggi normali) -> pochissime parole, secche e dinamiche; "
        "azioni importanti (tiri, corner, falli) -> una frase; "
        "momenti clou (gol, parate) -> massimo due frasi esaltate. "
        "Usa un verbo DIVERSO a ogni battuta. "
        "Pool di verbi (ruota): verticalizza, scarica, apre, lancia, conduce, cerca, "
        "avanza, tiene, tocca, smarca, gioca, distribuisce, mette in moto. "
        "Se aggiungi un complemento al verbo, usa solo riferimenti concreti al gioco: "
        "un nome di giocatore ('scarica per Haaland'), una zona ('verso l'area', 'in profondità', 'sulla fascia'). "
        "MAI termini tattici astratti ('il settore', 'il campo', 'lo spazio'). "
        "A volte (non sempre) inizia la battuta con un breve connettivo per creare flusso: "
        "'E poi', 'Ancora', 'Ma ecco che', 'Ora', 'Ed e''. Non abusarne: max una volta ogni 3-4 battute. "
        "NON citare mai il giocatore ricevente: sai solo chi ha la palla, non a chi la passa. "
        "Basati ESCLUSIVAMENTE sui dati JSON. Testo piano, niente markdown, niente virgolette. "
        "NON usare mai la parola 'pallone' o 'palla': sono sottointesi."
    ),
    "en": (
        "You are a professional English football commentator, in the style of Martin Tyler or Peter Drury. "
        "The length of each line depends on the event importance: "
        "routine actions (normal passes) -> very few words, punchy and dynamic; "
        "important actions (shots, corners, fouls) -> one sentence; "
        "key moments (goals, saves) -> up to two excited sentences. "
        "Use a DIFFERENT verb in every line. "
        "Verb pool (rotate): drives, releases, spreads, launches, carries, looks for, "
        "pushes forward, holds, touches, finds, plays, distributes, sets in motion. "
        "If you add a complement to the verb, use only concrete football references: "
        "a player name ('releases to Haaland'), a zone ('towards the box', 'down the wing'). "
        "NEVER use abstract tactical terms ('the sector', 'the space'). "
        "Sometimes (not always) start the line with a short connective for flow: "
        "'And then', 'Again', 'But here', 'Now', 'Oh and'. Don't overuse: max once every 3-4 lines. "
        "NEVER mention the receiving player: you only know who has the ball, not who receives it. "
        "Base yourself EXCLUSIVELY on the JSON data. Plain text, no markdown, no quotes. "
        "NEVER use the word 'ball' or 'football': they are implied."
    ),
}
# Backward compatibility: prompt of the current language.
LLM_SYSTEM_PROMPT: str = LLM_SYSTEM_PROMPTS.get(LANGUAGE, LLM_SYSTEM_PROMPTS["en"])
LLM_TEMPERATURE: float = 0.9

# --------------------------------------------------------------------------- #
# Phase 3 - Prosody model (THE CONTRIBUTION)                                   #
# --------------------------------------------------------------------------- #
PROSODY_TARGETS: list[str] = ["rate_factor", "pitch_semitones", "energy_gain"]
PROSODY_CLAMP: dict[str, tuple[float, float]] = {
    "rate_factor": (0.85, 1.45),  # 1.0 = normal speed
    "pitch_semitones": (-2.0, 5.0),  # 0   = normal pitch
    "energy_gain": (0.80, 1.80),  # 1.0 = normal volume
}
PROSODY_HIDDEN_DIMS: tuple[int, ...] = (32, 16)
PROSODY_EPOCHS: int = 200
PROSODY_LR: float = 1e-3
PROSODY_BATCH_SIZE: int = 16
RANDOM_SEED: int = 42

# Rule-based baseline (also the study condition + fallback if the model is missing).
RULE_BASED_PROSODY: dict[str, dict[str, float]] = {
    "goal": {"rate_factor": 1.35, "pitch_semitones": 4.0, "energy_gain": 1.6},
    "save": {"rate_factor": 1.30, "pitch_semitones": 3.0, "energy_gain": 1.5},
    "shot_on_goal": {"rate_factor": 1.30, "pitch_semitones": 3.5, "energy_gain": 1.5},
    "shot_off": {"rate_factor": 1.20, "pitch_semitones": 2.5, "energy_gain": 1.3},
    "corner": {"rate_factor": 1.10, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "free_kick": {"rate_factor": 1.10, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "turnover": {"rate_factor": 1.15, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "foul": {"rate_factor": 1.10, "pitch_semitones": 1.0, "energy_gain": 1.1},
    "carry": {"rate_factor": 1.00, "pitch_semitones": 0.0, "energy_gain": 1.0},
    "pass": {"rate_factor": 1.00, "pitch_semitones": 0.0, "energy_gain": 1.0},
    "idle": {"rate_factor": 0.92, "pitch_semitones": -0.5, "energy_gain": 0.9},
}

# --------------------------------------------------------------------------- #
# Phase 4 - Audio synthesis                                                    #
# --------------------------------------------------------------------------- #
GAP_BETWEEN_UTTERANCES_S: float = 0.15
# Synchronization: each line starts at its event's timestamp (track aligned to
# the action). If the previous block is not finished, the next one QUEUES right
# after (no overlapping voice). Two lines closer than this threshold are merged
# into a single block (connected prosody); beyond it, the block is split and the
# second one starts at its own timestamp with silence in between.
SYNC_MERGE_MAX_GAP_S: float = 1.5
TTS_ENGINE: str = "coqui"  # only supported engine: Coqui XTTS v2
COQUI_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
COQUI_LANGUAGE: str = LANGUAGE  # derived from the commentary language
# NB: the cloned voice (COQUI_SPEAKER_WAV/_EXCITED) and the synthesis parameters
# (COQUI_SPEED, _CLAMP, _CHUNK_MAX_CHARS, EMPHASIS_..., ONSET_FADE_MS) are
# defined further below, in the "Cloned voice" and "XTTS synthesis parameters"
# sections. The merge had duplicated those constants here: duplication removed.

# --- Cloned voice (zero-shot voice cloning XTTS v2) ------------------------- #
# BASE voice reference: the commentator's "working" tone (present but NOT
# shouted voice). Must be a MONO WAV, clean (stereo degrades the cloning).
# Voice of Lele Adani, a real commentator: as a base it avoids the raspiness
# that gol_saliente gave (which is a SHOUTED celebration -> XTTS clones its
# strained sound even on calm passes). Path relative to the project root.
# PER-LANGUAGE reference voices. For languages without a dedicated wav None is
# used (default XTTS voice, no voice cloning: better than a wrong accent).
COQUI_SPEAKER_WAVS: dict[str, str | None] = {
    "it": "data/raw/commentary/ref.wav",
    "en": "data/raw/commentary/ref.wav",  # uses the Italian reference (slight accent)
}
COQUI_SPEAKER_WAVS_EXCITED: dict[str, str | None] = {
    "it": "data/raw/commentary/ref.wav",  # same file (gol_saliente_mono not available)
    "en": "data/raw/commentary/ref.wav",
}
COQUI_SPEAKER_WAV: str | None = COQUI_SPEAKER_WAVS.get(LANGUAGE)
# EXCITED voice reference: used ONLY on the important events (importance
# >= EMPHASIS_IMPORTANCE_THRESHOLD), e.g. goals/saves. Here the shouted
# celebration is in the RIGHT place -> "roar" on the key moments, calm voice on the rest.
COQUI_SPEAKER_WAV_EXCITED: str | None = COQUI_SPEAKER_WAVS_EXCITED.get(LANGUAGE)

# --- ORIGINAL / neutral voice (no cloning) ---------------------------------- #
# XTTS v2 includes 58 built-in "studio" voices, callable with the `speaker=`
# parameter (NOT speaker_wav): they give a neutral narrator, without imitating a
# character, at a regular pace. Activated when COMMENTARY_SPEAKER=builtin
# (the frontend sets it for the "Original" voice): in that case Phase 4 ignores
# the speaker_wav and uses COQUI_BUILTIN_SPEAKER. Name swappable among the 58
# available voices (see tts.speakers).
USE_BUILTIN_SPEAKER: bool = _os.environ.get("COMMENTARY_SPEAKER", "") == "builtin"
COQUI_BUILTIN_SPEAKER: str = "Gilberto Mathias"

# --- XTTS synthesis parameters --------------------------------------------- #
# Sample rate of the SYNTHESIZED audio (Phase 4). XTTS natively produces 24 kHz:
# keeping it avoids downsampling to SAMPLE_RATE (22050), which cut the high
# frequencies of the speech making it sound "muffled". NB: distinct from
# SAMPLE_RATE, which stays 22050 for the audio ANALYSIS (Phase 1c), a separate thing.
COQUI_OUTPUT_SAMPLE_RATE: int = 24000
# Base speech speed (1.0 = natural). The prosody model (Phase 3) modulates it
# per event via rate_factor; COQUI_SPEED_CLAMP keeps it in range.
COQUI_SPEED: float = 1.0
COQUI_SPEED_CLAMP: tuple[float, float] = (0.85, 1.35)
# Maximum length (characters) of a block of lines merged and synthesized in a
# single pass -> connected prosody. XTTS handles ~200-250 characters well.
COQUI_CHUNK_MAX_CHARS: int = 220
# Above this importance the event is "excited": activates the excited voice (if
# present) and lets the block receive its own emphasis.
EMPHASIS_IMPORTANCE_THRESHOLD: float = 0.60
# Fade-in ramp (ms) at the head of each block: softens the "raspy" attack of XTTS.
ONSET_FADE_MS: int = 15


# =========================================================================== #
# HUD PROFILES  (multi-interface)                                            #
# =========================================================================== #
# Each profile describes ONE interface: HUD positions + roster + jersey colors +
# active side. The profile is chosen at runtime (by resolution or via --profile)
# and applied with apply_profile(), which rewrites the constants used by the phases.
HUD_PROFILES: dict[str, dict] = {
    # Brazil vs Haiti (vertical rotated clip -> normalized ~1706x886).
    "bra_hai": {
        "regions": {
            **HUD_REGIONS,
            "minimap": (0.35, 0.82, 0.65, 0.98),  # Added minimap
        },
        "roster_home": ROSTER_HOME,
        "roster_away": ROSTER_AWAY,
        "team_codes": TEAM_CODES,
        "jersey_hsv": TEAM_JERSEY_HSV,
        "active_side": HUD_ACTIVE_SIDE,
        "aspect_min": 1.85,  # wide/short frame (~1.93)
    },
    # Tottenham vs Marseille (1920x1080 horizontal, standard EA FC HUD).
    "tot_om": {
        "regions": {
            "score": (0.135, 0.055, 0.155, 0.108),  # stacked digits "2"/"1"
            "clock": (0.085, 0.110, 0.140, 0.140),  # "x:05" (best effort)
            "active_player_home": (0.045, 0.886, 0.210, 0.930),  # "23 PORRO"
            "active_player_away": (0.780, 0.886, 0.955, 0.930),  # "PAIXAO 14"
        },
        "roster_home": [  # Tottenham
            "VICARIO",
            "PORRO",
            "ROMERO",
            "VAN DE VEN",
            "UDOGIE",
            "BISSOUMA",
            "BENTANCUR",
            "MADDISON",
            "KULUSEVSKI",
            "SON",
            "RICHARLISON",
            "JOHNSON",
            "SARR",
            "SOLANKE",
        ],
        "roster_away": [  # Olympique Marseille
            "RULLI",
            "PAIXAO",
            "BALERDI",
            "BRASSIER",
            "MURILLO",
            "HOJBJERG",
            "KONDOGBIA",
            "RABIOT",
            "HARIT",
            "GREENWOOD",
            "VERETOUT",
            "MERLIN",
            "GARCIA",
        ],
        "team_codes": {"home": "TOT", "away": "OM"},
        # WARNING: both jerseys are WHITE -> possession FROM COLOR here is not
        # reliable. Indicative values on the secondary color (navy vs light
        # blue). For this interface color clustering is needed (see note).
        "jersey_hsv": {
            "home": [((105, 60, 40), (130, 255, 255))],  # navy (Tottenham)
            "away": [((90, 50, 110), (104, 255, 255))],  # light blue (Marseille)
        },
        "active_side": "home",
        "aspect_min": 0.0,  # fallback (16:9 ~1.78)
    },
    # Manchester City vs Bayern München (1920x1080, EA FC 26 HUD).
    "mci_bay": {
        "regions": {
            "score": (0.13, 0.05, 0.16, 0.12),
            "clock": (0.060, 0.090, 0.115, 0.120),
            "active_player_home": (0.040, 0.880, 0.220, 0.930),
            "active_player_away": (0.780, 0.880, 0.960, 0.930),
            "minimap": (0.38, 0.83, 0.62, 0.98),
        },
        "roster_home": [  # Manchester City
            "EDERSON",
            "WALKER",
            "DIAS",
            "AKANJI",
            "GVARDIOL",
            "RODRI",
            "DE BRUYNE",
            "BERNARDO SILVA",
            "BERNARDO",
            "SILVA",
            "FODEN",
            "HAALAND",
            "GREALISH",
            "DOKU",
            "KOVACIC",
            "NUNES",
            "LEWIS",
            "STONES",
            "SAVINHO",
            "O'REILLY",
            "SEMENYO",
            "WRIGHT",
            "McATEE",
            "CHERKI",
        ],
        "roster_away": [  # Bayern München
            "NEUER",
            "KIMMICH",
            "UPAMECANO",
            "KIM",
            "DAVIES",
            "GORETZKA",
            "MUSIALA",
            "SANE",
            "MULLER",
            "GNABRY",
            "KANE",
            "COMAN",
            "LAIMER",
            "GUERREIRO",
            "PAVLOVIC",
            "STANIŠIĆ",
            "STANISIC",
            "OLISE",
            "Tel",
            "PALHINHA",
            "RAPHAËL GUERREIRO",
        ],
        "team_codes": {"home": "MCI", "away": "BAY"},
        "jersey_hsv": {
            "home": [((90, 40, 120), (115, 255, 255))],  # light blue (Man City)
            "away": [((0, 80, 80), (10, 255, 255)), ((165, 80, 80), (180, 255, 255))],  # red
        },
        "active_side": None,  # no fixed side, uses the heuristic
        "aspect_min": 1.7,  # priority on 16:9 video (~1.77)
        # Tracking thresholds TUNED on this camera (1080p, wide framing:
        # the distances in px are larger than on the bra_hai 1706x886 videos).
        # With the FINE-TUNED detector the keeper localizes well: the real
        # shot+save has GK at 211-322px, midfield play ~900px. shot_goal_view
        # lowered from 1200 (patch for the old model) to 600: separates the two.
        "ball_tracking": {
            "shot_goal_view_px": 600.0,
            "near_player_px": 160.0,
            "near_goalkeeper_px": 300.0,
        },
        # Radar markers of THIS interface, measured on the real frames:
        # light-blue triangles = City (home), crimson circles bordered with white =
        # Bayern (away), yellow-ochre cross = ball.
        "radar_hsv": {
            "ball": [((20, 120, 120), (35, 255, 255))],
            "home": [((90, 60, 120), (115, 255, 255))],
            "away": [((165, 60, 60), (180, 255, 230)), ((0, 60, 60), (8, 255, 230))],
        },
    },
}

DEFAULT_PROFILE: str = "bra_hai"


def select_profile(frame_w: int, frame_h: int, name: str = "auto") -> tuple[str, dict]:
    """
    Active HUD profile: by explicit name, or 'auto' based on the aspect ratio.
    It compares the aspect ratio of the video to the predefined profiles to select the best match.
    """
    if name and name != "auto":
        return name, HUD_PROFILES[name]
    aspect = frame_w / max(frame_h, 1)
    cands = [
        (p.get("aspect_min", 0.0), n, p)
        for n, p in HUD_PROFILES.items()
        if aspect >= p.get("aspect_min", 0.0)
    ]
    if cands:
        _, n, p = max(cands, key=lambda c: c[0])
        return n, p
    return DEFAULT_PROFILE, HUD_PROFILES[DEFAULT_PROFILE]


def apply_profile(prof: dict) -> None:
    """Rewrite the global constants used by the phases with the profile's values."""
    global HUD_REGIONS, ROSTER_HOME, ROSTER_AWAY, TEAM_CODES, TEAM_JERSEY_HSV
    global HUD_ACTIVE_SIDE, ROSTER, BALL_TRACKING, RADAR_HSV
    HUD_REGIONS = prof["regions"]
    ROSTER_HOME = prof["roster_home"]
    ROSTER_AWAY = prof["roster_away"]
    TEAM_CODES = prof["team_codes"]
    TEAM_JERSEY_HSV = prof["jersey_hsv"]
    HUD_ACTIVE_SIDE = prof["active_side"]
    ROSTER = {**{n: "home" for n in ROSTER_HOME}, **{n: "away" for n in ROSTER_AWAY}}
    # Per-camera tracking thresholds: pristine defaults + profile overrides
    # (the distances in px depend on the resolution and zoom of the framing).
    BALL_TRACKING = {**_BALL_TRACKING_DEFAULTS, **prof.get("ball_tracking", {})}
    # Radar marker colors of this interface (default = bra_hai).
    RADAR_HSV = {**_RADAR_HSV_DEFAULTS, **prof.get("radar_hsv", {})}
