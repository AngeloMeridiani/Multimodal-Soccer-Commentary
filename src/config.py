"""
config.py
=========
Configurazione centralizzata della pipeline di telecronaca per videogiochi.

Tutti i percorsi sono RELATIVI alla root del progetto. L'unico file da toccare
per cambiare comportamento (rotazione/crop video, regioni HUD, soglie,
iperparametri, template) e' questo.

Layout PIATTO: questo file e gli script "NN_*.py" stanno nella stessa cartella,
che e' anche la root del progetto. Le cartelle dati vengono create QUI dentro.

Pipeline "a staffetta":
    Fase 0  (norm.)    -> raddrizza il video (rotazione) e ritaglia le bande nere.
    Fase 1  (eventi)   -> OCR dell'HUD (punteggio + targhette) -> log eventi JSON.
    Fase 1b (visivo)   -> YOLO + OpenCV: tiri, dribbling, parate, corner.
    Fase 1c (audio)    -> eccitazione del tifo (Librosa) + trascrizione (Whisper).
    Fase 2  (script)   -> eventi -> testo di telecronaca (template).
    Fase 2b (LLM)      -> eventi -> testo di telecronaca (LLM, alternativa).
    Fase 3  (prosodia) -> ADDESTRA il modello evento->prosodia (il CONTRIBUTO).
    Fase 4  (sintesi)  -> testo + prosodia -> audio espressivo (Coqui XTTS v2).
    Fase 5  (valutaz.) -> studio A/B sugli ascoltatori + test statistico.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Percorsi (layout PIATTO: la root e' la cartella di questo file)              #
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
GAMEPLAY_DIR: Path = DATA_DIR / "raw" / "gameplay"  # video di gioco (input)
COMMENTARY_DIR: Path = DATA_DIR / "raw" / "commentary"  # clip telecronache reali
PROSODY_ANNOTATIONS: Path = DATA_DIR / "raw" / "prosody_annotations.csv"

FEATURES_DIR: Path = PROJECT_ROOT / "features"
NORMALIZED_DIR: Path = FEATURES_DIR / "normalized"  # frame/anteprime normalizzate
EVENTS_DIR: Path = FEATURES_DIR / "events"  # log eventi JSON (Fase 1)
SCRIPTS_DIR: Path = FEATURES_DIR / "scripts"  # testo telecronaca (Fase 2)
PROSODY_DIR: Path = FEATURES_DIR / "prosody"  # dataset prosodico (Fase 3)
PROSODY_DATASET: Path = PROSODY_DIR / "prosody_dataset.npz"

MODELS_DIR: Path = PROJECT_ROOT / "models"
PROSODY_MODEL_PATH: Path = MODELS_DIR / "prosody_mlp.pt"
PROSODY_SCALER_PATH: Path = MODELS_DIR / "prosody_scaler.joblib"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
AUDIO_OUT_DIR: Path = OUTPUTS_DIR / "audio"
STUDY_DIR: Path = OUTPUTS_DIR / "study"

# --------------------------------------------------------------------------- #
# Fase 0 - Normalizzazione video (rotazione + ritaglio letterbox)             #
# --------------------------------------------------------------------------- #
# I video registrati da telefono possono avere un flag di rotazione che OpenCV
# NON applica, e bande nere laterali. Qui si raddrizza e si ritaglia, una volta
# sola, prima di tutto il resto.
#
# VIDEO_ROTATION: "auto" legge la rotazione dai metadati (cv2/ffprobe);
#   altrimenti forza un valore in gradi orari: 0, 90, 180, 270.
VIDEO_ROTATION: str | int = "auto"

# LETTERBOX_CROP: "auto" rileva le bande nere sul primo frame e ritaglia;
#   None disattiva il crop; oppure una tupla normalizzata (x1, y1, x2, y2) in [0,1]
#   per fissare manualmente l'area di gioco (consigliato dopo la calibrazione).
LETTERBOX_CROP: str | None | tuple[float, float, float, float] = "auto"

# Sotto questa intensita' (0-255) un pixel e' considerato "nero" (bordo).
LETTERBOX_BLACK_THRESHOLD: int = 18
# Frazione minima di pixel non-neri perche' una riga/colonna sia "contenuto".
LETTERBOX_MIN_FILL: float = 0.05

# --------------------------------------------------------------------------- #
# Fase 1 - Estrazione eventi (OCR dell'HUD)                                    #
# --------------------------------------------------------------------------- #
# L'HUD cambia lentamente: 2 fps bastano e tengono leggero l'OCR.
FRAMES_PER_SECOND: float = 2.0

# La Fase 1b (visiva) campiona PIU' fitto dell'OCR: un tiro dura pochi decimi
# di secondo e a 2 fps "sparisce" tra un campione e l'altro (la palla percorre
# piu' di max_ball_jump_px e il moto viene scartato come glitch). A 8 fps il
# tragitto tra due campioni resta sotto la soglia anti-glitch e il tiro si vede.
VISUAL_FRAMES_PER_SECOND: float = 8.0

# Regioni dell'HUD in coordinate NORMALIZZATE [0,1] (x1, y1, x2, y2), riferite
# al frame GIA' normalizzato (raddrizzato + ritagliato). Vanno calibrate sul TUO
# video con: python 01_calibrate_hud.py --video <clip> --ocr
# (questi sono valori di partenza tipici per l'HUD di EA FC, punteggio in alto a
#  sinistra e due targhette nomi in basso ai due lati).
HUD_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # Calibrate sui fotogrammi REALI di questa clip (1920x886, crop letterbox):
    "score": (0.115, 0.025, 0.205, 0.10),  # SOLO le due cifre "1 .. 0"
    "clock": (0.27, 0.02, 0.37, 0.11),  # "16:55"
    "active_player_home": (0.035, 0.90, 0.22, 0.985),  # "11 RAPHINHA" (Brasile, sx)
    "active_player_away": (0.74, 0.88, 0.95, 0.99),  # "BELLEGARDE 10" (Haiti, dx)
}

# Lato della targhetta da usare come "giocatore in possesso" quando NON c'e' una
# fonte affidabile di possesso (la Fase 1b/visivo). Le due targhette sono SEMPRE
# entrambe a schermo (mostrano il giocatore SELEZIONATO di ogni squadra), quindi
# dalla sola HUD il possesso non e' deducibile.
#   "home" / "away" -> forza il lato (usalo quando SAI chi attacca nella clip).
#   None            -> non si forza: si segue il lato che cambia, possesso incerto.
HUD_ACTIVE_SIDE: str | None = None  # Impostato a None per abilitare il cambio di possesso dinamico

# Codici squadra mostrati nel punteggio (per disambiguare home/away dall'OCR).
TEAM_CODES: dict[str, str] = {"home": "BRA", "away": "HAI"}

# Rose SEPARATE per squadra. Servono ad agganciare (snap) i nomi letti dall'OCR
# alla rosa nota e a derivare la squadra dell'attore. Nomi in MAIUSCOLO.
# Inserisci qui le rose REALI delle due squadre della clip.
ROSTER_HOME: list[str] = [  # Brasile (home) - COMPLETA con la rosa REALE della clip
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
    "CUNHA",  # mancavano dal debug OCR
]
ROSTER_AWAY: list[str] = [  # Haiti (away) - COMPLETA con la rosa REALE della clip
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
    "MARCUS",  # mancavano dal debug OCR
]

# ROSTER combinato giocatore -> squadra ("home"/"away"), derivato dalle due rose.
# Mantenuto per retro-compatibilita' con chi usa ancora la mappa unica.
ROSTER: dict[str, str] = {
    **{n: "home" for n in ROSTER_HOME},
    **{n: "away" for n in ROSTER_AWAY},
}

OCR_LANGUAGES: list[str] = ["en"]  # lingue per EasyOCR
OCR_MIN_CONFIDENCE: float = 0.30  # sotto questa confidenza la lettura si scarta
# Soglia piu' bassa per i NOMI: conviene leggere tutto il cognome (anche a bassa
# confidenza) e poi agganciarlo alla rosa, invece di scartare pezzi e ritrovarsi
# frammenti come "NDRO"/"INHOS".
OCR_NAME_MIN_CONFIDENCE: float = 0.15

# Un punteggio di calcio oltre questo valore e' quasi certamente OCR rotto:
# i numeri letti fuori range vengono ignorati nel parsing del punteggio.
MAX_PLAUSIBLE_SCORE: int = 9
# Un cambio di punteggio deve persistere per N letture consecutive prima di
# valere come gol (debounce contro il flicker dell'OCR).
GOAL_CONFIRM_FRAMES: int = 2

# --------------------------------------------------------------------------- #
# Importanza degli eventi (0 = banale, 1 = clou). Guida la prosodia.           #
# --------------------------------------------------------------------------- #
EVENT_IMPORTANCE: dict[str, float] = {
    "goal": 1.00,
    "save": 0.80,  # parata del portiere
    "shot_on_goal": 0.85,  # tiro nello specchio
    "shot_off": 0.60,  # tiro fuori / ribattuto
    "corner": 0.45,
    "free_kick": 0.50,
    "turnover": 0.55,  # palla persa / intercetto
    "foul": 0.40,
    "dribble": 0.50,
    "pass": 0.25,
    "idle": 0.10,  # gioco a centrocampo
}

# UNICA lista usata per il one-hot, sia in training (Fase 3) sia in sintesi
# (Fase 4). Coerenza garantita: ogni tipo di evento ha la sua colonna.
EVENT_TYPES: list[str] = [
    "goal",
    "save",
    "shot_on_goal",
    "shot_off",
    "corner",
    "free_kick",
    "turnover",
    "foul",
    "dribble",
    "pass",
    "idle",
]

# --------------------------------------------------------------------------- #
# Fase 1b - Modulo Visivo (YOLO + OpenCV)                                      #
# --------------------------------------------------------------------------- #
YOLO_MODEL: str = "models/best.pt"  # modello FIFA addestrato su Roboflow
# Il modello FIFA usa nomi diversi da COCO. Li mappiamo ai nomi interni del progetto.
YOLO_CLASS_MAP: dict[str, str] = {
    "player": "person",  # giocatori
    # Il portiere resta una classe A PARTE: serve al rilevamento delle PARATE
    # (palla veloce che si ferma vicino al portiere = respinta/presa).
    "goalkeeper": "goalkeeper",
    "referee": "referee",  # arbitro
    "ball": "sports_ball",  # pallone
    # NB: la classe "goalpost" del modello e' stata provata come segnale di
    # "porta inquadrata" per il tiro e SCARTATA: falsi positivi del palo a
    # centrocampo riportavano i tiri fantasma (retest 4 clip, 4 lug 2026).
}
YOLO_CONFIDENCE: float = 0.35
# La palla e' piccola e spesso sfocata dal motion blur: con la soglia dei
# giocatori (0.35) YOLO la vede in <10% dei frame e la velocita' non si puo'
# calcolare. Soglia dedicata piu' bassa: qualche falso positivo in piu' e'
# tollerabile (il filtro max_ball_jump_px scarta i salti implausibili).
YOLO_BALL_CONFIDENCE: float = 0.15

# --- Possesso palla dal COLORE MAGLIA del giocatore piu' vicino alla palla --- #
# Range HSV (OpenCV: H 0-179, S 0-255, V 0-255), TARATI sui frame reali della clip.
# home = Brasile (giallo); away = Haiti (rosso, che in HSV sta a cavallo di 0/180).
TEAM_JERSEY_HSV: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "home": [((20, 90, 90), (38, 255, 255))],
    "away": [((0, 90, 70), (10, 255, 255)), ((168, 90, 70), (179, 255, 255))],
}
# Porzione del bounding box del giocatore da campionare (torso = alto-centrale),
# in frazioni del box: (x1, y1, x2, y2). Evita gambe/erba e numero schiena.
JERSEY_SAMPLE_BOX: tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.55)
# Frazione minima di pixel del colore squadra perche' la classificazione valga.
JERSEY_MIN_FILL: float = 0.12
# Margine: il colore vincente deve superare l'altro di almeno questo fattore.
JERSEY_MIN_MARGIN: float = 1.3
# Raggio massimo (px sul frame normalizzato) entro cui un giocatore "ha" la palla.
POSSESSION_MAX_DIST_PX: float = 90.0
# Il possesso cambia squadra solo dopo N frame coerenti (anti-flicker).
POSSESSION_CONFIRM_FRAMES: int = 2

# Soglie di tracking della palla sul frame normalizzato.
# Le velocita' sono in PIXEL AL SECONDO (px/s), NON per frame: cosi' restano
# valide qualunque sia VISUAL_FRAMES_PER_SECOND (prima erano px/campione a
# 2 fps: 120 px/campione = 240 px/s, 40 = 80).
BALL_TRACKING: dict[str, float] = {
    "min_speed_shot": 240.0,  # px/s: sopra = possibile tiro
    "min_speed_pass": 80.0,  # px/s: sopra = passaggio, sotto = possesso
    "save_drop_ratio": 0.35,  # parata: la velocita' crolla sotto questa frazione
    "near_player_px": 100.0,  # raggio (px) per "giocatore vicino alla palla"
    # Raggio palla-portiere per la PARATA. Calibrato sulla telemetria reale
    # (possesso_palla_e_parata_ball.json): al crollo di velocita' il portiere
    # era a ~184 px dalla palla.
    "near_goalkeeper_px": 200.0,
    # Un TIRO vale solo col portiere in vista entro questo raggio: distingue
    # il tiro vero (verso la porta) dal passaggio forte a centrocampo, che ha
    # velocita' simili (380-550 px/s) ma nessun portiere vicino. Calibrato:
    # tiro vero = GK a 259-374 px; falsi positivi = GK assente o >660 px.
    "shot_goal_view_px": 500.0,
    "max_ball_jump_px": 400.0,  # px TRA DUE CAMPIONI: oltre = glitch di tracking
    # Per quanti secondi il classificatore RICORDA l'ultimo stato palla quando
    # YOLO la perde: nella parata il motion blur nasconde la palla per diversi
    # frame e senza memoria il "crollo di velocita'" non viene mai visto.
    "ball_memory_s": 1.0,
    # Memoria dell'ultima POSIZIONE del portiere, usata SOLO dalla parata:
    # nel tuffo YOLO perde il GK proprio nei frame decisivi, ma la sua
    # posizione a schermo resta valida per un paio di secondi. Il TIRO invece
    # richiede il portiere visto ORA (con la memoria tornerebbero i tiri
    # fantasma filtrati su possesso_palla_1).
    "gk_memory_s": 2.0,
    # Via SEQUENZIALE alla parata (indipendente dalla camera): se entro questa
    # finestra da un TIRO rilevato la palla si ferma quasi del tutto, e' una
    # parata anche senza portiere visibile (su camera larga YOLO confonde o
    # perde i GK e le distanze in px non sono affidabili).
    "shot_to_save_max_s": 2.5,
}

# Copia PRISTINA dei default: apply_profile fonde gli override del profilo
# partendo sempre da qui, cosi' cambiare profilo non eredita i valori del
# profilo applicato prima.
_BALL_TRACKING_DEFAULTS: dict[str, float] = dict(BALL_TRACKING)

# Colori dei SEGNALINI sulla minimappa/radar (non le maglie: il radar usa
# colori stilizzati suoi). Range HSV OpenCV (H 0-179); ogni voce e' una LISTA
# di intervalli perche' il rosso sta a cavallo dello 0/180. Override
# per-profilo con la chiave "radar_hsv" (vedi HUD_PROFILES).
# Default = radar di bra_hai: palla arancione, home giallo, away rosso.
RADAR_HSV: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "ball": [((10, 100, 150), (25, 255, 255))],
    "home": [((25, 30, 150), (45, 255, 255))],
    "away": [((0, 50, 100), (10, 255, 255)), ((160, 50, 100), (180, 255, 255))],
}
_RADAR_HSV_DEFAULTS = dict(RADAR_HSV)

# Due eventi visivi dello STESSO tipo entro questa finestra sono la stessa
# azione: a 8 fps una parata soddisfa la condizione su piu' frame consecutivi
# e senza cooldown verrebbe commentata due volte.
VISUAL_EVENT_COOLDOWN_S: float = 3.0

# Zone del campo in coordinate normalizzate (per classificare la posizione palla).
FIELD_ZONES: dict[str, tuple[float, float, float, float]] = {
    "penalty_area_home": (0.00, 0.15, 0.20, 0.85),
    "penalty_area_away": (0.80, 0.15, 1.00, 0.85),
    "midfield": (0.30, 0.00, 0.70, 1.00),
    "wing_left": (0.00, 0.00, 1.00, 0.25),
    "wing_right": (0.00, 0.75, 1.00, 1.00),
}

# --------------------------------------------------------------------------- #
# Fase 1c - Modulo Uditivo (Crowd Excitement + Whisper)                        #
# --------------------------------------------------------------------------- #
SAMPLE_RATE: int = 22050  # usato da audio e sintesi
WHISPER_MODEL_SIZE: str = "base"  # tiny/base/small/medium/large
WHISPER_LANGUAGE: str = "it"  # lingua della telecronaca originale
AUDIO_ANALYSIS_WINDOW_S: float = 2.0
CROWD_EXCITEMENT_THRESHOLDS: dict[str, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,  # sopra 0.75 = "peak"
}

# --------------------------------------------------------------------------- #
# Fase 2 - Generazione testo (template)                                        #
# --------------------------------------------------------------------------- #
TEMPLATES: dict[str, list[str]] = {
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
    "dribble": [
        "{player} salta l'uomo! Che dribbling!",
        "Numero di {player}, supera il difensore!",
    ],
    "pass": ["{player} controlla.", "Apertura per {player}.", "La gioca {player}."],
    "idle": ["Si fa girare il pallone.", "Fase di studio a centrocampo."],
}

# --------------------------------------------------------------------------- #
# Fase 2b - Generazione testo con LLM                                          #
# --------------------------------------------------------------------------- #
LLM_PROVIDER: str = "ollama"  # "ollama" (locale) | "openai" | "anthropic"
LLM_CONFIG: dict[str, dict] = {
    # llama3.2:3b invece di llama3 (8B): entra comodo nei 7.6 GB di RAM
    # disponibili e resta abbastanza veloce su CPU per generare le battute.
    "ollama": {"model": "llama3.2:3b", "base_url": "http://localhost:11434"},
    "openai": {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"model": "claude-haiku-4-5-20251001", "api_key_env": "ANTHROPIC_API_KEY"},
}
LLM_SYSTEM_PROMPT: str = (
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
)
LLM_TEMPERATURE: float = 0.9

# --------------------------------------------------------------------------- #
# Fase 3 - Modello di prosodia (IL CONTRIBUTO)                                 #
# --------------------------------------------------------------------------- #
PROSODY_TARGETS: list[str] = ["rate_factor", "pitch_semitones", "energy_gain"]
PROSODY_CLAMP: dict[str, tuple[float, float]] = {
    "rate_factor": (0.85, 1.45),  # 1.0 = velocita' normale
    "pitch_semitones": (-2.0, 5.0),  # 0   = pitch normale
    "energy_gain": (0.80, 1.80),  # 1.0 = volume normale
}
PROSODY_HIDDEN_DIMS: tuple[int, ...] = (32, 16)
PROSODY_EPOCHS: int = 200
PROSODY_LR: float = 1e-3
PROSODY_BATCH_SIZE: int = 16
RANDOM_SEED: int = 42

# Baseline a regole (anche condizione dello studio + fallback se manca il modello).
RULE_BASED_PROSODY: dict[str, dict[str, float]] = {
    "goal": {"rate_factor": 1.35, "pitch_semitones": 4.0, "energy_gain": 1.6},
    "save": {"rate_factor": 1.30, "pitch_semitones": 3.0, "energy_gain": 1.5},
    "shot_on_goal": {"rate_factor": 1.30, "pitch_semitones": 3.5, "energy_gain": 1.5},
    "shot_off": {"rate_factor": 1.20, "pitch_semitones": 2.5, "energy_gain": 1.3},
    "corner": {"rate_factor": 1.10, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "free_kick": {"rate_factor": 1.10, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "turnover": {"rate_factor": 1.15, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "foul": {"rate_factor": 1.10, "pitch_semitones": 1.0, "energy_gain": 1.1},
    "dribble": {"rate_factor": 1.20, "pitch_semitones": 2.0, "energy_gain": 1.3},
    "pass": {"rate_factor": 1.00, "pitch_semitones": 0.0, "energy_gain": 1.0},
    "idle": {"rate_factor": 0.92, "pitch_semitones": -0.5, "energy_gain": 0.9},
}

# --------------------------------------------------------------------------- #
# Fase 4 - Sintesi audio                                                       #
# --------------------------------------------------------------------------- #
GAP_BETWEEN_UTTERANCES_S: float = 0.15
<<<<<<< HEAD
VOICE_STYLE: str = "dark_hero"          # etichetta descrittiva (NON voce clonata)
TTS_ENGINE: str = "coqui"               # unico motore realmente supportato (XTTS v2)
COQUI_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
=======
TTS_ENGINE: str = "coqui"  # unico motore supportato: Coqui XTTS v2
COQUI_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
# Voce da clonare: clip CONCITATA (esultanza da gol) al posto della calma
# ref.wav — prova per dare piu' enfasi a tutta la telecronaca.
COQUI_SPEAKER_WAV: str | None = "data/raw/commentary/gol_saliente.wav"
COQUI_SPEAKER_WAV_EXCITED: str | None = (
    None  # 2a clip concitata dello STESSO telecronista (opzionale)
)
>>>>>>> a1ab307357c04f9227ca7d57cca00223e3d5f462
COQUI_LANGUAGE: str = "it"
COQUI_SPEED: float = 1.1  # passo base (poi modulato da rate_factor di prosody_mlp)
COQUI_SPEED_CLAMP: tuple[float, float] = (0.9, 1.35)  # limiti di sicurezza sullo speed finale
COQUI_CHUNK_MAX_CHARS: int = 180  # battute unite in blocchi <= N char -> prosodia connessa
EMPHASIS_IMPORTANCE_THRESHOLD: float = (
    0.6  # sopra questa soglia = evento "concitato" (gol, parata...)
)
ONSET_FADE_MS: int = 80  # smorza l'attacco rauco d'avvio di XTTS
# NB: l'enfasi (velocita' + volume) arriva da prosody_mlp (Fase 3):
#     rate_factor -> speed XTTS,  energy_gain -> volume.

# --- Voce clonata (voice cloning zero-shot XTTS v2) ------------------------- #
# Riferimento vocale BASE: il tono "di lavoro" del telecronista (voce calda,
# presente ma non urlata). XTTS clona timbro e stile da questo file.
# WAV mono, pulito, ~10-20s. Percorso relativo alla root del progetto.
COQUI_SPEAKER_WAV: str | None = "data/raw/voice/mia_voce.wav"
# Riferimento vocale CONCITATO (opzionale): usato SOLO sugli eventi importanti
# (importanza >= EMPHASIS_IMPORTANCE_THRESHOLD), es. gol/parate. Se None, per
# tutti gli eventi si usa COQUI_SPEAKER_WAV. Consigliato per un effetto "boato".
COQUI_SPEAKER_WAV_EXCITED: str | None = None
# Es. per attivarlo: "data/raw/voice/mia_voce_concitata.wav"

# --- Parametri di sintesi XTTS --------------------------------------------- #
# Velocità base del parlato (1.0 = naturale). Il modello di prosodia (Fase 3)
# la modula per evento tramite rate_factor; COQUI_SPEED_CLAMP la tiene in range.
COQUI_SPEED: float = 1.0
COQUI_SPEED_CLAMP: tuple[float, float] = (0.85, 1.35)
# Lunghezza massima (caratteri) di un blocco di battute unite e sintetizzate in
# un'unica passata -> prosodia connessa. XTTS regge bene ~200-250 caratteri.
COQUI_CHUNK_MAX_CHARS: int = 220
# Sopra questa importanza l'evento e' "concitato": attiva la voce excited (se
# presente) e permette al blocco di ricevere enfasi propria.
EMPHASIS_IMPORTANCE_THRESHOLD: float = 0.70
# Rampa di fade-in (ms) in testa a ogni blocco: smorza l'attacco "rauco" di XTTS.
ONSET_FADE_MS: int = 15


# =========================================================================== #
# PROFILI HUD  (multi-interfaccia)                                            #
# =========================================================================== #
# Ogni profilo descrive UNA interfaccia: posizioni HUD + rosa + colori maglia +
# lato attivo. Il profilo si sceglie a runtime (per risoluzione o via --profile)
# e si applica con apply_profile(), che riscrive le costanti usate dalle fasi.
HUD_PROFILES: dict[str, dict] = {
    # Brasile vs Haiti (clip verticale ruotata -> normalizzata ~1706x886).
    "bra_hai": {
        "regions": {
            **HUD_REGIONS,
            "minimap": (0.35, 0.82, 0.65, 0.98),  # Aggiunto minimap
        },
        "roster_home": ROSTER_HOME,
        "roster_away": ROSTER_AWAY,
        "team_codes": TEAM_CODES,
        "jersey_hsv": TEAM_JERSEY_HSV,
        "active_side": HUD_ACTIVE_SIDE,
        "aspect_min": 1.85,  # frame largo/basso (~1.93)
    },
    # Tottenham vs Marseille (1920x1080 orizzontale, HUD standard EA FC).
    "tot_om": {
        "regions": {
            "score": (0.135, 0.055, 0.155, 0.108),  # cifre "2"/"1" impilate
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
        # ATTENZIONE: entrambe le maglie sono BIANCHE -> il possesso DAL COLORE qui
        # non e' affidabile. Valori indicativi sul colore secondario (navy vs
        # azzurro). Per questa interfaccia serve il clustering colori (vedi nota).
        "jersey_hsv": {
            "home": [((105, 60, 40), (130, 255, 255))],  # navy (Tottenham)
            "away": [((90, 50, 110), (104, 255, 255))],  # azzurro (Marseille)
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
            "home": [((90, 40, 120), (115, 255, 255))],  # azzurro (Man City)
            "away": [((0, 80, 80), (10, 255, 255)), ((165, 80, 80), (180, 255, 255))],  # rosso
        },
        "active_side": None,  # nessun lato fisso, usa euristica
        "aspect_min": 1.7,  # priorita' su video 16:9 (~1.77)
        # Soglie di tracking TARATE su questa camera (1080p, inquadratura larga:
        # le distanze in px sono maggiori che sui video bra_hai 1706x886).
        # Telemetria: tiri veri con GK a 937/1118 px (vs 259-374 su bra_hai);
        # a t=13.4 il tiratore era oltre i 100 px del raggio di default.
        "ball_tracking": {
            "shot_goal_view_px": 1200.0,
            "near_player_px": 160.0,
            "near_goalkeeper_px": 300.0,
        },
        # Segnalini del radar di QUESTA interfaccia, misurati sui frame reali:
        # triangoli azzurri = City (home), cerchi cremisi bordati di bianco =
        # Bayern (away), croce giallo-ocra = palla.
        "radar_hsv": {
            "ball": [((20, 120, 120), (35, 255, 255))],
            "home": [((90, 60, 120), (115, 255, 255))],
            "away": [((165, 60, 60), (180, 255, 230)), ((0, 60, 60), (8, 255, 230))],
        },
    },
}

DEFAULT_PROFILE: str = "bra_hai"


def select_profile(frame_w: int, frame_h: int, name: str = "auto") -> tuple[str, dict]:
    """Profilo HUD attivo: per nome esplicito, o 'auto' in base alle proporzioni."""
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
    """Riscrive le costanti globali usate dalle fasi con i valori del profilo."""
    global HUD_REGIONS, ROSTER_HOME, ROSTER_AWAY, TEAM_CODES, TEAM_JERSEY_HSV
    global HUD_ACTIVE_SIDE, ROSTER, BALL_TRACKING, RADAR_HSV
    HUD_REGIONS = prof["regions"]
    ROSTER_HOME = prof["roster_home"]
    ROSTER_AWAY = prof["roster_away"]
    TEAM_CODES = prof["team_codes"]
    TEAM_JERSEY_HSV = prof["jersey_hsv"]
    HUD_ACTIVE_SIDE = prof["active_side"]
    ROSTER = {**{n: "home" for n in ROSTER_HOME}, **{n: "away" for n in ROSTER_AWAY}}
    # Soglie di tracking per-camera: default pristini + override del profilo
    # (le distanze in px dipendono da risoluzione e zoom dell'inquadratura).
    BALL_TRACKING = {**_BALL_TRACKING_DEFAULTS, **prof.get("ball_tracking", {})}
    # Colori dei segnalini radar di questa interfaccia (default = bra_hai).
    RADAR_HSV = {**_RADAR_HSV_DEFAULTS, **prof.get("radar_hsv", {})}
