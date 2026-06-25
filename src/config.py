"""
config.py
=========
Configurazione centralizzata della pipeline di telecronaca per videogiochi.

Tutti i percorsi sono RELATIVI alla root del progetto. L'unico file da toccare
per cambiare comportamento (regioni HUD, soglie, iperparametri, template) e'
questo.

Pipeline "a staffetta":
    Fase 1 (eventi)   -> legge un video di gameplay, fa OCR dell'HUD, scrive un
                         log eventi strutturato (JSON).
    Fase 1b (visivo)  -> YOLO object detection + OpenCV event detection.
    Fase 1c (audio)   -> Analisi tifo (crowd excitement) + trascrizione Whisper.
    Fase 2 (script)   -> trasforma gli eventi in testo di telecronaca (template).
    Fase 2b (LLM)     -> genera telecronaca con un LLM (alternativa ai template).
    Fase 3 (prosodia) -> ADDESTRA il modello evento->prosodia (il CONTRIBUTO).
    Fase 4 (sintesi)  -> testo + prosodia predetta -> audio espressivo (TTS+DSP).
    Fase 5 (valutaz.) -> prepara lo studio A/B sugli ascoltatori e aggrega i voti.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Percorsi                                                                     #
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
GAMEPLAY_DIR: Path = DATA_DIR / "raw" / "gameplay"        # video di gioco (input Fase 1)
COMMENTARY_DIR: Path = DATA_DIR / "raw" / "commentary"    # clip telecronache reali (training prosodia)
PROSODY_ANNOTATIONS: Path = DATA_DIR / "raw" / "prosody_annotations.csv"  # etichette per il training

FEATURES_DIR: Path = PROJECT_ROOT / "features"
EVENTS_DIR: Path = FEATURES_DIR / "events"               # log eventi JSON (Fase 1)
SCRIPTS_DIR: Path = FEATURES_DIR / "scripts"             # testo telecronaca (Fase 2)
PROSODY_DIR: Path = FEATURES_DIR / "prosody"             # dataset prosodico estratto (Fase 3)
PROSODY_DATASET: Path = PROSODY_DIR / "prosody_dataset.npz"

MODELS_DIR: Path = PROJECT_ROOT / "models"
PROSODY_MODEL_PATH: Path = MODELS_DIR / "prosody_mlp.pt"
PROSODY_SCALER_PATH: Path = MODELS_DIR / "prosody_scaler.joblib"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
AUDIO_OUT_DIR: Path = OUTPUTS_DIR / "audio"              # telecronache sintetizzate
STUDY_DIR: Path = OUTPUTS_DIR / "study"                  # materiali e risultati dello studio

# --------------------------------------------------------------------------- #
# Fase 1 - Estrazione eventi (OCR dell'HUD)                                    #
# --------------------------------------------------------------------------- #
# Quanti frame analizzare al secondo. L'HUD cambia lentamente: 2 fps bastano.
FRAMES_PER_SECOND: float = 2.0

# Regioni dell'HUD da cui leggere il testo, in coordinate NORMALIZZATE [0,1]
# (x1, y1, x2, y2). Vanno calibrate sul TUO video con lo script di calibrazione
# (vedi README). Questi sono valori di partenza tipici per un HUD in alto.
HUD_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "score":         (0.42, 0.04, 0.58, 0.11),   # punteggio centrale in alto
    "active_player": (0.05, 0.85, 0.45, 0.95),   # nome giocatore con la palla (in basso)
}

# Mappa giocatore -> squadra. Serve a distinguire "passaggio" (stessa squadra)
# da "palla persa" (squadra avversaria). Da compilare con le rose delle 2 squadre.
# Esempio di roster: nomi in MAIUSCOLO.
ROSTER: dict[str, str] = {
    # --- Squadra di casa (home) ---
    "DONNARUMMA": "home", "BASTONI": "home", "BARELLA": "home",
    "CHIESA": "home", "RETEGUI": "home", "TONALI": "home",
    "DIMARCO": "home", "CALAFIORI": "home", "FRATTESI": "home",
    "JORGINHO": "home", "RASPADORI": "home",
    # --- Squadra ospite (away) ---
    "MBAPPE": "away", "GRIEZMANN": "away", "DEMBELE": "away",
    "TCHOUAMENI": "away", "CAMAVINGA": "away", "RABIOT": "away",
    "HERNANDEZ": "away", "KOUNDE": "away", "SALIBA": "away",
    "UPAMECANO": "away", "MAIGNAN": "away",
}

OCR_LANGUAGES: list[str] = ["en"]   # lingue per EasyOCR
OCR_MIN_CONFIDENCE: float = 0.30    # sotto questa confidenza la lettura viene scartata

# --------------------------------------------------------------------------- #
# Importanza degli eventi (0 = banale, 1 = clou). Guida la prosodia.           #
# --------------------------------------------------------------------------- #
EVENT_IMPORTANCE: dict[str, float] = {
    "goal":          1.00,
    "shot_on_goal":  0.85,   # tiro in porta (nello specchio)
    "shot_off":      0.60,   # tiro fuori / parato
    "corner":        0.45,
    "free_kick":     0.50,
    "turnover":      0.55,   # palla persa / intercetto
    "foul":          0.40,
    "pass":          0.25,
    "dribble":       0.50,
    "save":          0.75,   # parata del portiere
    "idle":          0.10,   # gioco a centrocampo, nessun evento saliente
}
# Tipi base (retrocompatibili con la pipeline originale)
EVENT_TYPES: list[str] = ["goal", "turnover", "pass", "idle"]  # ordine per one-hot
# Tipi estesi (inclusi i nuovi eventi del Modulo Visivo)
EVENT_TYPES_EXTENDED: list[str] = [
    "goal", "shot_on_goal", "shot_off", "corner", "free_kick",
    "turnover", "foul", "pass", "dribble", "save", "idle",
]

# --------------------------------------------------------------------------- #
# Fase 1b - Modulo Visivo (YOLO + OpenCV)                                      #
# --------------------------------------------------------------------------- #
# Modello YOLO da usare. yolov8n e' il piu' leggero; yolov8s/m/l per accuracy.
YOLO_MODEL: str = "yolov8n.pt"

# Classi COCO rilevanti per il contesto FIFA
YOLO_CLASSES_OF_INTEREST: dict[int, str] = {
    0:  "person",       # giocatori + arbitro
    32: "sports_ball",  # pallone
}

# Confidenza minima per le detection YOLO
YOLO_CONFIDENCE: float = 0.35

# Parametri per il tracking della palla
BALL_TRACKING: dict[str, float] = {
    "min_speed_shot":    120.0,   # pixel/frame: sopra questa = possibile tiro
    "min_speed_pass":    40.0,    # pixel/frame: sopra = passaggio, sotto = possesso
    "goal_zone_y_ratio": 0.15,   # zona porta (top/bottom 15% del frame)
    "penalty_area_x":    0.20,   # area di rigore (20% laterale del frame)
}

# Zone del campo (in coordinate normalizzate per classificare la posizione palla)
FIELD_ZONES: dict[str, tuple[float, float, float, float]] = {
    "penalty_area_home": (0.00, 0.15, 0.20, 0.85),
    "penalty_area_away": (0.80, 0.15, 1.00, 0.85),
    "midfield":          (0.30, 0.00, 0.70, 1.00),
    "wing_left":         (0.00, 0.00, 1.00, 0.25),
    "wing_right":        (0.00, 0.75, 1.00, 1.00),
}

# --------------------------------------------------------------------------- #
# Fase 1c - Modulo Uditivo (Crowd Excitement + Whisper)                        #
# --------------------------------------------------------------------------- #
# Whisper: dimensione del modello ("tiny", "base", "small", "medium", "large")
WHISPER_MODEL_SIZE: str = "base"
WHISPER_LANGUAGE: str = "it"  # lingua della telecronaca originale

# Analisi crowd excitement: finestra di analisi e soglie
AUDIO_ANALYSIS_WINDOW_S: float = 2.0   # secondi di audio da analizzare per evento
CROWD_EXCITEMENT_THRESHOLDS: dict[str, float] = {
    "low":    0.25,   # RMS normalizzato sotto questa soglia
    "medium": 0.50,
    "high":   0.75,
    # sopra 0.75 = "peak"
}

# --------------------------------------------------------------------------- #
# Fase 2 - Generazione testo (template)                                        #
# --------------------------------------------------------------------------- #
# {player} e {other} vengono sostituiti. Piu' varianti per evitare ripetizioni.
TEMPLATES: dict[str, list[str]] = {
    "goal":         ["GOOOL! Ha segnato {player}!", "Rete di {player}! Incredibile!",
                     "{player} la mette dentro! GOL!"],
    "shot_on_goal": ["Che tiro di {player}! Nello specchio!",
                     "{player} calcia! Tiro potente verso la porta!"],
    "shot_off":     ["{player} calcia... fuori di poco!", "Tiro di {player}, palla a lato!"],
    "corner":       ["Calcio d'angolo. Si prepara {player}.", "Corner, batte {player}."],
    "free_kick":    ["Punizione per {player}. Posizione interessante.",
                     "Calcio di punizione, si incarica {player}."],
    "turnover":     ["Palla persa, la recupera {player}.", "Se ne impossessa {player}.",
                     "Cambio di fronte, ora {player}."],
    "foul":         ["Fallo su {player}! Intervento duro.", "{player} viene atterrato."],
    "pass":         ["{player} controlla.", "Apertura per {player}.",
                     "La gioca {player}."],
    "dribble":      ["{player} salta l'uomo! Che dribbling!",
                     "Numero di {player}, supera il difensore!"],
    "save":         ["Parata! Il portiere respinge!", "Che intervento del portiere!"],
    "idle":         ["Si fa girare il pallone.", "Fase di studio a centrocampo."],
}

# --------------------------------------------------------------------------- #
# Fase 2b - Generazione testo con LLM                                         #
# --------------------------------------------------------------------------- #
# Provider LLM: "ollama" (locale, gratuito), "openai", "anthropic"
LLM_PROVIDER: str = "ollama"

# Configurazione per ogni provider
LLM_CONFIG: dict[str, dict] = {
    "ollama": {
        "model": "llama3",
        "base_url": "http://localhost:11434",
    },
    "openai": {
        "model": "gpt-4",
        "api_key_env": "OPENAI_API_KEY",  # nome della variabile d'ambiente
    },
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}

# Prompt di sistema per l'LLM telecronista
LLM_SYSTEM_PROMPT: str = (
    "Sei un telecronista sportivo italiano, appassionato e coinvolgente. "
    "Il tuo stile e' vivace, colorito, esaltato nei momenti clou e calmo "
    "nelle fasi di gioco tranquille. Commenta gli eventi di una partita di "
    "calcio basandoti esclusivamente sui dati che ti vengono forniti in JSON. "
    "Genera una singola battuta di telecronaca (1-2 frasi) per ogni evento. "
    "Non inventare fatti non presenti nei dati. Usa esclamazioni, enfasi e "
    "il ritmo tipico della telecronaca italiana."
)

# Temperatura per la generazione LLM (0.0 = deterministico, 1.0 = creativo)
LLM_TEMPERATURE: float = 0.8

# --------------------------------------------------------------------------- #
# Fase 3 - Modello di prosodia (IL CONTRIBUTO)                                 #
# --------------------------------------------------------------------------- #
# Target prosodici predetti dal modello (moltiplicatori/spostamenti rispetto al
# parlato neutro): velocita', spostamento di pitch (semitoni), guadagno energia.
PROSODY_TARGETS: list[str] = ["rate_factor", "pitch_semitones", "energy_gain"]

# Range di sicurezza entro cui clampare le predizioni (evita audio innaturale).
PROSODY_CLAMP: dict[str, tuple[float, float]] = {
    "rate_factor":     (0.85, 1.45),   # 1.0 = velocita' normale
    "pitch_semitones": (-2.0, 5.0),    # 0  = pitch normale
    "energy_gain":     (0.80, 1.80),   # 1.0 = volume normale
}

PROSODY_HIDDEN_DIMS: tuple[int, ...] = (32, 16)
PROSODY_EPOCHS: int = 200
PROSODY_LR: float = 1e-3
PROSODY_BATCH_SIZE: int = 16
RANDOM_SEED: int = 42

# Feature estese per il modello arricchito (Livello 3):
# aggiunge crowd_excitement e ball_zone alle feature base
PROSODY_USE_EXTENDED_FEATURES: bool = False   # True dopo aver raccolto dati arricchiti

# Mappatura "a regole" usata come BASELINE e come fallback se manca il modello
# addestrato. E' anche una delle condizioni dello studio (regole vs appreso).
RULE_BASED_PROSODY: dict[str, dict[str, float]] = {
    "goal":          {"rate_factor": 1.35, "pitch_semitones": 4.0, "energy_gain": 1.6},
    "shot_on_goal":  {"rate_factor": 1.30, "pitch_semitones": 3.5, "energy_gain": 1.5},
    "shot_off":      {"rate_factor": 1.20, "pitch_semitones": 2.5, "energy_gain": 1.3},
    "corner":        {"rate_factor": 1.10, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "free_kick":     {"rate_factor": 1.10, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "turnover":      {"rate_factor": 1.15, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "foul":          {"rate_factor": 1.10, "pitch_semitones": 1.0, "energy_gain": 1.1},
    "pass":          {"rate_factor": 1.00, "pitch_semitones": 0.0, "energy_gain": 1.0},
    "dribble":       {"rate_factor": 1.20, "pitch_semitones": 2.0, "energy_gain": 1.3},
    "save":          {"rate_factor": 1.30, "pitch_semitones": 3.0, "energy_gain": 1.5},
    "idle":          {"rate_factor": 0.92, "pitch_semitones": -0.5, "energy_gain": 0.9},
}

# --------------------------------------------------------------------------- #
# Fase 4 - Sintesi audio                                                       #
# --------------------------------------------------------------------------- #
SAMPLE_RATE: int = 22050
GAP_BETWEEN_UTTERANCES_S: float = 0.15   # piccola pausa tra una battuta e l'altra
# Stile vocale: etichetta descrittiva (NON una voce clonata di personaggi reali).
VOICE_STYLE: str = "dark_hero"           # documentazione/expansione futura

# Motore TTS: "pyttsx3" (offline, semplice), "coqui" (espressivo, pesante)
TTS_ENGINE: str = "pyttsx3"

# Configurazione Coqui TTS (se TTS_ENGINE == "coqui")
COQUI_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
COQUI_SPEAKER_WAV: str | None = None  # path a un .wav di riferimento per la voce
COQUI_LANGUAGE: str = "it"
