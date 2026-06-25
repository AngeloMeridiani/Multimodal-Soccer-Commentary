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
    Fase 2 (script)   -> trasforma gli eventi in testo di telecronaca (template).
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

# Mappa giocatore -> squadra. Serve a distinguere "passaggio" (stessa squadra)
# da "palla persa" (squadra avversaria). Da compilare con le rose delle 2 squadre.
# Esempio: {"RONALDO": "home", "MESSI": "away"}. Nomi in MAIUSCOLO.
ROSTER: dict[str, str] = {}

OCR_LANGUAGES: list[str] = ["en"]   # lingue per EasyOCR
OCR_MIN_CONFIDENCE: float = 0.30    # sotto questa confidenza la lettura viene scartata

# --------------------------------------------------------------------------- #
# Importanza degli eventi (0 = banale, 1 = clou). Guida la prosodia.           #
# --------------------------------------------------------------------------- #
EVENT_IMPORTANCE: dict[str, float] = {
    "goal":      1.00,
    "turnover":  0.55,   # palla persa / intercetto
    "pass":      0.25,
    "idle":      0.10,   # gioco a centrocampo, nessun evento saliente
}
EVENT_TYPES: list[str] = ["goal", "turnover", "pass", "idle"]  # ordine per one-hot

# --------------------------------------------------------------------------- #
# Fase 2 - Generazione testo (template)                                        #
# --------------------------------------------------------------------------- #
# {player} e {other} vengono sostituiti. Piu' varianti per evitare ripetizioni.
TEMPLATES: dict[str, list[str]] = {
    "goal":     ["GOOOL! Ha segnato {player}!", "Rete di {player}! Incredibile!",
                 "{player} la mette dentro! GOL!"],
    "turnover": ["Palla persa, la recupera {player}.", "Se ne impossessa {player}.",
                 "Cambio di fronte, ora {player}."],
    "pass":     ["{player} controlla.", "Apertura per {player}.",
                 "La gioca {player}."],
    "idle":     ["Si fa girare il pallone.", "Fase di studio a centrocampo."],
}

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

# Mappatura "a regole" usata come BASELINE e come fallback se manca il modello
# addestrato. E' anche una delle condizioni dello studio (regole vs appreso).
RULE_BASED_PROSODY: dict[str, dict[str, float]] = {
    "goal":     {"rate_factor": 1.35, "pitch_semitones": 4.0, "energy_gain": 1.6},
    "turnover": {"rate_factor": 1.15, "pitch_semitones": 1.5, "energy_gain": 1.2},
    "pass":     {"rate_factor": 1.00, "pitch_semitones": 0.0, "energy_gain": 1.0},
    "idle":     {"rate_factor": 0.92, "pitch_semitones": -0.5, "energy_gain": 0.9},
}

# --------------------------------------------------------------------------- #
# Fase 4 - Sintesi audio                                                       #
# --------------------------------------------------------------------------- #
SAMPLE_RATE: int = 22050
GAP_BETWEEN_UTTERANCES_S: float = 0.15   # piccola pausa tra una battuta e l'altra
# Stile vocale: etichetta descrittiva (NON una voce clonata di personaggi reali).
VOICE_STYLE: str = "dark_hero"           # documentazione/expansione futura
