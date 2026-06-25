# Telecronaca Espressiva per Videogiochi con Prosodia Event-Aware

Sistema che genera una **telecronaca audio** a partire dal video di un
videogioco di calcio, e — questo è il contributo di ricerca — **modula la voce
in base all'importanza degli eventi di gioco**: si accende sul gol, resta calma a
centrocampo. La mappatura evento → espressività vocale **non è scritta a mano: è
appresa** da telecronache reali.

---

## 1. Il problema e il contributo

I sistemi di sintesi vocale (TTS) leggono il testo con un **tono piatto e
costante**. Una telecronaca vera è l'opposto: la sua forza è la *prosodia* — il
ritmo che accelera, il tono che sale, il volume che esplode nei momenti chiave.

**Domanda di ricerca:** un controllo prosodico *appreso e condizionato
dall'evento* rende la telecronaca sintetica più **naturale e coinvolgente** di un
TTS a tono fisso?

**Contributo:** un modello che impara, da telecronache reali, la relazione tra
*importanza dell'evento* e *parametri prosodici* (velocità, intonazione, energia),
applicata poi a una sintesi vocale. Lo validiamo con uno **studio sugli
ascoltatori** (Mean Opinion Score).

> Nota etica/legale: lo stile vocale è una *persona* generica controllabile
> (es. "voce drammatica da eroe"), **non** la clonazione della voce di personaggi
> o attori reali — pratica problematica per copyright e diritti della persona.

---

## 2. Come funziona: la pipeline "a staffetta"

Cinque fasi sequenziali; ognuna legge l'output della precedente e salva il proprio.

```
  video gameplay (.mp4)
        │
        ▼
┌──────────────────────┐
│ FASE 1 · EVENTI      │  OCR dell'HUD (punteggio + giocatore con palla)
│  (Vision/OCR)        │  ──► log eventi strutturato (JSON)
└──────────────────────┘     {t, type, player, importance}
        │
        ▼
┌──────────────────────┐
│ FASE 2 · SCRIPT      │  template per evento ──► testo della telecronaca
│  (NLP)               │     {t, text, event_type, importance}
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ FASE 3 · PROSODIA    │  ★ CONTRIBUTO ★  impara  evento ──► [rate,pitch,energy]
│  (Deep Learning)     │  da telecronache reali annotate ──► models/
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ FASE 4 · SINTESI     │  TTS neutro + applica prosodia (DSP) ──► audio .wav
│  (Audio/Signal)      │  modalità: flat | rule | learned
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ FASE 5 · VALUTAZIONE │  studio A/B sugli ascoltatori ──► flat vs learned
│  (Esperimento)       │  (Mean Opinion Score + test statistico)
└──────────────────────┘
```

Le tre modalità della Fase 4 sono anche le **condizioni dell'esperimento**:
`flat` (baseline piatta), `rule` (prosodia a regole), `learned` (modello appreso).

---

## 3. Dov'è il deep learning (e dov'è l'ingegneria)

Punto importante per inquadrare il progetto come *research-oriented*: non tutto è
deep learning, e va bene così. L'impalcatura usa strumenti pronti; il
**contributo addestrato** è concentrato in un punto.

| Fase | Natura | Note |
|------|--------|------|
| 1 · Eventi | Ingegneria (OCR) | Legge l'HUD, nessun training |
| 2 · Script | Ingegneria (template) | Affidabile; LLM = lavoro futuro |
| 3 · Prosodia | **Deep learning (nostro)** | Modello appreso evento→prosodia |
| 4 · Sintesi | Tool + DSP | TTS pronto + applicazione prosodia |
| 5 · Valutazione | Esperimento | Studio sugli ascoltatori |

---

## 4. Struttura della repository

```
fifa-commentary/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py              # percorsi, regioni HUD, template, iperparametri
│   ├── utils.py               # logging, I/O, frame, applicazione prosodia (DSP)
│   ├── 01_extract_events.py   # Fase 1 - OCR HUD → eventi
│   ├── 02_generate_script.py  # Fase 2 - eventi → testo
│   ├── 03_train_prosody.py    # Fase 3 - addestra il modello prosodia ★
│   ├── 04_synthesize.py       # Fase 4 - testo+prosodia → audio
│   └── 05_evaluate.py         # Fase 5 - studio ascoltatori
├── data/raw/
│   ├── gameplay/              # video di gioco (input)
│   ├── commentary/            # clip di telecronache reali (training prosodia)
│   └── prosody_annotations.csv# etichette segmenti (clip,start,end,event_type)
├── features/
│   ├── events/                # log eventi JSON (Fase 1)
│   ├── scripts/               # testi telecronaca (Fase 2)
│   └── prosody/               # dataset prosodico estratto (Fase 3)
├── models/                    # modello prosodia + scaler (Fase 3)
└── outputs/
    ├── audio/                 # telecronache sintetizzate (Fase 4)
    └── study/                 # materiali e risultati dello studio (Fase 5)
```

Tutti i percorsi sono **relativi** e centralizzati in `config.py`.

---

## 5. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -r requirements.txt

# Dipendenze di sistema:
#   Ubuntu: sudo apt-get install ffmpeg espeak
#   macOS:  brew install ffmpeg
```

**GPU consigliata** per EasyOCR e per il training: Google Colab va benissimo.

---

## 6. Esecuzione (ordine)

```bash
# FASE 1 - eventi dall'HUD di un video di gameplay
python src/01_extract_events.py --video data/raw/gameplay/match1.mp4

# FASE 2 - testo della telecronaca
python src/02_generate_script.py --events features/events/match1.json

# FASE 3 - addestra il modello di prosodia (due passi)
python src/03_train_prosody.py build-dataset    # estrae i target dalle clip reali
python src/03_train_prosody.py train            # addestra l'MLP

# FASE 4 - sintetizza le tre versioni (per lo studio)
python src/04_synthesize.py --script features/scripts/match1.json --mode flat
python src/04_synthesize.py --script features/scripts/match1.json --mode learned

# FASE 5 - prepara lo studio, raccogli i voti, analizza
python src/05_evaluate.py prepare --names match1
#  (distribuisci i file _A/_B.wav e responses_template.csv; raccogli i voti)
python src/05_evaluate.py analyze --responses outputs/study/responses.csv
```

> Suggerimento per il "vertical slice": esegui prima **flat** end-to-end (Fasi
> 1→2→4) per avere una telecronaca funzionante. Solo dopo aggiungi la Fase 3 e la
> versione **learned**. Così hai sempre qualcosa di consegnabile.

---

## 7. Calibrazione dell'HUD (Fase 1)

Le regioni in `config.HUD_REGIONS` sono in coordinate normalizzate [0,1] e vanno
adattate al **tuo** video: estrai un frame, individua dove stanno punteggio e
nome del giocatore, e aggiorna i quattro valori (x1,y1,x2,y2). Compila anche
`config.ROSTER` (giocatore → squadra) per distinguere *passaggio* da *palla persa*.

---

## 8. Limiti noti

- **Comprensione del gioco limitata all'HUD**: rileviamo gol, possesso e cambi di
  giocatore, *non* azioni come tiri o parate (richiederebbero riconoscimento
  visivo dell'azione — vedi "lavoro futuro").
- **Dati prosodici**: il modello è forte quanto le telecronache reali da cui
  impara; servono segmenti annotati a sufficienza e con criterio coerente.
- **TTS di default basilare**: `pyttsx3` è offline e semplice; per uno stile
  vocale espressivo si può sostituire con Coqui XTTS senza cambiare il resto.

## 9. Lavoro futuro

Modello di **salienza visiva** che stima l'importanza di un'azione dai frame
(arricchendo l'HUD); **generazione del testo con LLM** in stile telecronista;
**allineamento temporale** preciso dell'audio sul video; estensione ad altri
sport/giochi.
