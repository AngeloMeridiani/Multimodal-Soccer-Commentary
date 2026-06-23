# Riconoscimento Multimodale del Sarcasmo su MUStARD

Sistema di **sarcasm detection** che combina tre canali — **testo, audio e
video** — per classificare una battuta come *sarcastica* (1) o *normale* (0).
Il progetto è costruito su **Transfer Learning** (riuso di modelli pre-addestrati
per estrarre le feature) ed **Early Fusion** (concatenazione delle feature prima
del classificatore). Non viene addestrata alcuna rete end-to-end: questo lo rende
leggero, riproducibile e adatto a un prototipo di ricerca.

---

## 1. Il problema

Il sarcasmo è una forma di ironia in cui ciò che si dice è l'opposto di ciò che si
intende. È difficile da rilevare per una macchina perché **il testo da solo spesso
non basta**: la stessa frase ("Che bella giornata") può essere sincera o
sarcastica a seconda del *tono di voce* e dell'*espressione facciale*. È un
problema intrinsecamente **multimodale**.

Usiamo **MUStARD** (*Multimodal Sarcasm Detection Dataset*), un corpus di ~690
clip estratte da sitcom televisive (*Friends*, *The Big Bang Theory*, ecc.),
ciascuna etichettata manualmente come sarcastica o no, e bilanciata tra le due
classi. Ogni clip fornisce contemporaneamente la **trascrizione**, l'**audio** e
il **video** della battuta — i tre canali che il nostro sistema sfrutta.

**Ipotesi di lavoro:** unire i tre canali (cosa viene detto + come viene detto +
come appare chi parla) produce un classificatore più accurato rispetto a usare una
sola modalità. La pipeline è progettata proprio per verificare questa ipotesi.

---

## 2. Come funziona: l'idea in breve

Invece di costruire un'unica rete gigante che ingerisce testo, audio e video
insieme, scomponiamo il problema in **due passi**:

1. **Estrazione delle feature** (Transfer Learning). Ogni modalità viene data in
   pasto a un modello già addestrato su grandi quantità di dati (RoBERTa per il
   testo, ResNet50 per le immagini) o a un estrattore di feature classico
   (MFCC per l'audio). Da ogni clip otteniamo così **tre vettori numerici** che
   ne riassumono il contenuto testuale, acustico e visivo.

2. **Fusione e classificazione** (Early Fusion). I tre vettori vengono
   **concatenati** in un unico vettore "multimodale" e dati a un classificatore
   leggero (un MLP) che impara a separare sarcastico da normale.

Il vantaggio: i modelli pesanti (RoBERTa, ResNet) li usiamo solo in *inferenza*,
una volta, per generare le feature; l'unica cosa che addestriamo davvero è il
piccolo classificatore finale. Veloce da iterare, facile da riprodurre.

---

## 3. Architettura "a staffetta" (pipeline sequenziale)

Il codice è organizzato in **quattro fasi che si passano il testimone**: ogni fase
legge l'output della precedente, lo elabora e salva il proprio risultato su disco.
Non ci sono reti che girano insieme: si eseguono **in ordine**, una alla volta.

```
  sarcasm_data.json
  + videos/*.mp4
        │
        ▼
┌─────────────────┐
│  FASE 1 · TESTO │  RoBERTa  ──►  text_features.npy  +  manifest.csv ◄── "fonte di verità"
└─────────────────┘                                          │  (elenco ID + label)
        │                                                     │
        ▼                                                     │
┌─────────────────┐                                           │
│  FASE 2 · AUDIO │  MFCC     ──►  audio_features.npy          │ ogni fase
└─────────────────┘                                           │ itera sugli
        │                                                     │ stessi ID
        ▼                                                     │ del manifest
┌─────────────────┐                                           │
│ FASE 3 · VISION │  ResNet50 ──►  vision_features.npy ◄───────┘
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ FASE 4 · FUSIONE│  concatena i 3 vettori ──► MLP ──► models/  (modello + scaler)
└─────────────────┘
```

### Il "contratto" tra le fasi

Il punto critico di una pipeline sequenziale è **non disallineare le modalità**.
La soluzione adottata: la **Fase 1 è l'unica fonte di verità**. Scrive
`manifest.csv` (l'elenco ordinato degli `id` con la rispettiva `label`), e tutte
le fasi successive iterano *esattamente* su quegli ID. Così i tre vettori di una
stessa clip restano sempre accoppiati correttamente in fusione.

Inoltre ogni fase salva un **dizionario `{utterance_id: vettore}`** (non un array
"impilato"). Questo permette di gestire i buchi: se la Fase 3 non trova un volto in
una clip, quel vettore semplicemente manca dal dizionario, e la Fase 4 lo rimpiazza
con zeri **senza scartare l'intero campione multimodale**.

| Fase | Script | Legge | Scrive | Dim. vettore |
|------|--------|-------|--------|--------------|
| 1 | `01_extract_text.py` | `sarcasm_data.json` | `text_features.npy` + `manifest.csv` | 768 |
| 2 | `02_extract_audio.py` | `manifest.csv`, `<id>.mp4` | `audio_features.npy` | 240 |
| 3 | `03_extract_vision.py` | `manifest.csv`, `<id>.mp4` | `vision_features.npy` | 2048 |
| 4 | `04_train_fusion.py` | tutti i `.npy` + `manifest.csv` | `models/` | **3056** (fuso) |

---

## 4. Cosa fa ogni fase (in dettaglio)

**Fase 1 — Testo (`roberta-base`).** Legge le trascrizioni dal JSON. Ogni frase
viene tokenizzata e passata attraverso RoBERTa; dall'ultimo strato nascosto si
ricava un singolo vettore di frase tramite *mean-pooling mascherato* (media dei
token reali, ignorando il padding). Risultato: un vettore da **768** dimensioni per
clip. Il contesto conversazionale che precede la battuta è attivabile via
`USE_CONTEXT` in `config.py` — spesso migliora i risultati perché il sarcasmo si
capisce dal botta e risposta.

**Fase 2 — Audio (MFCC con `librosa`).** Estrae la traccia audio dal video e
calcola i **MFCC** (coefficienti cepstrali, lo standard per descrivere il timbro e
la prosodia della voce), insieme alle loro derivate prima e seconda (delta e
delta-delta, che catturano *come* il suono cambia nel tempo). Poiché ogni clip ha
durata diversa, si riassume la sequenza temporale con un **pooling statistico**
(media + deviazione standard), ottenendo un vettore a lunghezza fissa di **240**
dimensioni.

**Fase 3 — Vision (MediaPipe + `ResNet50`).** È la fase più delicata. Campiona
1–2 fotogrammi al secondo dal video, individua il **volto** principale con
MediaPipe, lo ritaglia e ne calcola un embedding con ResNet50 pre-addestrata
(privata del suo classificatore finale → vettore grezzo da **2048** dimensioni).
Gli embedding dei vari frame vengono mediati in un unico vettore per clip. La
gestione errori è robusta: frame senza volto vengono saltati; se in tutta la clip
non c'è alcun volto si ripiega sul fotogramma intero; se il video è corrotto la
clip viene segnalata e saltata, **senza fermare l'esecuzione**.

**Fase 4 — Fusione e classificazione.** Allinea i tre dizionari sul manifest,
riempie con zeri le modalità eventualmente mancanti, e **concatena** i tre vettori
(768 + 240 + 2048 = **3056**). Applica uno `StandardScaler` (fittato *solo* sul
train, per evitare data leakage) perché le tre modalità vivono su scale molto
diverse. Addestra infine un **MLP in PyTorch** con early stopping, e stampa
`classification_report` e matrice di confusione. Per confronto allena anche una
**Logistic Regression** (scikit-learn) come baseline rapida. Modello e scaler
vengono salvati in `models/` per future inferenze.

---

## 5. Struttura della repository

```
mustard-sarcasm-poc/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                          # tutto il codice Python
│   ├── config.py                 # percorsi relativi + iperparametri (UNICO posto da toccare)
│   ├── utils.py                  # logging, I/O feature, helper video
│   ├── 01_extract_text.py        # Fase 1
│   ├── 02_extract_audio.py       # Fase 2
│   ├── 03_extract_vision.py      # Fase 3
│   └── 04_train_fusion.py        # Fase 4
│
├── data/                         # i dati grezzi (NON versionati)
│   ├── raw/
│   │   ├── sarcasm_data.json      # annotazioni MUStARD
│   │   └── videos/                # clip <id>.mp4
│   └── processed/
│
├── features/                     # output intermedi delle fasi (rigenerabili)
│   ├── manifest.csv               # [Fase 1] id,label — il "contratto"
│   ├── text/text_features.npy     # [Fase 1]
│   ├── audio/audio_features.npy   # [Fase 2]
│   └── vision/vision_features.npy # [Fase 3]
│
└── models/                       # [Fase 4] fusion_mlp.pt + scaler.joblib
```

Tutti i percorsi sono **relativi** e centralizzati in `config.py`: per cambiare
qualsiasi cosa (nome modello, numero di frame, iperparametri dell'MLP) si modifica
solo quel file, senza toccare la logica.

---

## 6. Setup

**Requisiti:** Python 3.10+ e **FFmpeg** installato nel sistema (serve a `librosa`
per leggere l'audio dai `.mp4`).

```bash
# 1. Ambiente virtuale
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Dipendenze Python
pip install -r requirements.txt

# 3. FFmpeg (dipendenza di sistema)
#    Ubuntu/Debian:  sudo apt-get install ffmpeg
#    macOS (brew):   brew install ffmpeg
#    Windows:        https://ffmpeg.org  (aggiungere al PATH)
```

**Dati:** scaricare MUStARD da https://github.com/soujanyaporia/MUStARD e collocare
`sarcasm_data.json` in `data/raw/` e le clip in `data/raw/videos/`.

---

## 7. Esecuzione

Eseguire le quattro fasi **nell'ordine** (ognuna dipende dalla precedente):

```bash
python src/01_extract_text.py
python src/02_extract_audio.py
python src/03_extract_vision.py
python src/04_train_fusion.py
```

**Smoke test** — per verificare che tutto giri prima di lanciare l'intero dataset
(che può richiedere parecchi minuti, soprattutto la Fase 3), le fasi 1–3 accettano
`--limit` per processare solo poche clip:

```bash
python src/01_extract_text.py  --limit 20
python src/02_extract_audio.py --limit 20
python src/03_extract_vision.py --limit 20
python src/04_train_fusion.py
```

---

## 8. Risultati attesi e come leggerli

La Fase 4 stampa un `classification_report` con **precision, recall e F1** per
ciascuna classe, più la **matrice di confusione**. Trattandosi di un dataset
bilanciato, le metriche di riferimento sono **accuratezza** e **F1 macro**.

Cosa guardare: un esperimento interessante (e un bel risultato da mettere nel
paper) è **confrontare le singole modalità con la fusione** — addestrare il
classificatore prima solo sul testo, poi solo sull'audio, poi solo sulla vision,
e infine sui tre fusi. Se la fusione supera ogni singola modalità, si dimostra
sperimentalmente che il sarcasmo è un fenomeno multimodale: è il messaggio
centrale del progetto.

---

## 9. Risoluzione dei problemi più comuni

- **`FileNotFoundError: sarcasm_data.json`** → i dati non sono in `data/raw/`.
- **La Fase 2 fallisce su tutte le clip** → FFmpeg non è installato o non è nel PATH.
- **La Fase 3 è lentissima** → normale senza GPU; usa `--limit` per i test e, se
  possibile, una macchina con CUDA (il codice usa la GPU automaticamente se c'è).
- **`Manifest non trovato`** nelle fasi 2/3/4 → non hai eseguito prima la Fase 1.
- **Molte clip "senza volto" nei log della Fase 3** → atteso su alcune scene; il
  fallback sul frame intero (o lo zero-fill in fusione) gestisce il caso.

---

## 10. Scelte progettuali e limiti noti

- **Early Fusion** (concatenazione) è la strategia più semplice ed efficace per un
  PoC. Alternative più sofisticate (late fusion, attention cross-modale) sono
  possibili estensioni.
- **Lo split è stratificato 80/20.** Il paper originale di MUStARD valuta in
  modalità *speaker-independent* (5-fold), perché uno split casuale lascia gli
  stessi attori sia in train che in test e tende a gonfiare i risultati. Passare a
  quella valutazione è il primo upgrade consigliato per un confronto onesto con la
  letteratura.
- **Pooling statistico** per audio e vision: semplice ma perde l'informazione
  temporale fine. Un modello sequenziale (LSTM/Transformer sui frame) la
  recupererebbe, al costo di maggiore complessità.

---

## Riferimenti

Castro et al., *Towards Multimodal Sarcasm Detection (An _Obviously_ Perfect
Paper)*, ACL 2019 — dataset MUStARD: https://github.com/soujanyaporia/MUStARD