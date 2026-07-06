# Appunti — Migliorie da fare

> Stato al 7 luglio 2026. Ordinate per priorità; ogni voce dice dove si
> interviene e come si verifica il risultato.

## FATTO di recente (6-7 luglio 2026)

- **Fine-tuning del detector YOLO** (era B1): `models/best_finetuned.pt` (imgsz
  960), collegato in config (`YOLO_MODEL` + `YOLO_IMGSZ=960`). Detection su
  grafica nuova migliorata (portiere 40→66%, palla 27→47% su tiro_e_parata).
  Euristiche **ricalibrate** sulla vicinanza GK ora affidabile (via sequenziale
  della parata rimossa, `shot_goal_view_px` mci_bay 1200→600, parata-per-presa
  sulla scomparsa palla). Verificato: clip nuove City-Bayern OK (parata reale
  catturata, niente falsi positivi). ⚠️ Sulle clip vecchie bra_hai il fine-tuned
  introduce qualche parata falsa (dominio diverso) — legacy, fuori caso d'uso.
- **crowd_score → feature del modello prosodia** + poi **rinominato**
  `audio_energy` (vedi sotto "Provate e scartate": non è tifo, è energia audio).
- **Fase 1c ricalibrata**: score ri-scalato ai percentili della clip (ora i
  livelli low/medium/high/peak si popolano; prima "peak" non scattava mai);
  default neutro 0.5; Whisper `base`→`small`.
- **Fix qualità audio Fase 4**: output 24 kHz nativo (niente più "ovattato" dal
  downsampling a 22050); riferimento **mono**. Voce: **doppia** (era D1) →
  `COQUI_SPEAKER_WAV=lele_adani_mono` (base) + `_EXCITED=gol_saliente_mono`.
- **Groq / base_url configurabile** (era D3): `OpenAIProvider` con `base_url`,
  `LLM_CONFIG["groq"]` (Llama 70B), `--llm-provider groq`.
- **Fix bug naming Fase 2**: lo script era `<stem>_enriched.json` ma la Fase 4
  cercava `<stem>.json` → sintesi falliva su clip nuove. Ora normalizza lo stem.

## A. Priorità per la tesi (il contributo scientifico)

### A1. Dataset prosodico REALE (il punto più importante)
Il modello di prosodia oggi è addestrato su dati **sintetici derivati dalle
regole** (`build_dataset_synthetic` in `src/03_train_prosody.py`): il confronto
A/B "modello vs regole" è quasi nullo per costruzione, perché il modello ha
imparato le regole stesse. Serve annotare **50-100 segmenti di telecronaca
reale** (clip in `src/data/raw/commentary/` + righe in
`src/data/raw/prosody_annotations.csv`: clip, start, end, event_type).
Il codice di misura esiste già (`build_dataset_from_audio`): riempito il CSV,
basta rilanciare la Fase 3 senza `--synthetic`.
*Verifica*: `python test_prosody.py` — le colonne modello/regole devono
divergere davvero.
**MECCANISMO GIÀ PROVATO (6 lug)**: bootstrap con 10 segmenti-proxy → il
training reale gira e il modello **diverge** dalle regole (col sintetico erano
identici). Il CSV template è pronto in `data/raw/prosody_annotations.csv` con
10 righe d'esempio. Manca SOLO l'annotazione vera (a orecchio, per tipo di
evento). N.B.: le clip di riferimento (voce del telecronista) NON hanno tifo,
quindi `audio_energy` in questi campioni resta neutro (0.5).

### A2. Studio A/B con ascoltatori (Fase 5)
Mai eseguito con persone vere. Servono: 2 tracce per clip (già generabili),
10-20 ascoltatori, `python 05_evaluate.py make-study` + `analyze`.
Considerare il confronto a 3 condizioni: template+regole, template+modello,
LLM+modello (i file audio esistono già con nomi distinti).

## B. Robustezza del rilevamento eventi

### B1. Fine-tuning del detector YOLO — ✅ FATTO (vedi sezione "FATTO")
Completato il 6 lug: `best_finetuned.pt`, integrato, euristiche ricalibrate.
Resta aperto: il fine-tuned peggiora un po' sulle clip **vecchie** (bra_hai) —
se un giorno servono anche quelle, rifare il fine-tuning con più frame bra_hai
nel mix, oppure modello per-profilo.

### B2. Verifica eventi con VLM (ibrido "euristiche propongono, modello verifica")
Le euristiche hanno buon recall ma precisione fragile. Un modello visivo
verifica i soli candidati (2-3 frame attorno all'evento): robusto al cambio
di camera e alla grafica, zero soglie. Locale: `ollama pull qwen2.5vl:3b`
(~3.2 GB, entra nella RAM disponibile); oppure API cloud (centesimi/clip).
Da implementare come `EventVerifier` opzionale in `01b`.

### B3. Rilevamento su coordinate radar (semplificazione strutturale)
La minimappa è una vista dall'alto in coordinate campo, indipendente dalla
camera: spostare lì il rilevamento di tiro/parata eliminerebbe le soglie
per-profilo in pixel (`ball_tracking` nei profili), la memoria del portiere e
il vincolo porta-in-vista. `PossessionTracker` legge già palla e giocatori dal
radar (`RADAR_HSV` per-profilo). Rischio: risoluzione dei segnalini (3-5 px).
È la versione "game-native" della game state reconstruction di SoccerNet —
buon paragrafo di tesi.

## C. Fix piccoli già individuati (con casi concreti)

### C1. Riclassificare shot_off → shot_on_goal quando segue una parata
Coppia contraddittoria vista su tiro_e_parata: `shot_off` t=1.63 + `save`
t=3.27 (un tiro parato era nello specchio per definizione). Fix retroattivo in
`analyze_video`: se una save arriva entro `shot_to_save_max_s` da uno
shot_off, rietichettarlo. ~5 righe.

### C2. Il nome letto sul campo deve vincere sulla targhetta nel merge
Caso NEUER/HAALAND su tiro_fuori (t=13.5): possesso sbagliato per pressing
stretto → `merge_events` sceglie la targhetta della squadra sbagliata e il
tiro di Haaland risulta "di Neuer" (il portiere!). Se l'evento visivo porta
già il nome del portatore (`ControlledPlayerDetector`), quel nome deve avere
priorità sulla deduzione possesso→targhetta. ~4 righe in `merge_events`.

### C3. Validatore 2b: bloccare i nomi estranei a ENTRAMBE le rose
Il validatore intercetta i giocatori della rosa citati fuori contesto, ma non
i nomi completamente estranei: casi reali "Scarica Haaland." (su Brasile-Haiti)
e "Sampaio Corrêa". Aggiungere check sui nomi propri capitalizzati non
riconducibili a nessuna rosa quando l'evento ha un giocatore noto.

### C4. Possesso in pressing stretto (limite noto, documentare)
Il "puntino più vicino" sul radar sbaglia quando il portatore è marcato
spalla-a-spalla (finestra ~1s su tiro_fuori). Accettabile se documentato in
tesi; l'alternativa (margini/finestre) aggiunge complessità per un caso raro.

### C6. Turnover fantasma da cambio-targhetta stessa squadra (correttezza!)
Su EA SPORTS FC 26_20260705153103 (7 lug): 4 turnover in ~8s, ma il possesso
radar cambiava davvero solo 2 volte (t=14.71, 18.22). Causa: la Fase 1 (OCR)
spara un turnover a ogni cambio di NOME sulla targhetta, ma cambiare il
giocatore *selezionato* della STESSA squadra (STANISIC→MUSIALA, entrambi
Bayern) non è un turnover. Il merge sovrascrive `possession` col radar ma tiene
l'etichetta "turnover". Fix: in `merge_events`, validare i turnover contro la
timeline possesso radar — se il possesso non è cambiato attorno a t, il
turnover è spurio → riclassificarlo `pass` o eliminarlo. Alto impatto sulla
correttezza percepita (il ping-pong è vistoso all'ascolto). Anche: portiere
(NEUER) attribuito ad azioni di movimento → campanello di plausibilità (C2).

### C5. Auto-calibrazione HUD con le classi YOLO inutilizzate
`best.pt` conosce anche `score`, `gametime`, `team_name`, `playername`
(verificato: le playername box sono le DUE TARGHETTE in basso, conf
0.45-0.80). Un profilo nuovo potrebbe auto-localizzare le regioni HUD invece
della calibrazione manuale con 01_calibrate_hud. Sviluppo post-tesi.

## D. Audio e testo

### D1. Doppia voce XTTS — ✅ FATTO (vedi "FATTO")
Configurata: base `lele_adani_mono` + excited `gol_saliente_mono`. Resta il
**robotico residuo** dello zero-shot cloning: si abbatte solo con un
riferimento più lungo/pulito (30-60s dello stesso telecronista) o un
fine-tuning di voce dedicata. E la **pronuncia dei nomi stranieri** (NEUER,
KIMMICH letti all'italiana): fix cheap da provare = Title Case invece di
MAIUSCOLO nei nomi; fix completo = dizionario fonetico.

### D2. `test_prosody.py --say` richiede pyworld (non in requirements)
O si aggiunge `pyworld` a requirements.txt, o si migra il test alla sintesi
XTTS diretta. Senza `--say` il test funziona.

### D3. Provider LLM con base_url configurabile — ✅ FATTO (vedi "FATTO")
Groq disponibile con `--llm-provider groq` (serve `GROQ_API_KEY`). Da provare
per alzare la qualità del testo (Llama 70B vs il 3-4B locale).

### D4. Prompt hardening Fase 2b
Il modello 3B locale inventa ruoli e minutaggi ("l'imperatore del centrocampo",
"al quinto minuto del secondo tempo"). Stringere `LLM_SYSTEM_PROMPT` in
config.py (vietare riferimenti temporali non presenti nei dati, ecc.).

## Provate e scartate (non riprovare senza dati nuovi)

- **Modelli audio pre-addestrati per il "crowd/arousal"** (7 lug 2026):
  testati su 6 clip + audio di gioco: **AST** (audio tagging AudioSet) e **SER
  dimensionale** (`audeering/wav2vec2`, arousal). Entrambi **bocciati**:
  - AST classifica sempre "Speech", mai "Crowd/Cheering" → questo audio è VOCE
    (commento) + effetti, **non contiene tifo** da rilevare;
  - SER dà arousal ~0/negativo e non discrimina (calmo > urlato) → **mismatch di
    dominio** (modelli inglese/studio → italiano/gioco a bassa fedeltà).
  Conclusione: i pretrained non servono su questo audio. L'euristica acustica
  (ora rinominata `audio_energy`) è la scelta pragmatica. Un modello *addestrato
  sui dati tuoi* trasferirebbe — ma richiede annotazione (vedi A1). Prossima
  opzione robusta se serve rigore: **eGeMAPS/openSMILE** (feature acustiche
  standard, non un classifier → niente mismatch di dominio).
  → Da qui il **rename** `crowd_score`→`audio_energy`, `crowd_excitement`→
  `audio_energy_level`, `CrowdAnalyzer`→`AudioEnergyAnalyzer`: il nome ora dice
  cosa misura davvero (energia audio, non tifo).

- **Classe `goalpost` come segnale "porta inquadrata" nel tiro** (4 lug 2026):
  in OR col criterio del portiere riportava 3 tiri fantasma + 1 parata
  fantasma su possesso_palla_1 e una parata inesistente su tiro_fuori (falsi
  positivi del palo a centrocampo). Revert completo, regressione verificata.
  Potrebbe tornare utile solo DOPO il fine-tuning del detector (B1), se i
  falsi positivi del palo spariscono.

## E. Pulizia

- `src/00b_setup_match.py` genera `src/data/teams.json` che nessuno legge:
  collegarlo a `RADAR_HSV` (misura automatica dei colori radar!) o rimuoverlo.
- `ControlledPlayerDetector`: valutare la sostituzione con la deduzione
  possesso+targhetta in `merge_events` (proposta discussa; il proprietario ha
  preferito tenerla — riconsiderare dopo C2).
- Cartella `experiments/`: script scratch da rivedere/archiviare.
