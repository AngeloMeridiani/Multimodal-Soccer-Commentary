🎙️ The Multimodal Caressa: Automated Sports Commentary via Audio-Visual Fusion

Corso: AI Lab: Computer Vision and NLP - Università La Sapienza (A.A. 2025/2026)

📖 Abstract del Progetto

Questo progetto di ricerca esplora l'integrazione di tecniche di Computer Vision e Signal Analysis per generare telecronache sportive automatiche tramite un Large Language Model (LLM).

Il nostro obiettivo (Research Question) è dimostrare come l'aggiunta dell'analisi dell'inviluppo acustico (il "boato" del pubblico) come condizionamento per il modello linguistico, generi descrizioni più coerenti emotivamente rispetto a una pipeline puramente visiva.

🏗️ Architettura Multimodale

Il sistema (Early/Late Fusion) è composto da 3 moduli indipendenti:

Modulo Visivo (Zero-Shot Action Recognition): Utilizzo di CLIP (OpenAI) per estrarre il contesto dell'azione frame per frame dai video.

Modulo Segnali (Audio Emotion Analysis): Estrazione della Root Mean Square (RMS) Energy tramite librosa per misurare il livello di eccitazione della folla.

Modulo NLP (Generazione Condizionata): Prompting dinamico su modelli LLM avanzati per fondere l'azione visiva e il carico emotivo sonoro in una telecronaca naturale.

⚙️ Installazione e Utilizzo (Local Setup)

Clona la repository:

git clone [https://github.com/tuo-username/multimodal-caressa.git](https://github.com/tuo-username/multimodal-caressa.git)
cd multimodal-caressa


Installa le dipendenze:

pip install -r requirements.txt


(Opzionale) Inserisci la tua API Key nel file .env:

LLM_API_KEY=la_tua_chiave_qui


Lancia l'interfaccia interattiva:

streamlit run app.py


📊 Dataset e Valutazione (Ablation Study)

Il sistema è stato testato su un dataset proprietario di 40 clip raccolte da YouTube (In-the-wild videos).
La valutazione delle performance del modello linguistico è stata condotta confrontando la generazione Vision-Only contro la generazione Multimodal (Vision+Audio) utilizzando le metriche BLEU Score e ROUGE-L. I risultati completi sono consultabili nella cartella /notebooks.


Disclaimer: Progetto a scopo puramente accademico e di ricerca.
