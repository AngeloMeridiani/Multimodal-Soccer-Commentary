# 🎙️ The Multimodal Caressa: Automated Sports Commentary via Audio-Visual Fusion

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-ffcc00.svg)

> **Corso:** AI Lab: Computer Vision and NLP - Università La Sapienza (A.A. 2025/2026)  

## 📖 Abstract del Progetto

Questo progetto di ricerca esplora l'integrazione di tecniche di **Computer Vision** e **Signal Analysis** per generare telecronache sportive automatiche tramite un **Large Language Model (LLM)**.

Il nostro obiettivo (Research Question) è dimostrare come l'aggiunta dell'analisi dell'inviluppo acustico (il "boato" del pubblico) come condizionamento per il modello linguistico, generi descrizioni più coerenti emotivamente rispetto a una pipeline puramente visiva.

## 🏗️ Architettura Multimodale

Il sistema è composto da 3 moduli indipendenti:

1. **Modulo Visivo (Zero-Shot Action Recognition):** Utilizzo di `CLIP` (OpenAI) per estrarre il contesto dell'azione frame per frame dai video.
2. **Modulo Segnali (Audio Emotion Analysis):** Estrazione della *Root Mean Square (RMS) Energy* tramite `librosa` per misurare il livello di eccitazione della folla.
3. **Modulo NLP (Generazione Condizionata):** Prompting dinamico su modelli LLM avanzati per fondere l'azione visiva e il carico emotivo sonoro in una telecronaca naturale.

## 📂 Struttura delle Directory

Di seguito è riportata la struttura dei file e delle cartelle del progetto:

```text
multimodal-caressa/
├── data/                  # Cartella per i dati (esclusa da Git)
│   ├── raw_videos/        # Clip .mp4 originali scaricate da YouTube
│   └── processed/         # File JSON intermedi estratti dai modelli
├── notebooks/             # Jupyter Notebooks per analisi statistica
│   └── 01_data_exploration.ipynb
├── src/                   # Codice sorgente dei moduli principali
│   ├── __init__.py
│   ├── audio_module.py    # Modulo Signal Processing (Librosa)
│   ├── cv_module.py       # Modulo Computer Vision (CLIP)
│   └── nlp_module.py      # Modulo Generazione Testo (LLM Prompt)
├── .env                   # File per le chiavi API private
├── .gitignore             # Configurazione per escludere file pesanti o privati
├── app.py                 # Interfaccia grafica della demo (Streamlit)
├── README.md              # Documentazione del progetto
└── requirements.txt       # Elenco delle dipendenze Python