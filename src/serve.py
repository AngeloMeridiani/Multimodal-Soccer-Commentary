#!/usr/bin/env python3
"""
serve.py  (v2 - robusto su Windows)
===================================
Collega il frontend (web/telecronaca.html) alla pipeline reale.

Differenze rispetto alla v1:
  - forza UTF-8 nei sottoprocessi  -> niente crash di decodifica su Windows
    (i caratteri stampati dalla pipeline non rompono piu' nulla)
  - scrive TUTTO l'output della pipeline in outputs/last_run.log
  - in caso di errore restituisce al browser le ultime righe reali del log

METTI questo file in src\\ (accanto a config.py) e lancia da src\\:
    python serve.py
Poi apri  http://localhost:8000
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import json
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent          # = src\\  (dove sta config.py)
GAMEPLAY = ROOT / "data" / "raw" / "gameplay"
AUDIO_OUT = ROOT / "outputs" / "audio"
VIDEO_OUT = ROOT / "outputs" / "video"
SCRIPTS = ROOT / "features" / "scripts"
WEB = ROOT / "web"
LOG = ROOT / "outputs" / "last_run.log"

for d in (GAMEPLAY, VIDEO_OUT, AUDIO_OUT):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB


@app.route("/")
def home():
    return send_from_directory(WEB, "telecronaca.html")


@app.route("/genera", methods=["POST"])
def genera():
    if "clip" not in request.files:
        return jsonify(ok=False, error="Nessuna clip ricevuta"), 400

    lang = request.form.get("lang", "it")
    f = request.files["clip"]
    dest = GAMEPLAY / f.filename
    f.save(dest)
    stem = dest.stem

    # ambiente UTF-8 per i sottoprocessi (fix Windows)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # pipeline reale (stesso comando che funziona da terminale)
    cmd = [
        sys.executable, "run_pipeline.py",
        "--video", str(dest.relative_to(ROOT)),
        "--phases", "1", "1b", "1c", "2b", "3", "4",
        "--profile", "auto",
        "--llm-provider", "ollama",
        "--llm-text",
        "--language", lang,
    ]

    # output su file: niente decodifica in memoria = niente crash unicode
    with open(LOG, "w", encoding="utf-8") as lf:
        lf.write("Comando: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env,
            stdout=lf, stderr=subprocess.STDOUT,
        )

    log_tail = LOG.read_text(encoding="utf-8", errors="replace")[-3500:]

    if proc.returncode != 0:
        return jsonify(ok=False, error="Pipeline fallita",
                       log=log_tail, returncode=proc.returncode), 500

    # trova il WAV della telecronaca (Fase 4 aggiunge _model / _rule)
    wavs = sorted(AUDIO_OUT.glob(f"{stem}*.wav"), key=lambda p: p.stat().st_mtime)
    wavs = [w for w in wavs if w.stat().st_size > 1024]   # scarta WAV vuoti
    if not wavs:
        return jsonify(ok=False,
                       error="Nessun audio valido generato (WAV vuoto). "
                             "Controlla ref.wav in data/raw/commentary/.",
                       log=log_tail), 500
    wav = wavs[-1]

    # mux ffmpeg: SOLO video + SOLO telecronaca (niente audio originale)
    out_mp4 = VIDEO_OUT / f"{stem}_telecronaca.mp4"
    mux = [
        "ffmpeg", "-y",
        "-i", str(dest),
        "-i", str(wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out_mp4),
    ]
    m = subprocess.run(mux, capture_output=True)
    if m.returncode != 0:
        return jsonify(ok=False, error="Montaggio ffmpeg fallito",
                       log=m.stderr.decode("utf-8", "replace")[-2000:]), 500

    # testo generato per il frontend
    battute = []
    script_json = SCRIPTS / f"{stem}_llm.json"
    if script_json.exists():
        try:
            data = json.loads(script_json.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("lines", data.get("battute", []))
            for it in items:
                battute.append({
                    "t": str(it.get("time", it.get("t", ""))),
                    "s": it.get("text", it.get("s", it.get("testo", ""))),
                })
        except Exception:
            pass

    return jsonify(ok=True, file=out_mp4.name, audio=wav.name, transcript=battute)


@app.route("/scarica/<path:name>")
def scarica(name):
    return send_file(VIDEO_OUT / name)


if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        print("ATTENZIONE: ffmpeg non nel PATH.")
    print("Apri  ->  http://localhost:8000")
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)