#!/usr/bin/env python3
"""
serve.py  (v4 - cambio voce veloce)
===================================
Collega il frontend alla pipeline e sceglie automaticamente la strada:

  - PRIMA generazione di una clip  -> pipeline COMPLETA (1 1b 1c 2b 3 4)
  - STESSA clip, voce diversa       -> solo Fase 4 (sintesi) + montaggio  (VELOCE)

Come lo capisce: se esiste gia' features/scripts/<nome>_llm.json (il testo LLM
di quella clip), il testo e' gia' pronto e non va rigenerato: basta risintetizzare
la voce nuova. XTTS clona sempre da data/raw/commentary/ref.wav, quindi il server
copia la voce scelta su ref.wav prima di sintetizzare.

METTI in src\\ e lancia da src\\ (con l'ambiente (telecronaca) attivo):
    python serve.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import json
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent
GAMEPLAY = ROOT / "data" / "raw" / "gameplay"
COMMENTARY = ROOT / "data" / "raw" / "commentary"
REF = COMMENTARY / "ref.wav"
AUDIO_OUT = ROOT / "outputs" / "audio"
VIDEO_OUT = ROOT / "outputs" / "video"
SCRIPTS = ROOT / "features" / "scripts"
WEB = ROOT / "web"
LOG = ROOT / "outputs" / "last_run.log"

for d in (GAMEPLAY, COMMENTARY, VIDEO_OUT, AUDIO_OUT):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


def pretty(stem: str) -> str:
    return " ".join(w.capitalize() for w in stem.replace("-", "_").split("_") if w)


def media_duration(path) -> float | None:
    """Durata in secondi di un file audio/video via ffprobe, o None se illeggibile."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", str(path)],
            capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except (ValueError, OSError):
        return None


def list_voices():
    out = []
    for p in sorted(COMMENTARY.glob("*.wav")):
        if p.name.lower() == "ref.wav":
            continue
        out.append({"id": p.name, "name": pretty(p.stem)})
    return out


def run_pipeline(relvideo, phases, lang, env):
    """Esegue run_pipeline.py con le fasi date. Ritorna (returncode, log_tail)."""
    cmd = [sys.executable, "run_pipeline.py"]
    cmd += ["--video", relvideo]
    cmd += ["--phases", *phases,
            "--profile", "auto",
            "--llm-provider", "ollama",
            "--llm-text",
            "--language", lang]
    with open(LOG, "w", encoding="utf-8") as lf:
        lf.write("Fasi: " + " ".join(phases) + "\nComando: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                              stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode, LOG.read_text(encoding="utf-8", errors="replace")[-3500:]


@app.route("/")
def home():
    return send_from_directory(WEB, "telecronaca.html")


@app.route("/voci")
def voci():
    return jsonify(voices=list_voices())


@app.route("/voce/<path:name>")
def voce(name):
    p = (COMMENTARY / name).resolve()
    if p.parent != COMMENTARY.resolve() or not p.exists():
        return "Voce non trovata", 404
    return send_file(p)


@app.route("/genera", methods=["POST"])
def genera():
    if "clip" not in request.files:
        return jsonify(ok=False, error="Nessuna clip ricevuta"), 400

    voice = (request.form.get("voice") or "").strip()
    if not voice:
        return jsonify(ok=False, error="Seleziona una voce prima di generare."), 400
    src_wav = (COMMENTARY / voice).resolve()
    if src_wav.parent != COMMENTARY.resolve() or not src_wav.exists():
        return jsonify(ok=False, error=f"Voce non trovata: {voice}"), 400
    shutil.copyfile(src_wav, REF)          # la voce scelta -> ref.wav

    lang = request.form.get("lang", "it")
    f = request.files["clip"]
    dest = GAMEPLAY / f.filename
    f.save(dest)
    stem = dest.stem
    relvideo = str(dest.relative_to(ROOT))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # ── decide la strada: testo gia' pronto? -> solo Fase 4 ───────────
    llm_json = SCRIPTS / f"{stem}_llm.json"
    if llm_json.exists():
        phases = ["4"]
        mode = "fast"     # cambio voce veloce
    else:
        phases = ["1", "1b", "1c", "2b", "3", "4"]
        mode = "full"     # prima volta: pipeline completa

    rc, log_tail = run_pipeline(relvideo, phases, lang, env)

    # Cerca il WAV della telecronaca PRIMA di giudicare l'esito.
    # XTTS su Windows a volte crasha in chiusura (exit code 3221225477 / 0xC0000005)
    # DOPO aver gia' salvato l'audio: in quel caso il WAV c'e' ed e' valido,
    # quindi il risultato e' comunque utilizzabile e procediamo al montaggio.
    wavs = sorted(AUDIO_OUT.glob(f"{stem}*.wav"), key=lambda p: p.stat().st_mtime)
    wavs = [w for w in wavs if w.stat().st_size > 4096]   # WAV con contenuto reale
    if wavs:
        wav = wavs[-1]                # audio prodotto: si prosegue anche se rc != 0
    else:
        # nessun audio valido -> errore vero
        return jsonify(ok=False, error="Pipeline fallita (nessun audio generato).",
                       log=log_tail, returncode=rc, mode=mode), 500

    out_mp4 = VIDEO_OUT / f"{stem}_telecronaca.mp4"
    vdur = media_duration(dest)
    adur = media_duration(wav)
    # Se la telecronaca dura PIU' del video, con -shortest l'output finirebbe
    # con il video e l'ultima battuta verrebbe tagliata. Invece CONGELIAMO
    # l'ultimo fotogramma finche' la voce finisce di parlare (tpad clona il
    # frame finale per la differenza di durata). Niente -shortest in nessun
    # caso: l'audio non deve mai essere troncato.
    if adur and vdur and adur > vdur + 0.05:
        pad = adur - vdur
        mux = ["ffmpeg", "-y", "-i", str(dest), "-i", str(wav),
               "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f}[v]",
               "-map", "[v]", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out_mp4)]
    else:
        # Video piu' lungo (o uguale): copia diretta del video. Senza -shortest
        # l'audio suona per intero; dopo la voce resta il video (muto).
        mux = ["ffmpeg", "-y", "-i", str(dest), "-i", str(wav),
               "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", str(out_mp4)]
    m = subprocess.run(mux, capture_output=True)
    if m.returncode != 0:
        return jsonify(ok=False, error="Montaggio ffmpeg fallito",
                       log=m.stderr.decode("utf-8", "replace")[-2000:], mode=mode), 500

    battute = []
    if llm_json.exists():
        try:
            data = json.loads(llm_json.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("lines", data.get("battute", []))
            for it in items:
                battute.append({"t": str(it.get("time", it.get("t", ""))),
                                "s": it.get("text", it.get("s", it.get("testo", "")))})
        except Exception:
            pass

    return jsonify(ok=True, file=out_mp4.name, audio=wav.name,
                   voice=voice, mode=mode, transcript=battute)


@app.route("/scarica/<path:name>")
def scarica(name):
    return send_file(VIDEO_OUT / name)


if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        print("ATTENZIONE: ffmpeg non nel PATH.")
    print("Python:", sys.executable)
    print("Voci trovate:", [v["name"] for v in list_voices()] or "NESSUNA")
    print("Apri  ->  http://localhost:8000")
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)