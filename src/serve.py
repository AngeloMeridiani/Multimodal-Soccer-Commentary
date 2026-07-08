#!/usr/bin/env python3
"""
serve.py  (v4 - fast voice change)
===================================
Connects the frontend to the pipeline and picks the route automatically:

  - FIRST generation of a clip   -> FULL pipeline (1 1b 1c 2b 3 4)
  - SAME clip, different voice    -> only Phase 4 (synthesis) + muxing  (FAST)

How it knows: if features/scripts/<name>_llm.json (the LLM text of that clip)
already exists, the text is ready and does not need regenerating: it is enough
to re-synthesize the new voice. XTTS always clones from
data/raw/commentary/ref.wav, so the server copies the chosen voice onto ref.wav
before synthesizing.

PUT it in src\\ and launch from src\\ (with the (telecronaca) environment active):
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
    """
    Format a filename stem into a human-readable title.
    """
    return " ".join(w.capitalize() for w in stem.replace("-", "_").split("_") if w)


def media_duration(path) -> float | None:
    """Duration in seconds of an audio/video file via ffprobe, or None if unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", str(path)],
            capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except (ValueError, OSError):
        return None


# special id: XTTS native voice (no character cloning).
ORIGINAL_VOICE_ID = "__originale__"


def list_voices():
    """
    Scan the commentary directory for available speaker wavs and return them as a list of dicts.
    Always includes the built-in original voice as the first option.
    """
    # The "Original" voice (neutral, built-in XTTS) always at the top.
    out = [{"id": ORIGINAL_VOICE_ID, "name": "Originale (neutra)"}]
    for p in sorted(COMMENTARY.glob("*.wav")):
        if p.name.lower() == "ref.wav":
            continue
        out.append({"id": p.name, "name": pretty(p.stem)})
    return out


def run_pipeline(relvideo, phases, lang, env):
    """Run run_pipeline.py with the given phases. Returns (returncode, log_tail)."""
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
    """
    Serve the main frontend HTML interface.
    """
    return send_from_directory(WEB, "telecronaca.html")


@app.route("/voci")
def voci():
    """
    API endpoint returning a JSON list of available voices.
    """
    return jsonify(voices=list_voices())


@app.route("/voce/<path:name>")
def voce(name):
    """
    Serve an audio file representing a specific voice from the commentary directory.
    Prevents path traversal attacks.
    """
    p = (COMMENTARY / name).resolve()
    if p.parent != COMMENTARY.resolve() or not p.exists():
        return "Voce non trovata", 404
    return send_file(p)


@app.route("/genera", methods=["POST"])
def genera():
    """
    Main API endpoint to generate commentary for an uploaded video.
    Handles file upload, voice selection, language, routing between full pipeline
    and fast re-synthesis, and final audio/video muxing.
    """
    if "clip" not in request.files:
        return jsonify(ok=False, error="Nessuna clip ricevuta"), 400

    voice = (request.form.get("voice") or "").strip()
    if not voice:
        return jsonify(ok=False, error="Seleziona una voce prima di generare."), 400
    use_builtin = voice == ORIGINAL_VOICE_ID
    if not use_builtin:
        src_wav = (COMMENTARY / voice).resolve()
        if src_wav.parent != COMMENTARY.resolve() or not src_wav.exists():
            return jsonify(ok=False, error=f"Voce non trovata: {voice}"), 400
        shutil.copyfile(src_wav, REF)      # the chosen voice -> ref.wav (cloning)

    lang = request.form.get("lang", "it")
    f = request.files["clip"]
    dest = GAMEPLAY / f.filename
    f.save(dest)
    stem = dest.stem
    relvideo = str(dest.relative_to(ROOT))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Original/neutral voice: Phase 4 uses the built-in XTTS voice instead of
    # cloning from ref.wav. Propagated to subprocesses via env (config re-reads it).
    env["COMMENTARY_SPEAKER"] = "builtin" if use_builtin else ""

    # ── pick the route: text already ready IN THE CHOSEN LANGUAGE? -> only Phase 4
    # The text cache is PER LANGUAGE: {stem}_{lang}_llm.json. Without this, text
    # generated in English would be re-synthesized even when selecting Italian
    # (Phase 2b skipped) -> commentary in the wrong language.
    llm_json = SCRIPTS / f"{stem}_llm.json"          # file that Phase 4 reads
    llm_json_lang = SCRIPTS / f"{stem}_{lang}_llm.json"  # per-language archive
    if llm_json_lang.exists():
        shutil.copyfile(llm_json_lang, llm_json)     # restore the right text
        phases = ["4"]
        mode = "fast"     # same clip+language: re-synthesis only (voice change)
    else:
        phases = ["1", "1b", "1c", "2b", "3", "4"]
        mode = "full"     # first time in this language: full pipeline

    rc, log_tail = run_pipeline(relvideo, phases, lang, env)

    # Archive the just-generated text under its language (for the future
    # fast-path). After a full run, {stem}_llm.json is in the requested language.
    if mode == "full" and llm_json.exists():
        shutil.copyfile(llm_json, llm_json_lang)

    # Look for the commentary WAV BEFORE judging the outcome.
    # XTTS on Windows sometimes crashes on shutdown (exit code 3221225477 / 0xC0000005)
    # AFTER already saving the audio: in that case the WAV exists and is valid,
    # so the result is still usable and we proceed to the muxing.
    wavs = sorted(AUDIO_OUT.glob(f"{stem}*.wav"), key=lambda p: p.stat().st_mtime)
    wavs = [w for w in wavs if w.stat().st_size > 4096]   # WAV with real content
    if wavs:
        wav = wavs[-1]                # audio produced: continue even if rc != 0
    else:
        # no valid audio -> real error
        return jsonify(ok=False, error="Pipeline fallita (nessun audio generato).",
                       log=log_tail, returncode=rc, mode=mode), 500

    out_mp4 = VIDEO_OUT / f"{stem}_telecronaca.mp4"
    vdur = media_duration(dest)
    adur = media_duration(wav)
    # If the commentary lasts LONGER than the video, with -shortest the output
    # would end with the video and the last line would be cut. Instead we FREEZE
    # the last frame until the voice finishes speaking (tpad clones the final
    # frame for the duration difference). No -shortest in any case: the audio
    # must never be truncated.
    if adur and vdur and adur > vdur + 0.05:
        pad = adur - vdur
        mux = ["ffmpeg", "-y", "-i", str(dest), "-i", str(wav),
               "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f}[v]",
               "-map", "[v]", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out_mp4)]
    else:
        # Video longer (or equal): direct copy of the video. Without -shortest
        # the audio plays in full; after the voice the video remains (muted).
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
    """
    Serve the final muxed video file for download or playback.
    """
    return send_file(VIDEO_OUT / name)


if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        print("ATTENZIONE: ffmpeg non nel PATH.")
    print("Python:", sys.executable)
    print("Voci trovate:", [v["name"] for v in list_voices()] or "NESSUNA")
    print("Apri  ->  http://localhost:8000")
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)
