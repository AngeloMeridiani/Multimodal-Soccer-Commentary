#!/usr/bin/env python3
"""
run_pipeline.py
===============
Esegue l'intera pipeline di telecronaca in un solo comando.

Uso:
    # Pipeline completa su un video:
    python3 run_pipeline.py --video data/raw/gameplay/possesso_palla_1.mov

    # Pipeline completa su TUTTI i video nella cartella:
    python3 run_pipeline.py --all

    # Solo alcune fasi (es. solo Fase 1 e 2):
    python3 run_pipeline.py --video <clip> --phases 1 2

    # Fase 4 con entrambe le modalità (flat + learned) per confronto:
    python3 run_pipeline.py --video <clip> --phases 4 --both-modes

    # Pipeline completa + studio A/B (Fase 5):
    python3 run_pipeline.py --video <clip> --phases 1 2 3 4 5

    # Usa un profilo HUD specifico:
    python3 run_pipeline.py --video <clip> --profile tot_om
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Colori per output leggibile ──────────────────────────────────────────── #
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

SRC_DIR = Path(__file__).resolve().parent
PHASE_ORDER = ["1", "1b", "1c", "2", "2b", "3", "4", "5"]


def banner(text: str) -> None:
    width = 64
    print(f"\n{CYAN}{BOLD}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{RESET}\n")


def step_header(phase: int | str, desc: str) -> None:
    print(f"{BOLD}{YELLOW}── Fase {phase}: {desc} ──{RESET}")


def run(cmd: list[str], label: str) -> bool:
    """Esegue un comando e stampa il risultato."""
    print(f"  {CYAN}▸{RESET} {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  {GREEN}✓ {label} completata ({elapsed:.1f}s){RESET}\n")
        return True
    else:
        print(f"  {RED}✗ {label} fallita (exit code {result.returncode}){RESET}\n")
        return False


def get_stem(video_path: str) -> str:
    """Nome del video senza estensione (usato per i file intermedi)."""
    return Path(video_path).stem


def discover_videos() -> list[str]:
    """Trova tutti i video nella cartella gameplay."""
    gameplay_dir = SRC_DIR / "data" / "raw" / "gameplay"
    exts = {".mov", ".mp4", ".avi", ".mkv", ".webm"}
    videos = sorted(
        str(p)
        for p in gameplay_dir.iterdir()
        if p.suffix.lower() in exts and not p.name.startswith(".")
    )
    return videos


def run_pipeline(
    video: str,
    phases: list[str],
    profile: str,
    both_modes: bool,
    engine: str,
    epochs: int,
    no_whisper: bool = False,
    llm_provider: str | None = None,
    use_llm_text: bool = False,
) -> dict[str, bool]:
    """Esegue le fasi selezionate per un singolo video."""
    stem = get_stem(video)
    events_json = f"features/events/{stem}.json"
    enriched_json = f"features/events/{stem}_enriched.json"
    script_json = f"features/scripts/{stem}.json"
    results: dict[str, bool] = {}

    # ── FASE 1: Estrazione eventi (OCR HUD) ──────────────────────────── #
    if "1" in phases:
        step_header(1, "Estrazione eventi (OCR HUD)")
        cmd = [sys.executable, "01_extract_events.py", "--video", video]
        if profile:
            cmd += ["--profile", profile]
        results["1"] = run(cmd, "Fase 1")
        if not results["1"]:
            return results

    # ── FASE 1b: Analisi visiva (YOLO + Tracking) ────────────────────── #
    if "1b" in phases:
        step_header("1b", "Analisi visiva (YOLO + Tracking)")
        cmd = [sys.executable, "01b_visual_analysis.py", "--video", video]
        if profile:
            cmd += ["--profile", profile]
        if (SRC_DIR / events_json).exists():
            cmd += ["--merge", events_json]
        results["1b"] = run(cmd, "Fase 1b")
        if not results["1b"]:
            return results

    # ── FASE 1c: Analisi audio (Tifo + Whisper) ──────────────────────── #
    if "1c" in phases:
        step_header("1c", "Analisi audio (Tifo + Whisper)")
        cmd = [sys.executable, "01c_audio_analysis.py", "--video", video]
        if no_whisper:
            cmd += ["--no-whisper"]
        if (SRC_DIR / enriched_json).exists():
            cmd += ["--events", enriched_json]
        elif (SRC_DIR / events_json).exists():
            cmd += ["--events", events_json]
        results["1c"] = run(cmd, "Fase 1c")
        if not results["1c"]:
            return results

    # ── FASE 2: Generazione testo telecronaca (template) ─────────────── #
    if "2" in phases:
        step_header(2, "Generazione testo telecronaca")
        best_events = enriched_json if (SRC_DIR / enriched_json).exists() else events_json
        results["2"] = run(
            [sys.executable, "02_generate_script.py", "--events", best_events],
            "Fase 2",
        )
        if not results["2"]:
            return results

    # ── FASE 2b: Generazione testo con LLM ───────────────────────────── #
    if "2b" in phases:
        step_header("2b", "Generazione testo con LLM")
        best_events = enriched_json if (SRC_DIR / enriched_json).exists() else events_json
        cmd = [sys.executable, "02b_generate_llm.py", "--events", best_events]
        if llm_provider:
            cmd += ["--provider", llm_provider]
        results["2b"] = run(cmd, "Fase 2b")
        if not results["2b"]:
            return results

    # ── FASE 3: Training prosodia ────────────────────────────────────── #
    if "3" in phases:
        step_header(3, "Addestramento modello prosodia")
        cmd = [sys.executable, "03_train_prosody.py", "--synthetic"]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        results["3"] = run(cmd, "Fase 3")
        if not results["3"]:
            return results

    # ── FASE 4: Sintesi audio ────────────────────────────────────────── #
    if "4" in phases:
        step_header(4, "Sintesi audio")
        llm_script = f"features/scripts/{stem}_llm.json"
        # Scelta del TESTO da sintetizzare. Prima il template vinceva sempre
        # (lo script LLM veniva usato solo se il template mancava): la Fase 2b
        # girava per niente. Ora l'LLM ha la precedenza quando e' richiesto
        # esplicitamente (--llm-text) o quando la 2b fa parte di questa run;
        # in ogni caso si ripiega sull'altro file se il preferito non esiste.
        prefer_llm = use_llm_text or "2b" in phases
        candidates = [llm_script, script_json] if prefer_llm else [script_json, llm_script]
        actual_script = next((c for c in candidates if (SRC_DIR / c).exists()), script_json)
        print(f"  {CYAN}Testo:{RESET} {actual_script}")
        cmd = [sys.executable, "04_synthesize.py", "--script", actual_script]
        if engine:
            cmd += ["--engine", engine]
        results["4"] = run(cmd, "Fase 4 (learned)")

        # Anche rule-based se richiesto
        if both_modes:
            cmd_rule = [
                sys.executable,
                "04_synthesize.py",
                "--script",
                actual_script,
                "--rule-based",
            ]
            if engine:
                cmd_rule += ["--engine", engine]
            ok = run(cmd_rule, "Fase 4 (rule-based)")
            results["4"] = results["4"] and ok

    # ── FASE 5: Valutazione A/B ──────────────────────────────────────── #
    if "5" in phases:
        step_header(5, "Preparazione studio A/B")
        results["5"] = run(
            [sys.executable, "05_evaluate.py", "make-study", "--names", stem],
            "Fase 5",
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Esegue l'intera pipeline di telecronaca in un solo comando.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python3 run_pipeline.py --video data/raw/gameplay/possesso_palla_1.mov
  python3 run_pipeline.py --all
  python3 run_pipeline.py --video <clip> --phases 1 2
  python3 run_pipeline.py --video <clip> --phases 4 --both-modes
  python3 run_pipeline.py --video <clip> --profile tot_om
        """,
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Percorso del video di gameplay.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Esegui la pipeline su TUTTI i video in data/raw/gameplay/.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=str,
        default=["1", "1b", "1c", "2", "2b", "3", "4"],
        help="Fasi da eseguire (default: 1 1b 1c 2 2b 3 4). Usa 5 per lo studio A/B.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="auto",
        help="Profilo HUD (default: auto). Vedi config.HUD_PROFILES.",
    )
    parser.add_argument(
        "--both-modes",
        action="store_true",
        help="Genera sia learned che rule-based (per confronto/studio).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=["coqui"],
        help="Motore TTS (unico supportato: coqui / XTTS v2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Numero epoche per Fase 3 (default: da config.py).",
    )
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disabilita trascrizione Whisper nella Fase 1c.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["ollama", "openai", "anthropic", "groq"],
        help="Provider LLM per la Fase 2b (default: da config.py).",
    )
    parser.add_argument(
        "--llm-text",
        action="store_true",
        help="La Fase 4 sintetizza il testo dell'LLM (<nome>_llm.json) invece del "
        "template. Automatico quando la Fase 2b e' tra le fasi eseguite.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=None,  # validato sotto contro config.SUPPORTED_LANGUAGES
        help="Lingua della telecronaca (default: da config.py). "
        f"Supportate: {', '.join(__import__('config').SUPPORTED_LANGUAGES)}.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Percorso del wav di riferimento per la voce (voice cloning XTTS), "
        "relativo alla root del progetto (es. data/raw/commentary/voce_caressa.wav). "
        "Sovrascrive la voce di default per tutta la telecronaca.",
    )

    args = parser.parse_args()

    # --- Lingua ---------------------------------------------------------------- #
    import config as _cfg
    if args.language:
        if args.language not in _cfg.SUPPORTED_LANGUAGES:
            parser.error(
                f"Lingua '{args.language}' non supportata. "
                f"Scegli tra: {', '.join(_cfg.SUPPORTED_LANGUAGES)}"
            )
        # Riscrive la lingua e tutte le costanti derivate.
        # IMPORTANTE: imposta anche la variabile d'ambiente, perche' i
        # sottoprocessi (subprocess.run) reimportano config.py da zero.
        import os
        os.environ["COMMENTARY_LANGUAGE"] = args.language
        _cfg.LANGUAGE = args.language
        _cfg.COQUI_LANGUAGE = args.language
        _cfg.WHISPER_LANGUAGE = args.language
        _cfg.TEMPLATES = _cfg.get_templates(args.language)
        _cfg.LLM_SYSTEM_PROMPT = _cfg.LLM_SYSTEM_PROMPTS.get(
            args.language, _cfg.LLM_SYSTEM_PROMPTS["en"]
        )
        _cfg.COQUI_SPEAKER_WAV = _cfg.COQUI_SPEAKER_WAVS.get(args.language)
        _cfg.COQUI_SPEAKER_WAV_EXCITED = _cfg.COQUI_SPEAKER_WAVS_EXCITED.get(args.language)

    # --- Voce (voice cloning) -------------------------------------------------- #
    # La voce scelta dall'utente sovrascrive quella derivata dalla lingua. Come
    # per la lingua, si imposta anche la variabile d'ambiente perche' i
    # sottoprocessi (subprocess.run) reimportano config.py da zero.
    if args.voice:
        voice_path = Path(args.voice)
        abs_voice = voice_path if voice_path.is_absolute() else SRC_DIR / voice_path
        if not abs_voice.exists():
            parser.error(f"Voce non trovata: {abs_voice}")
        import os
        os.environ["COMMENTARY_VOICE"] = args.voice
        _cfg.COMMENTARY_VOICE = args.voice
        _cfg.COQUI_SPEAKER_WAV = args.voice
        _cfg.COQUI_SPEAKER_WAV_EXCITED = args.voice

    # Determina i video da processare
    if args.all:
        videos = discover_videos()
        if not videos:
            print(f"{RED}Nessun video trovato in data/raw/gameplay/{RESET}")
            sys.exit(1)
        print(f"{GREEN}Trovati {len(videos)} video.{RESET}")
    elif args.video:
        videos = [args.video]
    else:
        parser.error("Specifica --video <percorso> oppure --all.")

    banner("PIPELINE TELECRONACA AI PER EA FC / FIFA")
    print(f"  Video:  {len(videos)}")
    _sort = lambda p: PHASE_ORDER.index(p) if p in PHASE_ORDER else 99
    print(f"  Fasi:   {sorted(args.phases, key=_sort)}")
    print(f"  Profilo: {args.profile}")
    if args.both_modes:
        print(f"  Modalità: learned + rule-based")
    print()

    t_start = time.time()
    summary: dict[str, dict[str, bool]] = {}

    for i, video in enumerate(videos, 1):
        stem = get_stem(video)
        banner(f"[{i}/{len(videos)}]  {stem}")
        results = run_pipeline(
            video=video,
            phases=sorted(args.phases, key=_sort),
            profile=args.profile,
            both_modes=args.both_modes,
            engine=args.engine,
            epochs=args.epochs,
            no_whisper=args.no_whisper,
            llm_provider=args.llm_provider,
            use_llm_text=args.llm_text,
        )
        summary[stem] = results

    # ── Riepilogo finale ─────────────────────────────────────────────── #
    elapsed_total = time.time() - t_start
    banner("RIEPILOGO")
    for stem, results in summary.items():
        print(f"  {BOLD}{stem}{RESET}")
        for phase, ok in sorted(
            results.items(), key=lambda x: PHASE_ORDER.index(x[0]) if x[0] in PHASE_ORDER else 99
        ):
            icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
            print(f"    Fase {phase}: {icon}")
    print(f"\n  Tempo totale: {elapsed_total:.1f}s")

    # Output files
    print(f"\n{BOLD}Output generati:{RESET}")
    out_dir = SRC_DIR / "outputs" / "audio"
    if out_dir.exists():
        for f in sorted(out_dir.iterdir()):
            if f.suffix == ".wav":
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  🔊 {f.name} ({size_mb:.1f} MB)")

    all_ok = all(ok for r in summary.values() for ok in r.values())
    if all_ok:
        print(f"\n{GREEN}{BOLD}✓ Pipeline completata con successo!{RESET}")
    else:
        print(f"\n{RED}{BOLD}✗ Alcune fasi hanno avuto errori.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
