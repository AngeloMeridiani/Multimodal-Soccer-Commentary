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
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SRC_DIR = Path(__file__).resolve().parent


def banner(text: str) -> None:
    width = 64
    print(f"\n{CYAN}{BOLD}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{RESET}\n")


def step_header(phase: int, desc: str) -> None:
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
        str(p) for p in gameplay_dir.iterdir()
        if p.suffix.lower() in exts and not p.name.startswith(".")
    )
    return videos


def run_pipeline(
    video: str,
    phases: list[int],
    profile: str,
    both_modes: bool,
    engine: str,
    epochs: int,
) -> dict[int, bool]:
    """Esegue le fasi selezionate per un singolo video."""
    stem = get_stem(video)
    events_json  = f"features/events/{stem}.json"
    script_json  = f"features/scripts/{stem}.json"
    results: dict[int, bool] = {}

    # ── FASE 1: Estrazione eventi ────────────────────────────────────── #
    if 1 in phases:
        step_header(1, "Estrazione eventi (OCR HUD)")
        cmd = [sys.executable, "01_extract_events.py", "--video", video]
        if profile:
            cmd += ["--profile", profile]
        results[1] = run(cmd, "Fase 1")
        if not results[1]:
            return results

    # ── FASE 2: Generazione testo ────────────────────────────────────── #
    if 2 in phases:
        step_header(2, "Generazione testo telecronaca")
        results[2] = run(
            [sys.executable, "02_generate_script.py", "--events", events_json],
            "Fase 2",
        )
        if not results[2]:
            return results

    # ── FASE 3: Training prosodia ────────────────────────────────────── #
    if 3 in phases:
        step_header(3, "Addestramento modello prosodia")
        cmd = [sys.executable, "03_train_prosody.py", "--synthetic"]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        results[3] = run(cmd, "Fase 3")
        if not results[3]:
            return results

    # ── FASE 4: Sintesi audio ────────────────────────────────────────── #
    if 4 in phases:
        step_header(4, "Sintesi audio")
        # Modalità learned (default)
        cmd = [sys.executable, "04_synthesize.py", "--script", script_json]
        if engine:
            cmd += ["--engine", engine]
        results[4] = run(cmd, "Fase 4 (learned)")

        # Anche rule-based se richiesto
        if both_modes:
            cmd_rule = [
                sys.executable, "04_synthesize.py",
                "--script", script_json, "--rule-based",
            ]
            if engine:
                cmd_rule += ["--engine", engine]
            ok = run(cmd_rule, "Fase 4 (rule-based)")
            results[4] = results[4] and ok

    # ── FASE 5: Valutazione A/B ──────────────────────────────────────── #
    if 5 in phases:
        step_header(5, "Preparazione studio A/B")
        results[5] = run(
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
        "--video", type=str, default=None,
        help="Percorso del video di gameplay.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Esegui la pipeline su TUTTI i video in data/raw/gameplay/.",
    )
    parser.add_argument(
        "--phases", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Fasi da eseguire (default: 1 2 3 4). Usa 5 per lo studio A/B.",
    )
    parser.add_argument(
        "--profile", type=str, default="auto",
        help="Profilo HUD (default: auto). Vedi config.HUD_PROFILES.",
    )
    parser.add_argument(
        "--both-modes", action="store_true",
        help="Genera sia learned che rule-based (per confronto/studio).",
    )
    parser.add_argument(
        "--engine", type=str, default=None, choices=["gtts", "pyttsx3", "coqui"],
        help="Motore TTS (default: da config.py).",
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="Numero epoche per Fase 3 (default: da config.py).",
    )

    args = parser.parse_args()

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
    print(f"  Fasi:   {sorted(args.phases)}")
    print(f"  Profilo: {args.profile}")
    if args.both_modes:
        print(f"  Modalità: learned + rule-based")
    print()

    t_start = time.time()
    summary: dict[str, dict[int, bool]] = {}

    for i, video in enumerate(videos, 1):
        stem = get_stem(video)
        banner(f"[{i}/{len(videos)}]  {stem}")
        results = run_pipeline(
            video=video,
            phases=sorted(args.phases),
            profile=args.profile,
            both_modes=args.both_modes,
            engine=args.engine,
            epochs=args.epochs,
        )
        summary[stem] = results

    # ── Riepilogo finale ─────────────────────────────────────────────── #
    elapsed_total = time.time() - t_start
    banner("RIEPILOGO")
    for stem, results in summary.items():
        print(f"  {BOLD}{stem}{RESET}")
        for phase, ok in sorted(results.items()):
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
