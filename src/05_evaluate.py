"""
05_evaluate.py
==============
FASE 5 - Valutazione: studio A/B sugli ascoltatori + test statistico.

Confronta due tracce della STESSA partita:
  A) prosodia APPRESA (modello, Fase 3)      ->  python 04_synthesize.py --script ...
  B) prosodia a REGOLE (baseline)            ->  python 04_synthesize.py --script ... --rule-based

Due comandi:
  - make-study : prepara un foglio risposte (ascoltatori indicano A o B alla cieca).
  - analyze    : legge le risposte e calcola la preferenza + test binomiale
                 (H0: nessuna preferenza, p=0.5).

Uso:
    python 05_evaluate.py make-study --model outputs/audio/match1_model.wav \\
                                     --rule  outputs/audio/match1_rulebased.wav --raters 20
    python 05_evaluate.py analyze --responses outputs/study/responses.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import config
from utils import ensure_dir, get_logger, set_seed

logger = get_logger("fase5_valutazione")

RESPONSES_CSV = config.STUDY_DIR / "responses.csv"
MANIFEST_JSON = config.STUDY_DIR / "study_manifest.json"


def make_study(model_wav: Path, rule_wav: Path, n_raters: int) -> None:
    """Crea il manifest (con assegnazione cieca A/B casuale) e un CSV da compilare."""
    set_seed(config.RANDOM_SEED)
    if not model_wav.exists() or not rule_wav.exists():
        raise FileNotFoundError("Servono entrambe le tracce (model e rule-based).")

    ensure_dir(MANIFEST_JSON)
    rows = []
    for rid in range(1, n_raters + 1):
        # Per ogni ascoltatore randomizziamo quale traccia e' "A" (anti-bias d'ordine).
        model_is_a = random.random() < 0.5
        rows.append({
            "rater_id": rid,
            "track_A": str(model_wav if model_is_a else rule_wav),
            "track_B": str(rule_wav if model_is_a else model_wav),
            "model_label": "A" if model_is_a else "B",
        })

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(RESPONSES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rater_id", "preference", "model_label"])
        writer.writeheader()
        for r in rows:
            # 'preference' va compilato dall'ascoltatore con 'A' o 'B'.
            writer.writerow({"rater_id": r["rater_id"], "preference": "",
                             "model_label": r["model_label"]})

    logger.info("Manifest -> %s", MANIFEST_JSON)
    logger.info("Foglio risposte da compilare (colonna 'preference' = A/B) -> %s", RESPONSES_CSV)
    logger.info("Domanda agli ascoltatori: \"Quale telecronaca e' piu' coinvolgente, A o B?\"")


def analyze(responses_csv: Path) -> dict:
    """Legge le risposte e calcola preferenza per il modello + test binomiale."""
    with open(responses_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("preference", "").strip()]

    if not rows:
        raise ValueError("Nessuna risposta compilata nella colonna 'preference'.")

    n = len(rows)
    # Conta quante volte e' stata preferita la traccia del MODELLO (gestendo la
    # randomizzazione A/B: model_label dice quale lettera era il modello).
    model_pref = sum(1 for r in rows
                     if r["preference"].strip().upper() == r["model_label"].strip().upper())
    rule_pref = n - model_pref
    prop = model_pref / n

    try:
        from scipy.stats import binomtest
        p_value = binomtest(model_pref, n, 0.5, alternative="two-sided").pvalue
    except Exception:  # fallback senza scipy
        p_value = _binom_two_sided(model_pref, n, 0.5)

    result = {
        "n_raters": n,
        "model_preferred": model_pref,
        "rule_preferred": rule_pref,
        "model_preference_rate": round(prop, 3),
        "p_value": round(float(p_value), 4),
        "significant_at_0.05": bool(p_value < 0.05),
    }

    logger.info("=== Risultati studio A/B ===")
    logger.info("  Ascoltatori           : %d", n)
    logger.info("  Preferiscono il modello: %d (%.1f%%)", model_pref, prop * 100)
    logger.info("  Preferiscono le regole : %d", rule_pref)
    logger.info("  p-value (binomiale)    : %.4f", p_value)
    logger.info("  Significativo (a=0.05) : %s", result["significant_at_0.05"])

    out_path = config.STUDY_DIR / "study_results.json"
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Risultati -> %s", out_path)
    return result


def _binom_two_sided(k: int, n: int, p: float) -> float:
    """Test binomiale a due code senza scipy (somma delle code <= prob osservata)."""
    from math import comb
    probs = [comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    observed = probs[k]
    return float(min(1.0, sum(pr for pr in probs if pr <= observed + 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 5 - Valutazione A/B")
    sub = parser.add_subparsers(dest="command", required=True)

    p_make = sub.add_parser("make-study", help="Prepara lo studio A/B.")
    p_make.add_argument("--model", required=True, help="Traccia con prosodia appresa.")
    p_make.add_argument("--rule", required=True, help="Traccia con prosodia a regole.")
    p_make.add_argument("--raters", type=int, default=20)

    p_an = sub.add_parser("analyze", help="Analizza le risposte.")
    p_an.add_argument("--responses", default=str(RESPONSES_CSV))

    args = parser.parse_args()
    if args.command == "make-study":
        make_study(Path(args.model), Path(args.rule), args.raters)
    elif args.command == "analyze":
        analyze(Path(args.responses))
    logger.info("Fase 5 completata.")


if __name__ == "__main__":
    main()
