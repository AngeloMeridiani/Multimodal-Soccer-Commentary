"""
05_evaluate.py
==============
FASE 5 - Valutazione: studio sugli ascoltatori (Mean Opinion Score).

E' la parte che rende il progetto RICERCA e non solo una demo: confronta le
versioni 'flat' (baseline) e 'learned' (contributo) - e opzionalmente 'rule' -
chiedendo a degli ascoltatori di votarle, e verifica se la prosodia appresa
migliora naturalezza e coinvolgimento.

Due comandi:
    prepare : crea i materiali del test - coppie randomizzate e anonimizzate
              (A/B) + un foglio risposte CSV vuoto da distribuire.
    analyze : legge il foglio compilato e calcola medie, differenza e un test
              statistico appaiato (flat vs learned).

Uso:
    python src/05_evaluate.py prepare --names match1 match2
    python src/05_evaluate.py analyze --responses outputs/study/responses.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import config
from utils import ensure_dir, get_logger, set_seed

logger = get_logger("fase5_valutazione")

CONDITIONS = ["flat", "learned"]   # aggiungi "rule" per il confronto a 3 condizioni


def prepare(names: list[str]) -> None:
    """
    Per ogni telecronaca, crea una coppia A/B anonimizzata (ordine casuale) e
    registra la chiave di mappatura A/B -> condizione (nascosta agli ascoltatori).
    """
    set_seed(config.RANDOM_SEED)
    ensure_dir(config.STUDY_DIR / "_")
    key, response_rows = [], []

    for name in names:
        # Verifica che le versioni esistano
        paths = {c: config.AUDIO_OUT_DIR / f"{name}_{c}.wav" for c in CONDITIONS}
        missing = [c for c, p in paths.items() if not p.exists()]
        if missing:
            logger.warning("Salto '%s': mancano le versioni %s (esegui la Fase 4).",
                           name, missing)
            continue

        # Randomizza quale condizione e' "A" e quale "B"
        conditions = CONDITIONS[:]
        random.shuffle(conditions)
        for label, cond in zip(["A", "B"], conditions):
            dst = config.STUDY_DIR / f"{name}_{label}.wav"
            shutil.copy(paths[cond], dst)
            key.append({"item": name, "label": label, "condition": cond})

        response_rows.append({"item": name, "naturalezza_preferita": "",
                              "coinvolgimento_preferito": ""})

    # Chiave segreta (NON darla agli ascoltatori)
    with open(config.STUDY_DIR / "key.json", "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)

    # Foglio risposte: per ogni item, l'ascoltatore indica se preferisce A o B
    sheet = config.STUDY_DIR / "responses_template.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "naturalezza_preferita",
                                               "coinvolgimento_preferito"])
        writer.writeheader()
        writer.writerows(response_rows)

    logger.info("Materiali pronti in %s", config.STUDY_DIR)
    logger.info("Distribuisci i file *_A.wav / *_B.wav e 'responses_template.csv'.")
    logger.info("NON condividere 'key.json' con gli ascoltatori.")


def analyze(responses_path: Path) -> None:
    """
    Converte le preferenze A/B in preferenze per condizione usando key.json e
    calcola la percentuale di volte in cui 'learned' batte 'flat', con un test
    binomiale semplice.
    """
    import pandas as pd
    from scipy import stats

    key = pd.read_json(config.STUDY_DIR / "key.json")
    responses = pd.read_csv(responses_path)

    def resolve(row, column):
        """Dal voto A/B ricava la condizione preferita per quell'item."""
        pref_label = str(row[column]).strip().upper()
        match = key[(key["item"] == row["item"]) & (key["label"] == pref_label)]
        return match["condition"].iloc[0] if len(match) else None

    wins = {"naturalezza": [], "coinvolgimento": []}
    for _, row in responses.iterrows():
        for metric, col in [("naturalezza", "naturalezza_preferita"),
                            ("coinvolgimento", "coinvolgimento_preferito")]:
            cond = resolve(row, col)
            if cond in CONDITIONS:
                wins[metric].append(1 if cond == "learned" else 0)

    logger.info("=== Risultati studio (preferenza per 'learned' vs 'flat') ===")
    for metric, votes in wins.items():
        if not votes:
            logger.info("%-15s: nessun voto valido.", metric)
            continue
        n, k = len(votes), sum(votes)
        pref = k / n
        # Test binomiale: 'learned' e' preferito piu' del caso (50%)?
        p_value = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
        logger.info("%-15s: 'learned' preferito %d/%d (%.0f%%), p=%.4f",
                    metric, k, n, pref * 100, p_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 5 - Studio sugli ascoltatori")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Crea i materiali A/B e il foglio risposte.")
    p_prep.add_argument("--names", nargs="+", required=True,
                        help="Nomi delle telecronache (senza estensione/condizione).")

    p_an = sub.add_parser("analyze", help="Analizza il foglio risposte compilato.")
    p_an.add_argument("--responses", required=True, help="CSV delle risposte raccolte.")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.names)
    else:
        analyze(Path(args.responses))


if __name__ == "__main__":
    main()
