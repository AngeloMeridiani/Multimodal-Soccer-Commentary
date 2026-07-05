"""
test_prosody.py — diagnostica rapida del modello di prosodia (Fase 3).
=====================================================================
Mostra, per OGNI tipo di evento, cosa predice il MODELLO appreso
(prosody_mlp.pt + scaler) accanto ai valori a REGOLE: cosi' vedi
a colpo d'occhio se e quanto il modello ha imparato qualcosa di
diverso dalla baseline.

Con --say sintetizza UNA frase di prova in tre versioni nella
cartella outputs/audio/ , cosi' puoi ascoltare la differenza:
    _test_neutral.wav    voce TTS neutra (nessuna prosodia)
    _test_rulebased.wav  prosodia a REGOLE
    _test_model.wav      prosodia del MODELLO appreso

Mettilo nella ROOT del progetto (accanto a config.py / 04_synthesize.py).

Uso:
    python test_prosody.py
    python test_prosody.py --say "GOOOL! Ha segnato Messi!" --event goal
    python test_prosody.py --say "Si fa girare il pallone." --event idle
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

import config


def _load_module(filename: str, modname: str):
    """Importa un modulo il cui nome file inizia con una cifra (es. 04_*.py)."""
    spec = importlib.util.spec_from_file_location(modname, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostica prosodia (Fase 3)")
    ap.add_argument(
        "--say", type=str, default=None, help="Frase di prova da sintetizzare nelle 3 versioni."
    )
    ap.add_argument(
        "--event", type=str, default="goal", help="Tipo di evento per --say (default: goal)."
    )
    args = ap.parse_args()

    synth = _load_module("04_synthesize.py", "synth_mod")

    # Due predittori che condividono ESATTAMENTE il codice della Fase 4.
    pred_model = synth.ProsodyPredictor(force_rule_based=False)
    pred_rule = synth.ProsodyPredictor(force_rule_based=True)

    if pred_model.mode != "learned":
        print(
            "\n!!! ATTENZIONE: il MODELLO non e' stato caricato (modalita' =", pred_model.mode, ")."
        )
        print(
            "    Controlla che prosody_mlp.pt e prosody_scaler.joblib siano in:",
            config.PROSODY_MODEL_PATH.parent,
            "\n",
        )

    # ----- Tabella di confronto MODELLO vs REGOLE ------------------------- #
    print(
        f"\n{'evento':<14}{'imp':>5}   "
        f"{'rate (mod/reg)':>20}{'pitch (mod/reg)':>20}{'energy (mod/reg)':>20}"
    )
    print("-" * 100)
    for et in config.EVENT_TYPES:
        imp = config.EVENT_IMPORTANCE.get(et, 0.1)
        m = pred_model.predict(et, imp)
        r = pred_rule.predict(et, imp)
        print(
            f"{et:<14}{imp:>5.2f}   "
            f"{m['rate_factor']:>8.2f} / {r['rate_factor']:<8.2f}"
            f"{m['pitch_semitones']:>9.2f} / {r['pitch_semitones']:<8.2f}"
            f"{m['energy_gain']:>10.2f} / {r['energy_gain']:<8.2f}"
        )
    print("-" * 100)
    print(
        "Se le colonne mod/reg sono quasi identiche, e' NORMALE quando il "
        "training\ne' su dataset SINTETICO (derivato dalle regole). Con clip "
        "reali annotate\nle differenze diventano marcate.\n"
    )

    # ----- Sintesi di prova (opzionale) ----------------------------------- #
    if args.say:
        import soundfile as sf
        from utils import apply_prosody

        et = args.event
        imp = config.EVENT_IMPORTANCE.get(et, 0.5)
        out_dir = config.AUDIO_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        tts = synth.make_tts_engine(config.TTS_ENGINE)
        neutral, sr = tts.synth_neutral(args.say)

        pm = pred_model.predict(et, imp)
        pr = pred_rule.predict(et, imp)
        voiced_model = apply_prosody(
            neutral, sr, pm["rate_factor"], pm["pitch_semitones"], pm["energy_gain"]
        )
        voiced_rule = apply_prosody(
            neutral, sr, pr["rate_factor"], pr["pitch_semitones"], pr["energy_gain"]
        )

        sf.write(str(out_dir / "_test_neutral.wav"), neutral, sr)
        sf.write(str(out_dir / "_test_rulebased.wav"), voiced_rule, sr)
        sf.write(str(out_dir / "_test_model.wav"), voiced_model, sr)
        print(f"Evento '{et}' (imp={imp:.2f})")
        print(f"  modello -> {pm}")
        print(f"  regole  -> {pr}")
        print(f"Salvati in {out_dir}:")
        print("  _test_neutral.wav  _test_rulebased.wav  _test_model.wav\n")


if __name__ == "__main__":
    main()
