"""
02_generate_script.py
====================
FASE 2 - Generazione del testo della telecronaca (NLP, basata su template).

Trasforma il log eventi in battute testuali, scegliendo un template per evento e
riempiendo i segnaposto. I template sono affidabili (zero allucinazioni,
riproducibili) e spostano il valore di ricerca sulla prosodia (Fase 3).
L'LLM e' l'alternativa nella Fase 2b.

Output: features/scripts/<nome_video>.json   (lista: {t, text, event_type, importance})

Uso:
    python 02_generate_script.py --events features/events/match1.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import config
from utils import ensure_dir, get_logger, load_json, save_json, set_seed

logger = get_logger("fase2_script")


class ScriptGenerator:
    def __init__(self, templates: dict[str, list[str]]) -> None:
        self.templates = templates
        self._last: dict[str, str] = {}

    def _pick(self, event_type: str) -> str:
        options = self.templates.get(event_type, ["{player}."])
        if len(options) > 1 and event_type in self._last:
            options = [o for o in options if o != self._last[event_type]] or options
        choice = random.choice(options)
        self._last[event_type] = choice
        return choice

    def generate(self, events: list[dict]) -> list[dict]:
        script: list[dict] = []
        for ev in events:
            template = self._pick(ev["type"])
            # Scegli il nome giusto in base al possesso visivo.
            poss = ev.get("possession")
            if poss == "home" and ev.get("player_home"):
                player = ev["player_home"]
            elif poss == "away" and ev.get("player_away"):
                player = ev["player_away"]
            else:
                player = ev.get("player", config.get_unknown_player())
            text = template.format(player=player)
            script.append(
                {
                    "t": ev["t"],
                    "text": text,
                    "event_type": ev["type"],
                    "importance": ev["importance"],
                    # Eccitazione REALE del pubblico (Fase 1c), se presente:
                    # la Fase 4 la usa come feature della prosodia. 0.5 (neutro)
                    # se l'evento non e' stato arricchito con l'audio.
                    "audio_energy": ev.get("audio_energy", 0.5),
                }
            )
        return script


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2 - Generazione testo telecronaca")
    parser.add_argument("--events", required=True)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    events_path = Path(args.events)
    events = load_json(events_path)

    script = ScriptGenerator(config.get_templates()).generate(events)
    # Nome dello script = <stem>.json, SENZA i suffissi del file eventi
    # (_enriched/_audio): cosi' combacia con cio' che la Fase 4 si aspetta
    # (run_pipeline cerca features/scripts/<stem>.json). Stessa normalizzazione
    # che fa gia' la Fase 2b. Senza, la sintesi non trovava lo script.
    stem = events_path.stem.replace("_enriched", "").replace("_audio", "")
    out_path = config.SCRIPTS_DIR / f"{stem}.json"
    ensure_dir(out_path)
    save_json(script, out_path)

    logger.info("Generate %d battute -> %s", len(script), out_path)
    if script:
        logger.info('Anteprima: "%s"', script[0]["text"])
    logger.info("Fase 2 completata.")


if __name__ == "__main__":
    main()
