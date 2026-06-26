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
            text = template.format(player=ev.get("player", "il giocatore"))
            script.append({
                "t": ev["t"],
                "text": text,
                "event_type": ev["type"],
                "importance": ev["importance"],
            })
        return script


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2 - Generazione testo telecronaca")
    parser.add_argument("--events", required=True)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    events_path = Path(args.events)
    events = load_json(events_path)

    script = ScriptGenerator(config.TEMPLATES).generate(events)
    out_path = config.SCRIPTS_DIR / events_path.name
    ensure_dir(out_path)
    save_json(script, out_path)

    logger.info("Generate %d battute -> %s", len(script), out_path)
    if script:
        logger.info("Anteprima: \"%s\"", script[0]["text"])
    logger.info("Fase 2 completata.")


if __name__ == "__main__":
    main()
