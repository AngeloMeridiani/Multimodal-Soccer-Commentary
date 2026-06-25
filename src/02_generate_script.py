"""
02_generate_script.py
====================
FASE 2 - Generazione del testo della telecronaca (NLP, basata su template).

Trasforma il log eventi (Fase 1) in una sequenza di battute testuali, scegliendo
un template per ogni evento e riempiendo i segnaposto (nome giocatore).

Scelta progettuale: si usano TEMPLATE, non un LLM. Per un PoC i template sono
affidabili (zero allucinazioni, riproducibili) e spostano il valore di ricerca
sulla prosodia, dove sta il vero contributo. L'LLM resta un'estensione futura
(vedi README).

Output: features/scripts/<nome_video>.json
        lista di battute: {t, text, event_type, importance}

Uso:
    python src/02_generate_script.py --events features/events/match1.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import config
from utils import ensure_dir, get_logger, load_json, save_json, set_seed

logger = get_logger("fase2_script")


class ScriptGenerator:
    """Genera battute di telecronaca da una lista di eventi, via template."""

    def __init__(self, templates: dict[str, list[str]]) -> None:
        self.templates = templates
        self._last_choice: dict[str, str] = {}  # per evitare ripetizioni consecutive

    def _pick_template(self, event_type: str) -> str:
        """Sceglie un template evitando, se possibile, di ripetere il precedente."""
        options = self.templates.get(event_type, ["{player}."])
        if len(options) > 1 and event_type in self._last_choice:
            options = [o for o in options if o != self._last_choice[event_type]] or options
        choice = random.choice(options)
        self._last_choice[event_type] = choice
        return choice

    def generate(self, events: list[dict]) -> list[dict]:
        script: list[dict] = []
        for event in events:
            template = self._pick_template(event["type"])
            text = template.format(player=event.get("player", "il giocatore"))
            script.append({
                "t": event["t"],
                "text": text,
                "event_type": event["type"],
                "importance": event["importance"],
            })
        return script


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2 - Generazione testo telecronaca")
    parser.add_argument("--events", required=True, help="JSON eventi della Fase 1.")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)  # rende riproducibile la scelta dei template

    events_path = Path(args.events)
    events = load_json(events_path)

    generator = ScriptGenerator(config.TEMPLATES)
    script = generator.generate(events)

    out_path = config.SCRIPTS_DIR / events_path.name
    ensure_dir(out_path)
    save_json(script, out_path)

    logger.info("Generate %d battute -> %s", len(script), out_path)
    if script:
        logger.info("Anteprima: \"%s\"", script[0]["text"])
    logger.info("Fase 2 completata.")


if __name__ == "__main__":
    main()
