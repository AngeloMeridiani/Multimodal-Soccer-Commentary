"""
02_generate_script.py
====================
PHASE 2 - Commentary text generation (NLP, template-based).

Turns the event log into text lines by picking a template per event and filling
in the placeholders. Templates are reliable (zero hallucinations, reproducible)
and shift the research value onto the prosody (Phase 3). The LLM is the
alternative in Phase 2b.

Output: features/scripts/<video_name>.json   (list: {t, text, event_type, importance})

Usage:
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
        """
        Initialize the script generator with a dictionary of templates for each event type.
        """
        self.templates = templates
        self._last: dict[str, str] = {}

    def _pick(self, event_type: str) -> str:
        """
        Randomly select a template for a given event type.
        Avoids repeating the exact same template consecutively if alternatives exist.
        """
        options = self.templates.get(event_type, ["{player}."])
        if len(options) > 1 and event_type in self._last:
            options = [o for o in options if o != self._last[event_type]] or options
        choice = random.choice(options)
        self._last[event_type] = choice
        return choice

    def generate(self, events: list[dict]) -> list[dict]:
        """
        Generate a text script from a list of events by populating templates.
        Resolves player names using possession context when available.
        """
        script: list[dict] = []
        for ev in events:
            template = self._pick(ev["type"])
            # Pick the right name based on the visual possession.
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
                    # REAL crowd excitement (Phase 1c), if present: Phase 4 uses
                    # it as a prosody feature. 0.5 (neutral) when the event was
                    # not enriched with the audio.
                    "audio_energy": ev.get("audio_energy", 0.5),
                }
            )
        return script


def main() -> None:
    """
    Main entry point for Phase 2.
    Loads events, generates a textual script using templates, and saves it.
    """
    parser = argparse.ArgumentParser(description="Fase 2 - Generazione testo telecronaca")
    parser.add_argument("--events", required=True)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    events_path = Path(args.events)
    events = load_json(events_path)

    script = ScriptGenerator(config.get_templates()).generate(events)
    # Script name = <stem>.json, WITHOUT the event-file suffixes
    # (_enriched/_audio): this way it matches what Phase 4 expects
    # (run_pipeline looks for features/scripts/<stem>.json). Same normalization
    # Phase 2b already does. Without it, synthesis could not find the script.
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
