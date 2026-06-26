"""
02b_generate_llm.py
===================
FASE 2b - Generazione telecronaca con LLM (alternativa ai template).

I dati strutturati di ogni evento vengono passati a un LLM che genera la battuta
come un telecronista italiano esaltato. Provider: ollama (locale), openai, anthropic.

Output: features/scripts/<nome_video>_llm.json

Uso:
    python 02b_generate_llm.py --events features/events/match1_enriched.json
    python 02b_generate_llm.py --events features/events/match1.json --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import config
from utils import ensure_dir, get_logger, load_json, save_json, set_seed

logger = get_logger("fase2b_llm")


class LLMProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, base_url: str) -> None:
        self.model, self.base_url = model, base_url.rstrip("/")
        logger.info("Provider: Ollama (model=%s)", model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request, urllib.error
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as exc:
            logger.error("Errore Ollama: %s ('ollama serve' attivo?)", exc)
            raise


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str) -> None:
        self.model, self.api_key = model, api_key
        logger.info("Provider: OpenAI (model=%s)", model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "temperature": config.LLM_TEMPERATURE, "max_tokens": 200,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str) -> None:
        self.model, self.api_key = model, api_key
        logger.info("Provider: Anthropic (model=%s)", model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model, "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": config.LLM_TEMPERATURE, "max_tokens": 200,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["content"][0]["text"].strip()


def create_provider(provider_name: str, model: str | None = None) -> LLMProvider:
    cfg = config.LLM_CONFIG.get(provider_name)
    if not cfg:
        raise ValueError(f"Provider sconosciuto: {provider_name}")
    actual_model = model or cfg["model"]
    if provider_name == "ollama":
        return OllamaProvider(actual_model, cfg["base_url"])
    api_key = os.environ.get(cfg.get("api_key_env", ""), "")
    if not api_key:
        raise EnvironmentError(f"API key mancante: esporta {cfg.get('api_key_env')}.")
    if provider_name == "openai":
        return OpenAIProvider(actual_model, api_key)
    if provider_name == "anthropic":
        return AnthropicProvider(actual_model, api_key)
    raise ValueError(f"Provider non supportato: {provider_name}")


def build_event_prompt(event: dict, idx: int, total: int) -> str:
    data = {
        "evento_numero": f"{idx + 1}/{total}",
        "timestamp_s": event.get("t", 0),
        "tipo_evento": event.get("type", "unknown"),
        "giocatore": event.get("player", "sconosciuto"),
        "importanza": event.get("importance", 0.5),
    }
    for src, dst in [("ball_zone", "zona_palla"), ("ball_speed", "velocita_palla"),
                     ("players_nearby", "giocatori_vicini"),
                     ("crowd_excitement", "eccitazione_tifo"),
                     ("original_commentary", "telecronaca_originale")]:
        if event.get(src):
            data[dst] = event[src]
    if event.get("score"):
        data["punteggio"] = f"{event['score'][0]}-{event['score'][1]}"

    return (
        "Genera UNA battuta di telecronaca per questo evento.\n"
        f"Dati (JSON):\n```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```\n\n"
        "Regole: 1-2 frasi; tono proporzionato all'importanza; se il tifo e' alto, "
        "esalta; se c'e' la telecronaca originale usane nomi/contesto; rispondi SOLO "
        "con la battuta."
    )


class LLMScriptGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.system_prompt = config.LLM_SYSTEM_PROMPT

    def generate(self, events: list[dict], delay: float = 0.5) -> list[dict]:
        script: list[dict] = []
        total = len(events)
        for i, event in enumerate(events):
            try:
                text = self.provider.generate(self.system_prompt,
                                              build_event_prompt(event, i, total))
            except Exception as exc:
                logger.warning("Errore LLM evento %d: %s. Uso fallback template.", i + 1, exc)
                text = self._fallback(event)
            script.append({
                "t": event.get("t", 0), "text": text,
                "event_type": event.get("type", "idle"),
                "importance": event.get("importance", 0.1),
                "crowd_excitement": event.get("crowd_excitement", "low"),
                "source": "llm",
            })
            if i < total - 1:
                time.sleep(delay)
            if (i + 1) % 10 == 0:
                logger.info("Generate %d/%d battute.", i + 1, total)
        return script

    @staticmethod
    def _fallback(event: dict) -> str:
        import random
        templates = config.TEMPLATES.get(event.get("type", "idle"), ["Azione in corso."])
        return random.choice(templates).format(player=event.get("player", "il giocatore"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2b - Generazione telecronaca con LLM")
    parser.add_argument("--events", required=True)
    parser.add_argument("--provider", choices=["ollama", "openai", "anthropic"],
                        default=config.LLM_PROVIDER)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    events_path = Path(args.events)
    events = load_json(events_path)
    logger.info("Caricati %d eventi da %s", len(events), events_path)

    provider = create_provider(args.provider, args.model)
    script = LLMScriptGenerator(provider).generate(events, delay=args.delay)

    out_name = events_path.stem.replace("_enriched", "").replace("_audio", "")
    out_path = config.SCRIPTS_DIR / f"{out_name}_llm.json"
    ensure_dir(out_path)
    save_json(script, out_path)
    logger.info("Generate %d battute LLM -> %s", len(script), out_path)
    if script:
        logger.info("Anteprima: \"%s\"", script[0]["text"])
    logger.info("Fase 2b completata.")


if __name__ == "__main__":
    main()
