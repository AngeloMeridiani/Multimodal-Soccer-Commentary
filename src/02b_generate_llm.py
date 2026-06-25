"""
02b_generate_llm.py
===================
FASE 2b - Generazione telecronaca con LLM (alternativa ai template).

Invece di usare template fissi (Fase 2), qui i dati strutturati degli eventi
vengono passati a un Large Language Model che genera battute di telecronaca
naturali, colorite e contestualizzate.

Supporta tre provider:
  - ollama   : modelli locali gratuiti (LLaMA 3, Mistral, etc.) — RACCOMANDATO
  - openai   : GPT-4 / GPT-3.5 via API
  - anthropic: Claude via API

Ogni evento viene trasformato in un prompt JSON, e l'LLM genera la telecronaca
come se fosse un commentatore sportivo italiano esaltato.

Output: features/scripts/<nome_video>_llm.json

Uso:
    python src/02b_generate_llm.py --events features/events/match1_enriched.json
    python src/02b_generate_llm.py --events features/events/match1.json --provider openai
    python src/02b_generate_llm.py --events features/events/match1.json --provider ollama --model llama3
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


# --------------------------------------------------------------------------- #
# Provider LLM                                                                 #
# --------------------------------------------------------------------------- #
class LLMProvider:
    """Classe base per i provider LLM."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """
    Provider per Ollama (modelli locali).

    Richiede Ollama installato e in esecuzione:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull llama3
        ollama serve  (in un terminale separato)
    """

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        logger.info("Provider: Ollama (model=%s, url=%s)", model, base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/api/chat"
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE},
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as exc:
            logger.error("Errore Ollama: %s. Assicurati che 'ollama serve' sia in esecuzione.", exc)
            raise


class OpenAIProvider(LLMProvider):
    """Provider per OpenAI GPT-4 / GPT-3.5."""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key
        logger.info("Provider: OpenAI (model=%s)", model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request

        url = "https://api.openai.com/v1/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": 200,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()


class AnthropicProvider(LLMProvider):
    """Provider per Anthropic Claude."""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key
        logger.info("Provider: Anthropic (model=%s)", model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request

        url = "https://api.anthropic.com/v1/messages"
        payload = json.dumps({
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": 200,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"].strip()


def create_provider(provider_name: str, model: str | None = None) -> LLMProvider:
    """Factory per creare il provider LLM appropriato."""
    cfg = config.LLM_CONFIG.get(provider_name)
    if not cfg:
        raise ValueError(f"Provider sconosciuto: {provider_name}. "
                         f"Opzioni: {list(config.LLM_CONFIG.keys())}")

    actual_model = model or cfg["model"]

    if provider_name == "ollama":
        return OllamaProvider(actual_model, cfg["base_url"])

    # Per provider cloud, serve la API key
    api_key_env = cfg.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise EnvironmentError(
            f"API key non trovata. Imposta la variabile d'ambiente '{api_key_env}'.\n"
            f"Esempio: export {api_key_env}='sk-...'"
        )

    if provider_name == "openai":
        return OpenAIProvider(actual_model, api_key)
    elif provider_name == "anthropic":
        return AnthropicProvider(actual_model, api_key)

    raise ValueError(f"Provider non supportato: {provider_name}")


# --------------------------------------------------------------------------- #
# Costruzione del prompt                                                       #
# --------------------------------------------------------------------------- #
def build_event_prompt(event: dict, event_index: int, total_events: int) -> str:
    """
    Costruisce il prompt utente per un singolo evento.
    Include tutti i dati disponibili (visivi, uditivi, OCR).
    """
    # Filtra i campi piu' utili per il telecronista
    prompt_data = {
        "evento_numero": f"{event_index + 1}/{total_events}",
        "timestamp_s": event.get("t", 0),
        "tipo_evento": event.get("type", "unknown"),
        "giocatore": event.get("player", "sconosciuto"),
        "importanza": event.get("importance", 0.5),
    }

    # Aggiungi dati visivi se disponibili
    if "ball_zone" in event:
        prompt_data["zona_palla"] = event["ball_zone"]
    if "ball_speed" in event:
        prompt_data["velocita_palla"] = event["ball_speed"]
    if "players_nearby" in event:
        prompt_data["giocatori_vicini"] = event["players_nearby"]
    if "score" in event and event["score"]:
        prompt_data["punteggio"] = f"{event['score'][0]}-{event['score'][1]}"

    # Aggiungi dati audio se disponibili
    if "crowd_excitement" in event:
        prompt_data["eccitazione_tifo"] = event["crowd_excitement"]
    if "crowd_score" in event:
        prompt_data["livello_tifo_numerico"] = event["crowd_score"]
    if "original_commentary" in event and event["original_commentary"]:
        prompt_data["telecronaca_originale"] = event["original_commentary"]

    prompt = (
        f"Genera UNA battuta di telecronaca per questo evento di gioco.\n"
        f"Dati evento (JSON):\n"
        f"```json\n{json.dumps(prompt_data, ensure_ascii=False, indent=2)}\n```\n\n"
        f"Regole:\n"
        f"- Una o due frasi al massimo.\n"
        f"- Adatta il tono all'importanza: importanza alta = voce esaltata, "
        f"importanza bassa = tono calmo.\n"
        f"- Se c'e' il tifo alto, esalta di piu' il commento.\n"
        f"- Se c'e' la telecronaca originale, usa i nomi/contesto che contiene.\n"
        f"- Rispondi SOLO con la battuta, senza spiegazioni o meta-commenti."
    )

    return prompt


# --------------------------------------------------------------------------- #
# Generazione dello script                                                     #
# --------------------------------------------------------------------------- #
class LLMScriptGenerator:
    """Genera l'intero script di telecronaca usando un LLM, evento per evento."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.system_prompt = config.LLM_SYSTEM_PROMPT

    def generate(self, events: list[dict], delay_between_calls: float = 0.5) -> list[dict]:
        """
        Genera una battuta per ogni evento.

        `delay_between_calls`: pausa tra le chiamate API per evitare rate limiting.
        """
        script: list[dict] = []
        total = len(events)

        for i, event in enumerate(events):
            user_prompt = build_event_prompt(event, i, total)

            try:
                text = self.provider.generate(self.system_prompt, user_prompt)
            except Exception as exc:
                logger.warning("Errore LLM per evento %d/%d: %s. Uso fallback template.", i + 1, total, exc)
                text = self._fallback_template(event)

            # Classificazione automatica dell'intensita' emotiva dal testo generato
            emotion_intensity = self._estimate_emotion(text, event)

            script.append({
                "t": event.get("t", 0),
                "text": text,
                "event_type": event.get("type", "idle"),
                "importance": event.get("importance", 0.1),
                "emotion_intensity": emotion_intensity,
                "crowd_excitement": event.get("crowd_excitement", "low"),
                "source": "llm",
            })

            if i < total - 1:
                time.sleep(delay_between_calls)

            if (i + 1) % 10 == 0:
                logger.info("Generati %d/%d battute.", i + 1, total)

        return script

    @staticmethod
    def _fallback_template(event: dict) -> str:
        """Template di fallback se l'LLM fallisce."""
        templates = config.TEMPLATES.get(event.get("type", "idle"), ["Azione in corso."])
        import random
        template = random.choice(templates)
        return template.format(player=event.get("player", "il giocatore"))

    @staticmethod
    def _estimate_emotion(text: str, event: dict) -> str:
        """
        Stima l'intensita' emotiva di una battuta basandosi su euristiche semplici.
        Per una stima piu' precisa, si potrebbe usare un modello di sentiment analysis.
        """
        # Conta indicatori di eccitazione nel testo
        exclamations = text.count("!")
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        importance = event.get("importance", 0.5)

        score = (
            0.40 * importance +
            0.30 * min(exclamations / 3.0, 1.0) +
            0.30 * min(uppercase_ratio * 3, 1.0)
        )

        if score >= 0.75:
            return "very_high"
        elif score >= 0.50:
            return "high"
        elif score >= 0.25:
            return "medium"
        return "low"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2b - Generazione telecronaca con LLM")
    parser.add_argument("--events", required=True, help="JSON eventi (Fase 1/1b/1c).")
    parser.add_argument("--provider", choices=["ollama", "openai", "anthropic"],
                        default=config.LLM_PROVIDER,
                        help="Provider LLM da usare.")
    parser.add_argument("--model", type=str, default=None,
                        help="Modello specifico (sovrascrive config).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Pausa tra le chiamate LLM (secondi).")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    events_path = Path(args.events)
    events = load_json(events_path)
    logger.info("Caricati %d eventi da %s", len(events), events_path)

    # Crea il provider LLM
    provider = create_provider(args.provider, args.model)
    generator = LLMScriptGenerator(provider)

    # Genera lo script
    script = generator.generate(events, delay_between_calls=args.delay)

    # Salva
    out_name = events_path.stem.replace("_enriched", "").replace("_audio", "")
    out_path = config.SCRIPTS_DIR / f"{out_name}_llm.json"
    ensure_dir(out_path)
    save_json(script, out_path)

    logger.info("Generate %d battute LLM -> %s", len(script), out_path)
    if script:
        logger.info("Anteprima prima battuta: \"%s\"", script[0]["text"])
        logger.info("Anteprima ultima battuta: \"%s\"", script[-1]["text"])

    # Statistiche emozioni
    emotions = [s["emotion_intensity"] for s in script]
    emotion_dist = {e: emotions.count(e) for e in set(emotions)}
    logger.info("Distribuzione emozioni: %s", emotion_dist)
    logger.info("Fase 2b completata.")


if __name__ == "__main__":
    main()
 
