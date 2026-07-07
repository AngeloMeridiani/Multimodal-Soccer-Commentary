"""
02b_generate_llm.py
===================
FASE 2b - Generazione telecronaca con LLM (alternativa ai template).

I dati strutturati di ogni evento vengono passati a un LLM che genera la battuta
come un telecronista italiano esaltato. Provider: ollama (locale), openai, anthropic.

Ogni battuta generata passa da un VALIDATORE anti-allucinazioni: se cita un
giocatore della rosa che non c'entra con l'evento (nome inventato dall'LLM),
viene scartata e sostituita dal template equivalente, che non puo' allucinare.

Output: features/scripts/<nome_video>_llm.json

Uso:
    python 02b_generate_llm.py --events features/events/match1_enriched.json
    python 02b_generate_llm.py --events features/events/match1.json --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from difflib import SequenceMatcher
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

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": config.LLM_TEMPERATURE},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as exc:
            logger.error("Errore Ollama: %s ('ollama serve' attivo?)", exc)
            raise


class OpenAIProvider(LLMProvider):
    """Client per l'API 'chat completions' stile OpenAI. base_url configurabile:
    stesso client serve OpenAI vero e qualunque endpoint compatibile (Groq,
    LM Studio, Gemini...), cambia solo l'URL e la chiave."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        self.model, self.api_key = model, api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        logger.info("Provider: OpenAI-compatibile (model=%s, url=%s)", model, self.base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": 200,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str) -> None:
        self.model, self.api_key = model, api_key
        logger.info("Provider: Anthropic (model=%s)", model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": 200,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
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
    if provider_name in ("openai", "groq"):
        # Groq espone un endpoint OpenAI-compatibile: stessa classe, solo
        # base_url e chiave diversi (letti da LLM_CONFIG).
        return OpenAIProvider(actual_model, api_key, base_url=cfg.get("base_url"))
    if provider_name == "anthropic":
        return AnthropicProvider(actual_model, api_key)
    raise ValueError(f"Provider non supportato: {provider_name}")


def build_event_prompt(
    event: dict,
    idx: int,
    total: int,
    recent: list[str] | None = None,
    next_events: list[dict] | None = None,
) -> str:
    data = {
        "evento_numero" if config.LANGUAGE == "it" else "event_number": f"{idx + 1}/{total}",
        "timestamp_s": event.get("t", 0),
        "tipo_evento" if config.LANGUAGE == "it" else "event_type": event.get("type", "unknown"),
        "giocatore" if config.LANGUAGE == "it" else "player": event.get("player", config.get_unknown_player()),
        "importanza" if config.LANGUAGE == "it" else "importance": event.get("importance", 0.5),
    }
    field_map_it = [
        ("ball_zone", "zona_palla"),
        ("ball_speed", "velocita_palla"),
        ("players_nearby", "giocatori_vicini"),
        ("audio_energy_level", "eccitazione_tifo"),
        ("original_commentary", "telecronaca_originale"),
    ]
    field_map_en = [
        ("ball_zone", "ball_zone"),
        ("ball_speed", "ball_speed"),
        ("players_nearby", "players_nearby"),
        ("audio_energy_level", "crowd_excitement"),
        ("original_commentary", "original_commentary"),
    ]
    field_map = field_map_it if config.LANGUAGE == "it" else field_map_en
    for src, dst in field_map:
        if event.get(src):
            data[dst] = event[src]
    if event.get("score"):
        score_key = "punteggio" if config.LANGUAGE == "it" else "score"
        data[score_key] = f"{event['score'][0]}-{event['score'][1]}"

    evt = event.get("type", "pass")
    imp = float(event.get("importance", 0.5))

    # Verbi specifici per tipo di evento, così il modello non usa verbi da passaggio per un gol.
    VERB_HINTS_IT: dict[str, str] = {
        "goal": "USA SOLO verbi da gol: segna, insacca, trafigge, batte il portiere, la mette dentro. MAI 'filtra', 'lancia', 'verticalizza'. NON descrivere dove va: il modello non vede la traiettoria.",
        "save": "USA SOLO verbi da parata: para, dice di no, vola, blocca, devia, respinge.",
        "shot_on_goal": "USA verbi da tiro: calcia, conclude, ci prova, tenta la conclusione, scarica il destro/sinistro.",
        "shot_off": "REGOLA SPECIALE ASSOLUTA: Rispondi SOLO con una di queste 3 frasi: 'Tiro sul fondo!', 'Conclusione fuori!', 'A lato!'. NON usare nessun'altra parola.",
        "foul": "Fallo: viene atterrato, steso, trattenuto, il direttore di gara fischia.",
        "corner": "Corner: calcio d'angolo, batte il corner, cross in mezzo.",
        "free_kick": "Punizione: si incarica, prova la punizione diretta.",
        "turnover": "Cambio di possesso: usa verbi come 'recupera il possesso', 'riconquista', 'intercetta', 'cambio di fronte'. MAI usare 'smarca', verbi di passaggio o la parola 'vantaggio'.",
        "carry": "Conduzione: usa SINONIMI SEMPRE DIVERSI (es. conduce, avanza, si fa largo, sale, accelera, si invola). NON ripetere MAI il verbo della battuta precedente.",
    }
    VERB_HINTS_EN: dict[str, str] = {
        "goal": "Use ONLY goal verbs: scores, finishes, slots home, buries it, fires in. NEVER 'passes', 'launches'. Do NOT describe where it goes.",
        "save": "Use ONLY save verbs: saves, denies, blocks, tips over, pushes away, parries.",
        "shot_on_goal": "Use shot verbs: shoots, strikes, tries his luck, fires, lets fly.",
        "shot_off": "ABSOLUTE RULE: Respond ONLY with one of these 3 phrases: 'Off target!', 'Wide of the mark!', 'Just misses!'. Do NOT use any other words.",
        "foul": "Foul: brought down, taken out, held back, the referee blows.",
        "corner": "Corner: corner kick, takes the corner, cross into the box.",
        "free_kick": "Free kick: steps up, tries a direct free kick.",
        "turnover": "Change of possession: e.g. 'regains possession', 'wins it back', 'intercepts', 'change of play'. NEVER use passing verbs or the word 'advantage'.",
        "carry": "Carrying: use ALWAYS DIFFERENT synonyms (e.g. drives, surges, strides, advances, pushes on, bursts forward). NEVER repeat the verb from the previous line.",
    }
    verb_hints = VERB_HINTS_IT if config.LANGUAGE == "it" else VERB_HINTS_EN
    verb_hint = verb_hints.get(evt, "")

    if config.LANGUAGE == "it":
        if imp < 0.3:
            length_rule = "BREVISSIMA: 2-4 parole. Solo nome+verbo. Es: 'Kane scarica.', 'Bel tocco di Doku!', 'Si allarga Musiala.'. MAI 'pallone' o 'palla'."
        elif imp < 0.6:
            length_rule = "BREVE: massimo 10 parole. Una sola frase corta."
        else:
            length_rule = "ESALTATA: 1-2 frasi, max 20 parole. Momento clou, urla il gol!"
    else:
        if imp < 0.3:
            length_rule = "VERY SHORT: 2-4 words. Just name+verb. E.g.: 'Kane releases.', 'Nice touch from Doku!'. NEVER 'ball' or 'football'."
        elif imp < 0.6:
            length_rule = "SHORT: max 10 words. One short sentence."
        else:
            length_rule = "EXCITED: 1-2 sentences, max 20 words. Key moment, shout the goal!"

    recent_block = ""
    if recent:
        if config.LANGUAGE == "it":
            recent_header = "Battute appena dette (NON ripetere gli stessi verbi o strutture):"
        else:
            recent_header = "Recent lines (do NOT repeat the same verbs or structures):"
        recent_block = (
            f"{recent_header}\n"
            + "\n".join(f"  - {r}" for r in recent)
            + "\n\n"
        )

    verb_block = f"Verbi: {verb_hint}\n" if verb_hint else ""

    lookahead_block = ""
    if next_events:
        hints = []
        for ne in next_events[:2]:
            ne_type = ne.get("type", "?")
            ne_player = str(ne.get("player") or "").strip()
            unknown = config.get_unknown_player()
            ne_player = (
                ne_player
                if ne_player and ne_player.lower() not in (unknown, "sconosciuto", "the player", "")
                else "?"
            )
            hints.append(f"  - [{ne_type}] {ne_player}")
        if config.LANGUAGE == "it":
            lookahead_header = "Prossimi eventi (usali SOLO per il filo narrativo, NON citarli esplicitamente):"
        else:
            lookahead_header = "Next events (use ONLY for narrative flow, do NOT mention them explicitly):"
        lookahead_block = (
            f"{lookahead_header}\n"
            + "\n".join(hints)
            + "\n\n"
        )

    if config.LANGUAGE == "it":
        instruction = "Genera UNA battuta di telecronaca per questo evento."
        constraint1 = "VINCOLO 1: cita SOLO il giocatore che ha la palla (campo 'giocatore'). NON aggiungere mai il nome di chi riceve."
        constraint2 = "VINCOLO 2: NON inventare MAI dettagli fisici (es. pali, traverse, salvataggi sulla linea) se non deducibili dai dati."
        outro = "Se c'e' la telecronaca originale usane nomi/contesto. Rispondi SOLO con la battuta, niente altro."
        length_label = "Lunghezza"
    else:
        instruction = "Generate ONE commentary line for this event."
        constraint1 = "RULE 1: mention ONLY the player who has the ball ('player' field). NEVER add the name of the receiver."
        constraint2 = "RULE 2: NEVER invent physical details (e.g. posts, crossbars, goal-line saves) unless deducible from the data."
        outro = "If there is original commentary, use its names/context. Reply ONLY with the line, nothing else."
        length_label = "Length"

    return (
        f"{instruction}\n"
        f"Dati (JSON):\n```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```\n\n"
        f"{recent_block}"
        f"{lookahead_block}"
        f"{verb_block}"
        f"{length_label}: {length_rule}\n"
        f"{constraint1}\n"
        f"{constraint2}\n"
        f"{outro}"
    )


# --------------------------------------------------------------------------- #
# Validatore anti-allucinazioni (il "Validator" della pipeline di refinery)    #
# --------------------------------------------------------------------------- #
# Similarita' minima perche' una parola del testo "sia" un nome della rosa.
# Alta apposta: deve riconoscere flessioni/refusi (RAPHINA~RAPHINHA = 0.93)
# senza confondere nomi simili tra loro (PIERRE~PIERROT = 0.77 -> distinti).
NAME_SIMILARITY = 0.85


def mentioned_roster_names(text: str) -> set[str]:
    """Nomi della rosa (home+away) citati nel testo.

    Confronta ogni parola del testo con le parole dei nomi in rosa: match
    esatto oppure fuzzy sopra NAME_SIMILARITY. I nomi multi-parola
    ("BRUNO GUIMARAES") contano se una qualunque loro parola compare.
    """
    roster = set(config.ROSTER_HOME) | set(config.ROSTER_AWAY)
    words = set(re.findall(r"[A-Za-zÀ-ÿ]{3,}", text.upper()))
    mentioned = set()
    for name in roster:
        for part in name.split():
            if part in words or any(
                SequenceMatcher(None, w, part).ratio() >= NAME_SIMILARITY for w in words
            ):
                mentioned.add(name)
                break
    return mentioned


def validate_line(text: str, event: dict) -> bool:
    """True se la battuta cita SOLO giocatori pertinenti all'evento.

    Pertinenti = i nomi presenti nell'evento (player, player_home,
    player_away) piu' quelli della telecronaca originale (il prompt invita
    l'LLM a usarli). Qualunque ALTRO nome della rosa nel testo e' inventato
    dall'LLM -> la battuta va scartata (fallback al template).
    Una battuta senza nomi ("Che parata!") e' sempre valida.
    """
    allowed: set[str] = set()
    for key in ("player", "player_home", "player_away"):
        name = str(event.get(key) or "").upper().strip()
        if name:
            allowed |= mentioned_roster_names(name) | {name}
    if event.get("original_commentary"):
        allowed |= mentioned_roster_names(str(event["original_commentary"]))

    invented = mentioned_roster_names(text) - allowed
    if invented:
        logger.debug("Nomi non pertinenti nella battuta: %s", invented)
    return not invented


def _strip_markdown(text: str) -> str:
    """Rimuove formattazione markdown che alcuni LLM inseriscono nel testo.
    XTTS leggerebbe asterischi e underscore come caratteri, rovinando l'audio."""
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _snap_names(text: str, event_names: set[str] | None = None) -> str:
    """Sostituisce varianti LLM di nomi con il canonico (fuzzy match).

    Usa sia il roster da config sia i nomi estratti dagli eventi stessi, così
    funziona anche su clip di squadre non configurate nel profilo attivo.
    """
    roster_parts: list[str] = []
    for name in list(config.ROSTER_HOME) + list(config.ROSTER_AWAY):
        for part in name.split():
            if part not in roster_parts:
                roster_parts.append(part)
    for name in event_names or set():
        for part in name.split():
            if part not in roster_parts:
                roster_parts.append(part)

    if not roster_parts:
        return text

    result = []
    for word in text.split():
        m = re.match(r"([A-Za-zÀ-ÿ']{4,})(.*)", word)
        if m and m.group(1)[0].isupper():
            clean, suffix = m.group(1), m.group(2)
            best_ratio, best_part = 0.0, None
            for part in roster_parts:
                r = SequenceMatcher(None, clean.upper(), part).ratio()
                if r > best_ratio:
                    best_ratio, best_part = r, part
            if best_ratio >= NAME_SIMILARITY and best_part:
                result.append(best_part.capitalize() + suffix)
                continue
        result.append(word)
    return " ".join(result)


def _name_only(event: dict) -> str | None:
    """Restituisce solo il nome del giocatore se disponibile, altrimenti None."""
    player = str(event.get("player") or "").strip()
    unknown = config.get_unknown_player()
    # Controlla sia il placeholder della lingua corrente sia quelli noti.
    skip = {unknown.lower(), "il giocatore", "sconosciuto", "the player", "unknown", ""}
    if player and player.lower() not in skip:
        return player.capitalize() if player.isupper() else player
    return None


# Probabilità che un passaggio semplice venga commentato solo col nome (stile reale).
_NAME_ONLY_PROB = 0.45


class LLMScriptGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.system_prompt = config.LLM_SYSTEM_PROMPT

    def generate(self, events: list[dict], delay: float = 0.5) -> list[dict]:
        # Nomi canonici estratti dagli eventi (integrano il roster di config).
        event_names: set[str] = set()
        for ev in events:
            for key in ("player", "player_home", "player_away"):
                n = str(ev.get(key) or "").strip().upper()
                if n and n not in ("IL GIOCATORE", "SCONOSCIUTO", ""):
                    event_names.add(n)

        script: list[dict] = []
        recent: list[str] = []
        total = len(events)
        for i, event in enumerate(events):
            source = "llm"
            # Passaggi semplici: ~45% delle volte solo il nome, come fa un vero telecronista.
            if (
                event.get("type") == "pass"
                and float(event.get("importance", 0)) < 0.3
                and random.random() < _NAME_ONLY_PROB
            ):
                name = _name_only(event)
                if name:
                    text = name + "."
                    source = "name_only"
                    recent.append(text)
                    script.append(
                        {
                            "t": event.get("t", 0),
                            "text": text,
                            "event_type": event.get("type", "idle"),
                            "importance": event.get("importance", 0.1),
                            "audio_energy_level": event.get("audio_energy_level", "low"),
                            "audio_energy": event.get("audio_energy", 0.5),
                            "source": source,
                        }
                    )
                    continue
            try:
                text = self.provider.generate(
                    self.system_prompt,
                    build_event_prompt(event, i, total, recent[-4:], events[i + 1 : i + 3]),
                )
                text = _strip_markdown(text)
                text = _snap_names(text, event_names)
                # VALIDAZIONE: se l'LLM ha citato un giocatore che non c'entra
                # con l'evento, la battuta e' un'allucinazione -> template.
                if not validate_line(text, event):
                    logger.warning(
                        "Battuta LLM evento %d scartata (nome inventato): '%s'. Uso template.",
                        i + 1,
                        text,
                    )
                    text = self._fallback(event)
                    source = "template_fallback"
            except Exception as exc:
                logger.warning("Errore LLM evento %d: %s. Uso fallback template.", i + 1, exc)
                text = self._fallback(event)
                source = "template_fallback"
            recent.append(text)
            script.append(
                {
                    "t": event.get("t", 0),
                    "text": text,
                    "event_type": event.get("type", "idle"),
                    "importance": event.get("importance", 0.1),
                    "audio_energy_level": event.get("audio_energy_level", "low"),
                    "audio_energy": event.get("audio_energy", 0.5),
                    "source": source,
                }
            )
            if i < total - 1:
                time.sleep(delay)
            if (i + 1) % 10 == 0:
                logger.info("Generate %d/%d battute.", i + 1, total)
        return script

    @staticmethod
    def _fallback(event: dict) -> str:
        import random

        templates = config.get_templates().get(
            event.get("type", "idle"),
            ["Azione in corso." if config.LANGUAGE == "it" else "Play underway."],
        )
        return random.choice(templates).format(player=event.get("player", config.get_unknown_player()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2b - Generazione telecronaca con LLM")
    parser.add_argument("--events", required=True)
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic", "groq"],
        default=config.LLM_PROVIDER,
    )
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
        logger.info('Anteprima: "%s"', script[0]["text"])
    logger.info("Fase 2b completata.")


if __name__ == "__main__":
    main()
