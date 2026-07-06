"""
04_synthesize.py
================
FASE 4 - Sintesi audio: testo + prosodia -> voce espressiva.

Pipeline (Coqui XTTS v2, voice cloning zero-shot):
  1) PROSODIA: prosody_mlp (Fase 3) predice l'enfasi per evento; qui usiamo
     rate_factor -> speed XTTS e energy_gain -> volume. (Fallback: regole.)
  2) BLOCCHI: le battute vengono unite in blocchi e sintetizzate in un'unica
     passata -> prosodia connessa, con voce clonata da un audio di riferimento.
  3) RIFINITURA: trim dei silenzi + fade-in, poi concatenazione -> traccia.

Il modello prosodico importa ProsodyMLP da utils (nessun accoppiamento al
nome-file della Fase 3).

Output: outputs/audio/<nome>.wav

Uso:
    python 04_synthesize.py --script features/scripts/match1.json
    python 04_synthesize.py --script features/scripts/match1.json --rule-based   # baseline A/B
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config
from utils import (
    FEATURE_DIM,
    build_prosody_mlp,
    ensure_dir,
    event_to_features,
    get_logger,
    load_json,
)

logger = get_logger("fase4_sintesi")


# --------------------------------------------------------------------------- #
# Predizione prosodia                                                         #
# --------------------------------------------------------------------------- #
class ProsodyPredictor:
    """Predice (rate, pitch, energy) per evento: modello appreso o regole."""

    def __init__(self, force_rule_based: bool = False) -> None:
        self.model = None
        self.scaler = None
        if not force_rule_based:
            self._try_load_model()
        self.mode = "rule_based" if self.model is None else "learned"
        logger.info("Predittore prosodia: modalita' = %s", self.mode)

    def _try_load_model(self) -> None:
        if not config.PROSODY_MODEL_PATH.exists():
            logger.warning(
                "Modello non trovato (%s): uso i valori a regole.", config.PROSODY_MODEL_PATH
            )
            return
        try:
            import joblib
            import torch

            ckpt = torch.load(config.PROSODY_MODEL_PATH, map_location="cpu")
            self.model = build_prosody_mlp(
                ckpt.get("feature_dim", FEATURE_DIM),
                tuple(ckpt.get("hidden", config.PROSODY_HIDDEN_DIMS)),
                len(config.PROSODY_TARGETS),
            )
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.scaler = joblib.load(config.PROSODY_SCALER_PATH)
        except Exception as exc:
            logger.warning("Caricamento modello fallito (%s): uso le regole.", exc)
            self.model = None

    def predict(self, event_type: str, importance: float) -> dict[str, float]:
        if self.model is not None and self.scaler is not None:
            import torch

            feat = event_to_features(event_type, importance).reshape(1, -1)
            feat = self.scaler.transform(feat).astype(np.float32)
            with torch.no_grad():
                out = self.model(torch.tensor(feat)).numpy().ravel()
            values = {k: float(out[i]) for i, k in enumerate(config.PROSODY_TARGETS)}
        else:
            values = dict(
                config.RULE_BASED_PROSODY.get(event_type, config.RULE_BASED_PROSODY["idle"])
            )
        # Clamp di sicurezza (vale per entrambe le modalita').
        for key in config.PROSODY_TARGETS:
            lo, hi = config.PROSODY_CLAMP[key]
            values[key] = float(np.clip(values[key], lo, hi))
        return values


# --------------------------------------------------------------------------- #
# Motori TTS (voce neutra)                                                     #
# --------------------------------------------------------------------------- #
class TTSEngine:
    """Interfaccia motore TTS. Unica implementazione: CoquiEngine (XTTS v2)."""

    def synth_neutral(
        self, text: str, speed: float | None = None, excited: bool = False
    ) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class CoquiEngine(TTSEngine):
    """TTS espressivo (Coqui XTTS v2). Multilingue nativo (italiano incluso)
    con voice cloning zero-shot da speaker_wav. Modello caricato UNA volta."""

    def __init__(self) -> None:
        import os

        # XTTS v2 e' sotto licenza CPML: senza questo, al primo download il
        # loader chiede conferma interattiva e si blocca in modalita' batch.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS

        logger.info("TTS: Coqui (%s).", config.COQUI_MODEL)
        self.tts = TTS(config.COQUI_MODEL)
        # Risolve i riferimenti rispetto alla root del progetto (robusto al cwd).
        self.speaker_wav = (
            str(config.PROJECT_ROOT / config.COQUI_SPEAKER_WAV)
            if config.COQUI_SPEAKER_WAV
            else None
        )
        self.speaker_wav_excited = (
            str(config.PROJECT_ROOT / config.COQUI_SPEAKER_WAV_EXCITED)
            if config.COQUI_SPEAKER_WAV_EXCITED
            else None
        )

    def synth_neutral(
        self, text: str, speed: float | None = None, excited: bool = False
    ) -> tuple[np.ndarray, int]:
        # XTTS v2 verbalizza il punto finale ("punto") sulle frasi corte:
        # lo togliamo (l'intonazione di fine frase resta comunque naturale).
        # Virgole interne, ! e ? restano perche' guidano la prosodia.
        text = text.rstrip().rstrip(".").rstrip()
        # split_sentences=False: il blocco (piu' frasi) viene generato in
        # UN'UNICA passata -> prosodia continua/connessa tra le frasi.
        kwargs = {
            "text": text,
            "language": config.COQUI_LANGUAGE,
            "speed": config.COQUI_SPEED if speed is None else speed,
            "split_sentences": False,
        }
        # Riferimento concitato per gli eventi importanti (se disponibile).
        ref = (
            self.speaker_wav_excited if (excited and self.speaker_wav_excited) else self.speaker_wav
        )
        if ref:
            kwargs["speaker_wav"] = ref
        wav = np.asarray(self.tts.tts(**kwargs), dtype=np.float32)
        sr = getattr(self.tts.synthesizer, "output_sample_rate", config.SAMPLE_RATE)
        if sr != config.SAMPLE_RATE:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        return wav, config.SAMPLE_RATE


def make_tts_engine(name: str = "coqui") -> TTSEngine:
    """Unico motore supportato: Coqui XTTS v2 (voice cloning, italiano nativo)."""
    if name != "coqui":
        logger.warning("Motore '%s' non piu' supportato: uso XTTS (coqui).", name)
    return CoquiEngine()


# --------------------------------------------------------------------------- #
# Sintesi della traccia                                                       #
# --------------------------------------------------------------------------- #
def _trim_silence(
    wav: np.ndarray, sr: int, thresh: float = 0.02, margin_ms: int = 30
) -> np.ndarray:
    """Taglia il silenzio in testa/coda (XTTS aggiunge padding a ogni battuta).
    Soglia sull'ampiezza + piccolo margine per non tagliare gli attacchi."""
    active = np.where(np.abs(wav) > thresh)[0]
    if active.size == 0:
        return wav
    pad = int(sr * margin_ms / 1000)
    start = max(0, active[0] - pad)
    end = min(len(wav), active[-1] + 1 + pad)
    return wav[start:end]


def _fade_in(wav: np.ndarray, sr: int, ms: int) -> np.ndarray:
    """Rampa lineare in ingresso: smorza l'attacco 'rauco' d'avvio di XTTS."""
    n = min(len(wav), int(sr * ms / 1000))
    if n <= 0:
        return wav
    wav = wav.copy()
    wav[:n] *= np.linspace(0.0, 1.0, n, dtype=wav.dtype)
    return wav


def _is_excited(importance: float) -> bool:
    return importance >= config.EMPHASIS_IMPORTANCE_THRESHOLD


def _group_utterances(
    items: list[tuple[str, str, float, float]], max_chars: int, max_gap_s: float
) -> list[tuple[str, str, float, float]]:
    """Unisce battute consecutive in blocchi <= max_chars mantenendo la
    punteggiatura interna. Si spezza un blocco quando: cambia il livello
    emotivo (calmo<->concitato), cosi' un evento importante (es. gol) finisce
    in un blocco a se' e puo' ricevere enfasi propria; OPPURE due battute sono
    lontane nel tempo piu' di max_gap_s (le fonderemmo perdendo la sincronia).
    Ogni item e' (testo, event_type, importanza, t); il blocco eredita
    event_type/importanza della battuta PIU' importante (il suo "carattere") e
    il timestamp della PRIMA battuta (quando parte sulla timeline)."""
    chunks: list[tuple[str, str, float, float]] = []
    cur, cur_evt, cur_imp, cur_t = "", "idle", -1.0, 0.0
    prev_t: float | None = None
    for text, evt, imp, t in items:
        text = text.strip()
        if not text:
            continue
        candidate = f"{cur} {text}".strip()
        gap_too_big = prev_t is not None and (t - prev_t) > max_gap_s
        if cur and (
            _is_excited(imp) != _is_excited(cur_imp)
            or len(candidate) > max_chars
            or gap_too_big
        ):
            chunks.append((cur, cur_evt, cur_imp, cur_t))
            cur, cur_evt, cur_imp, cur_t = text, evt, imp, t
        else:
            if not cur:  # prima battuta del blocco: ne fissa il timestamp
                cur_t = t
            cur = candidate
            if imp > cur_imp:  # rappresentante del blocco = piu' importante
                cur_evt, cur_imp = evt, imp
        prev_t = t
    if cur:
        chunks.append((cur, cur_evt, cur_imp, cur_t))
    return chunks


def synthesize(script: list[dict], predictor: ProsodyPredictor, tts: TTSEngine) -> np.ndarray:
    sr = config.SAMPLE_RATE
    min_gap = int(config.GAP_BETWEEN_UTTERANCES_S * sr)

    # NOTA: la prosodia NON viene piu' applicata via DSP (WORLD/pyworld).
    # Passare una voce neurale gia' sintetica in un secondo vocoder la
    # "doppia-vocoda" e introduce l'artefatto metallico. Usiamo la voce del
    # TTS cosi' com'e': l'espressivita' arriva dall'audio di riferimento
    # (voice cloning). Il predittore resta per compatibilita' (log/suffix).
    #
    # Le battute vengono unite in BLOCCHI e sintetizzate in un'unica passata,
    # cosi' XTTS collega la prosodia tra le frasi (parlato piu' naturale) invece
    # di generare frasi isolate una dopo l'altra.
    items = [
        (
            line.get("text", ""),
            line.get("event_type", "idle"),
            float(line.get("importance", 0.0)),
            float(line.get("t", 0.0)),
        )
        for line in script
    ]
    chunks = _group_utterances(
        items, config.COQUI_CHUNK_MAX_CHARS, config.SYNC_MERGE_MAX_GAP_S
    )
    lo, hi = config.COQUI_SPEED_CLAMP

    # Sintesi + posizionamento sulla TIMELINE: ogni blocco parte al timestamp
    # del suo evento; se il precedente non e' ancora finito, si accoda (cursor)
    # per non sovrapporre le voci. La traccia finisce con l'ultima battuta.
    placed: list[tuple[int, np.ndarray]] = []  # (campione d'inizio, audio)
    cursor = 0  # fine (in campioni) dell'ultimo blocco piazzato
    for i, (chunk, evt, imp, t) in enumerate(chunks):
        # ENFASI DAL MODELLO (Fase 3): rate_factor -> speed, energy_gain -> volume.
        prosody = predictor.predict(evt, imp)
        speed = float(np.clip(config.COQUI_SPEED * prosody["rate_factor"], lo, hi))
        excited = _is_excited(imp)
        try:
            voiced, _ = tts.synth_neutral(chunk, speed=speed, excited=excited)
            voiced = _trim_silence(voiced, sr)
            voiced = _fade_in(voiced, sr, config.ONSET_FADE_MS)
            voiced = voiced * float(prosody["energy_gain"])
        except Exception as exc:
            logger.warning("Blocco %d saltato (%s): %s", i + 1, chunk[:30], exc)
            continue
        start = max(int(round(t * sr)), cursor)
        placed.append((start, voiced))
        cursor = start + len(voiced) + min_gap
        # Log SOLO a blocco riuscito: prima veniva stampato "sintetizzato"
        # anche per i blocchi saltati dall'except, confondendo il conteggio.
        logger.info(
            "Sintetizzati %d/%d blocchi a t=%.2fs (inizio reale %.2fs, %s, speed=%.2f, gain=%.2f).",
            i + 1,
            len(chunks),
            t,
            start / sr,
            evt,
            speed,
            prosody["energy_gain"],
        )

    if not placed:
        return np.zeros(sr, np.float32)
    total = max(start + len(voiced) for start, voiced in placed)
    track = np.zeros(total, np.float32)
    for start, voiced in placed:
        track[start : start + len(voiced)] += voiced
    peak = float(np.max(np.abs(track))) or 1.0
    return track / peak * 0.95  # normalizzazione finale


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 4 - Sintesi audio")
    parser.add_argument("--script", required=True, help="JSON dello script (Fase 2/2b).")
    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Forza la prosodia a regole (condizione baseline dello studio).",
    )
    parser.add_argument("--engine", choices=["coqui"], default="coqui")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    import soundfile as sf

    script_path = Path(args.script)
    script = load_json(script_path)
    logger.info("Caricate %d battute da %s", len(script), script_path)

    predictor = ProsodyPredictor(force_rule_based=args.rule_based)
    tts = make_tts_engine(args.engine)
    track = synthesize(script, predictor, tts)

    suffix = "_rulebased" if args.rule_based else "_model"
    out_path = (
        Path(args.out) if args.out else config.AUDIO_OUT_DIR / f"{script_path.stem}{suffix}.wav"
    )
    ensure_dir(out_path)
    sf.write(str(out_path), track, config.SAMPLE_RATE)
    logger.info("Traccia salvata (%.1fs) -> %s", len(track) / config.SAMPLE_RATE, out_path)
    logger.info("Fase 4 completata.")


if __name__ == "__main__":
    main()
