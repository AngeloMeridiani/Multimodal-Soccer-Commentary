"""
01_extract_text.py
===================
FASE 1 della pipeline - NLP.

Legge le annotazioni MUStARD (sarcasm_data.json), genera un embedding testuale
per ogni utterance tramite RoBERTa (mean-pooling sull'ultimo hidden state) e:

    1. salva features/text/text_features.npy   -> dict {id: vettore (768,)}
    2. salva features/manifest.csv              -> elenco ordinato (id, label)

Il manifest e' la "fonte di verita'" del progetto: tutte le fasi successive
iterano sugli ID che trovano qui, garantendo l'allineamento delle modalita'.

Uso:
    python src/01_extract_text.py
    python src/01_extract_text.py --limit 20   # smoke test su poche clip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import config
from utils import ensure_dir, get_logger, save_feature_dict, set_seed

logger = get_logger("fase1_text")


class TextEmbedder:
    """Genera embedding di frase tramite un modello Transformer di Hugging Face."""

    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        logger.info("Carico tokenizer e modello '%s' su %s...", model_name, device)
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pooling mascherato: media i token reali ignorando il padding."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Restituisce gli embedding (B, hidden_dim) di un batch di testi."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        output = self.model(**encoded)
        pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        return pooled.cpu().numpy()


def load_mustard(json_path: Path, use_context: bool) -> List[Tuple[str, str, int]]:
    """
    Carica MUStARD e restituisce una lista di tuple (id, testo, label).

    Se use_context=True, antepone il contesto della conversazione all'utterance
    (puo' aiutare a disambiguare il sarcasmo, al costo di sequenze piu' lunghe).
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Annotazioni MUStARD non trovate: {json_path}\n"
            f"Scarica 'sarcasm_data.json' nella cartella data/raw/."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Tuple[str, str, int]] = []
    for uid, entry in data.items():
        utterance = entry.get("utterance", "").strip()
        if use_context and entry.get("context"):
            context = " ".join(entry["context"])
            text = f"{context} </s> {utterance}"
        else:
            text = utterance
        label = int(bool(entry.get("sarcasm", False)))
        samples.append((uid, text, label))

    logger.info("Caricati %d campioni da MUStARD.", len(samples))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1 - Embedding testuali RoBERTa")
    parser.add_argument("--limit", type=int, default=None,
                        help="Processa solo i primi N campioni (debug).")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    samples = load_mustard(config.SARCASM_JSON, config.USE_CONTEXT)
    if args.limit:
        samples = samples[: args.limit]
        logger.warning("Modalita' debug: limito a %d campioni.", len(samples))

    embedder = TextEmbedder(
        model_name=config.TEXT_MODEL_NAME,
        device=config.DEVICE,
        max_length=config.TEXT_MAX_LENGTH,
    )

    features: Dict[str, np.ndarray] = {}
    manifest_rows: List[dict] = []
    batch_size = config.TEXT_BATCH_SIZE

    for start in tqdm(range(0, len(samples), batch_size), desc="Embedding testo"):
        batch = samples[start : start + batch_size]
        ids = [s[0] for s in batch]
        texts = [s[1] for s in batch]
        labels = [s[2] for s in batch]

        try:
            vectors = embedder.embed_batch(texts)
        except Exception as exc:  # batch corrotto: non deve fermare la pipeline
            logger.error("Errore sul batch a partire da %d: %s", start, exc)
            continue

        for uid, vec, label in zip(ids, vectors, labels):
            features[uid] = vec.astype(np.float32)
            manifest_rows.append({"id": uid, "label": label})

    # Salvataggio feature + manifest (entrambi ordinati nello stesso modo)
    save_feature_dict(features, config.TEXT_FEATURES_PATH)
    ensure_dir(config.MANIFEST_PATH)
    pd.DataFrame(manifest_rows).to_csv(config.MANIFEST_PATH, index=False)

    logger.info("Salvati %d embedding in %s", len(features), config.TEXT_FEATURES_PATH)
    logger.info("Manifest scritto in %s", config.MANIFEST_PATH)
    logger.info("Fase 1 completata.")


if __name__ == "__main__":
    main()