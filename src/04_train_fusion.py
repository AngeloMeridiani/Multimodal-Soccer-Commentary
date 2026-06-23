"""
04_train_fusion.py
==================
FASE 4 della pipeline - Fusione & Classificazione.

1. Carica i tre dizionari di feature (testo, audio, vision) e il manifest.
2. Allinea le modalita' per utterance_id. Le modalita' mancanti (tipicamente la
   vision, quando non viene trovato il volto) vengono riempite con zeri.
3. EARLY FUSION: concatena i tre vettori in un unico vettore per campione.
4. Standardizza le feature (fondamentale: RoBERTa, MFCC e ResNet hanno scale
   molto diverse) e divide in train/test in modo stratificato.
5. Addestra un MLP in PyTorch (con early stopping) e una baseline sklearn
   (Logistic Regression) per confronto rapido.
6. Stampa il classification report + confusion matrix e salva modello + scaler.

Input : features/{text,audio,vision}/*.npy + features/manifest.csv
Output: models/fusion_mlp.pt, models/scaler.joblib

Uso:
    python src/04_train_fusion.py
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from utils import ensure_dir, get_logger, load_feature_dict, load_manifest, set_seed

logger = get_logger("fase4_fusion")


# --------------------------------------------------------------------------- #
# Costruzione della matrice fusa                                              #
# --------------------------------------------------------------------------- #
def build_fused_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    Allinea le tre modalita' sul manifest e restituisce (X, y) per la Early Fusion.

    Per ogni id e ogni modalita': se il vettore esiste lo si usa, altrimenti si
    riempie con zeri della dimensione attesa (definita in config). Cosi' un
    fallimento della vision su una clip non scarta l'intero campione.
    """
    ids, labels = load_manifest(config.MANIFEST_PATH)

    text_feats = load_feature_dict(config.TEXT_FEATURES_PATH)
    audio_feats = load_feature_dict(config.AUDIO_FEATURES_PATH)
    vision_feats = load_feature_dict(config.VISION_FEATURES_PATH)

    dims = {
        "text": config.TEXT_EMBED_DIM,
        "audio": config.AUDIO_EMBED_DIM,
        "vision": config.VISION_EMBED_DIM,
    }
    missing = {"text": 0, "audio": 0, "vision": 0}

    rows = []
    for uid in ids:
        parts = []
        for name, store in (("text", text_feats),
                            ("audio", audio_feats),
                            ("vision", vision_feats)):
            vec = store.get(uid)
            if vec is None:
                vec = np.zeros(dims[name], dtype=np.float32)
                missing[name] += 1
            parts.append(np.asarray(vec, dtype=np.float32))
        rows.append(np.concatenate(parts))

    X = np.vstack(rows).astype(np.float32)
    y = labels.astype(np.int64)

    logger.info("Matrice fusa: X=%s, y=%s", X.shape, y.shape)
    logger.info("Modalita' mancanti (zero-fill) -> %s", missing)
    return X, y


# --------------------------------------------------------------------------- #
# Modello                                                                      #
# --------------------------------------------------------------------------- #
class FusionMLP(nn.Module):
    """MLP per classificazione binaria sulle feature multimodali concatenate."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
                       nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 2))  # 2 classi: normale / sarcastico
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionTrainer:
    """Gestisce training, early stopping e valutazione dell'MLP."""

    def __init__(self, input_dim: int, device: str) -> None:
        self.device = device
        self.model = FusionMLP(
            input_dim=input_dim,
            hidden_dims=config.MLP_HIDDEN_DIMS,
            dropout=config.MLP_DROPOUT,
        ).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.MLP_LR,
            weight_decay=config.MLP_WEIGHT_DECAY,
        )

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        best_f1, patience, best_state = -1.0, 0, None

        for epoch in range(1, config.MLP_EPOCHS + 1):
            self.model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0)

            val_f1 = self._validation_f1(X_val, y_val)
            logger.info("Epoch %3d | loss=%.4f | val_f1=%.4f",
                       epoch, running_loss / len(X_train), val_f1)

            # Early stopping sul F1 di validazione.
            if val_f1 > best_f1:
                best_f1, patience = val_f1, 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= config.EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping all'epoca %d (best val_f1=%.4f).",
                               epoch, best_f1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        logits = self.model(torch.from_numpy(X).to(self.device))
        return logits.argmax(dim=1).cpu().numpy()

    def _validation_f1(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        preds = self.predict(X_val)
        return f1_score(y_val, preds, average="macro")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 4 - Early Fusion + classificatore")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Salta la baseline scikit-learn.")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    # 1) Fusione
    X, y = build_fused_matrix()

    # 2) Split stratificato (mantiene il bilanciamento delle classi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    # 3) Standardizzazione: fit SOLO sul train per evitare data leakage
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # 4) MLP PyTorch
    logger.info("=== Training MLP (PyTorch) ===")
    trainer = FusionTrainer(input_dim=X_train.shape[1], device=config.DEVICE)
    trainer.fit(X_train, y_train, X_test, y_test)
    mlp_preds = trainer.predict(X_test)

    logger.info("\n%s", classification_report(
        y_test, mlp_preds, target_names=["normale", "sarcastico"], digits=4))
    logger.info("Confusion matrix MLP:\n%s", confusion_matrix(y_test, mlp_preds))

    # 5) Baseline rapida scikit-learn (utile come sanity check)
    if not args.no_baseline:
        logger.info("=== Baseline Logistic Regression (scikit-learn) ===")
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        lr_preds = clf.predict(X_test)
        logger.info("F1 macro baseline: %.4f", f1_score(y_test, lr_preds, average="macro"))

    # 6) Persistenza modello + scaler (servono entrambi per l'inferenza)
    ensure_dir(config.CLASSIFIER_PATH)
    torch.save(
        {"state_dict": trainer.model.state_dict(), "input_dim": X_train.shape[1]},
        config.CLASSIFIER_PATH,
    )
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info("Modello salvato in %s", config.CLASSIFIER_PATH)
    logger.info("Scaler salvato in %s", config.SCALER_PATH)
    logger.info("Fase 4 completata. Pipeline conclusa.")


if __name__ == "__main__":
    main()