"""
03_extract_vision.py
====================
FASE 3 della pipeline - Computer Vision (la piu' delicata).

Per ogni clip del manifest:
    1. campiona 1-2 frame al secondo (OpenCV);
    2. rileva il volto principale con MediaPipe Face Detection;
    3. ritaglia il volto e ne calcola l'embedding con ResNet50 (ImageNet,
       senza il layer finale -> vettore 2048-D);
    4. media gli embedding dei frame -> un vettore per clip.

Gestione errori (richiesta esplicita):
    - frame senza volto: si tenta il frame successivo;
    - clip senza alcun volto: fallback configurabile sul frame intero, altrimenti
      la clip viene marcata come "mancante" (la Fase 4 la gestira' con zero-fill);
    - video corrotto / non apribile: log e si prosegue, senza fermare la fase.

Input : features/manifest.csv + data/raw/videos/<id>.mp4
Output: features/vision/vision_features.npy -> dict {id: vettore (2048,)}

Uso:
    python src/03_extract_vision.py
    python src/03_extract_vision.py --limit 20
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

import config
from utils import get_logger, iter_sampled_frames, load_manifest, save_feature_dict, set_seed

# MediaPipe e protobuf tendono a essere rumorosi: silenziamo i warning non critici.
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger("fase3_vision")


class FaceDetector:
    """Wrapper su MediaPipe Face Detection: restituisce il ritaglio del volto piu' grande."""

    def __init__(self, min_confidence: float) -> None:
        self._mp = mp.solutions.face_detection
        self.detector = self._mp.FaceDetection(min_detection_confidence=min_confidence)

    def crop_largest_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Rileva i volti in un frame BGR e restituisce il ritaglio (RGB) del piu' grande.
        Restituisce None se non viene rilevato alcun volto.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)
        if not results.detections:
            return None

        h, w = frame_rgb.shape[:2]
        best_crop, best_area = None, 0
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            # Converto le coordinate relative in pixel e le limito ai bordi immagine.
            x1 = max(int(box.xmin * w), 0)
            y1 = max(int(box.ymin * h), 0)
            x2 = min(int((box.xmin + box.width) * w), w)
            y2 = min(int((box.ymin + box.height) * h), h)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area and x2 > x1 and y2 > y1:
                best_area = area
                best_crop = frame_rgb[y1:y2, x1:x2]
        return best_crop

    def close(self) -> None:
        self.detector.close()


class VisionEmbedder:
    """Calcola embedding visivi 2048-D con una ResNet50 pre-addestrata su ImageNet."""

    def __init__(self, device: str, image_size: int) -> None:
        self.device = device
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Identity()  # rimuovo il classificatore -> embedding grezzo
        self.model = backbone.to(device).eval()

        # Preprocessing standard ImageNet.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, rgb_crops: List[np.ndarray]) -> np.ndarray:
        """Restituisce l'embedding MEDIO (2048,) di una lista di ritagli RGB."""
        tensors = torch.stack([self.transform(c) for c in rgb_crops]).to(self.device)
        embeddings = self.model(tensors)            # (N, 2048)
        return embeddings.mean(dim=0).cpu().numpy().astype(np.float32)


def process_clip(
    video_path: Path,
    detector: FaceDetector,
    embedder: VisionEmbedder,
    fps: float,
    fullframe_fallback: bool,
) -> Optional[np.ndarray]:
    """
    Estrae il vettore visivo di una singola clip.

    Restituisce None se non si riesce a ottenere alcun frame/volto utilizzabile,
    cosi' la Fase 4 puo' trattare la modalita' come mancante.
    """
    face_crops: List[np.ndarray] = []
    frame_crops: List[np.ndarray] = []  # frame interi, usati solo come fallback

    for frame in iter_sampled_frames(video_path, fps):
        try:
            face = detector.crop_largest_face(frame)
            if face is not None and face.size > 0:
                face_crops.append(face)
            elif fullframe_fallback:
                frame_crops.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            # Un singolo frame problematico non deve compromettere l'intera clip.
            logger.debug("Frame saltato in %s: %s", video_path.name, exc)
            continue

    if face_crops:
        return embedder.embed(face_crops)
    if fullframe_fallback and frame_crops:
        logger.info("Nessun volto in %s: uso il frame intero (fallback).", video_path.name)
        return embedder.embed(frame_crops)

    return None  # nessuna informazione visiva recuperabile


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3 - Embedding visivi ResNet50")
    parser.add_argument("--limit", type=int, default=None,
                        help="Processa solo i primi N campioni (debug).")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    ids, _ = load_manifest(config.MANIFEST_PATH)
    if args.limit:
        ids = ids[: args.limit]
        logger.warning("Modalita' debug: limito a %d campioni.", len(ids))

    detector = FaceDetector(min_confidence=config.FACE_DETECTION_CONFIDENCE)
    embedder = VisionEmbedder(device=config.DEVICE, image_size=config.IMAGE_SIZE)

    features: Dict[str, np.ndarray] = {}
    no_face, failures = 0, 0

    try:
        for uid in tqdm(ids, desc="Embedding video"):
            video_path = config.VIDEO_DIR / f"{uid}.mp4"
            try:
                if not video_path.exists():
                    raise FileNotFoundError(f"File mancante: {video_path}")
                vector = process_clip(
                    video_path=video_path,
                    detector=detector,
                    embedder=embedder,
                    fps=config.FRAMES_PER_SECOND,
                    fullframe_fallback=config.VISION_FULLFRAME_FALLBACK,
                )
                if vector is None:
                    no_face += 1
                    logger.warning("Nessun vettore visivo per '%s' (clip saltata).", uid)
                    continue
                features[uid] = vector
            except Exception as exc:
                # Video corrotto/illeggibile: si registra e si continua.
                failures += 1
                logger.warning("Vision fallita per '%s': %s", uid, exc)
                continue
    finally:
        detector.close()  # rilascio sempre le risorse MediaPipe

    save_feature_dict(features, config.VISION_FEATURES_PATH)
    logger.info(
        "Salvati %d/%d vettori visivi (%d senza volto, %d errori) in %s",
        len(features), len(ids), no_face, failures, config.VISION_FEATURES_PATH,
    )
    logger.info("Fase 3 completata.")


if __name__ == "__main__":
    main()