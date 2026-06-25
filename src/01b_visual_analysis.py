"""
01b_visual_analysis.py
=====================
FASE 1b - Modulo Visivo: Object Detection (YOLO) + Event Detection (OpenCV).

Complementa la Fase 1 (OCR HUD) con una comprensione VISIVA del campo:
  - YOLO rileva giocatori, pallone e (opzionalmente) l'arbitro.
  - OpenCV traccia la traiettoria della palla e rileva eventi come tiri,
    passaggi filtranti, dribbling.
  - Classifica la zona del campo (area di rigore, centrocampo, fascia).

L'output e' un JSON "arricchito" che fonde i dati OCR (Fase 1) con quelli
visivi, producendo eventi piu' dettagliati e affidabili.

Output: features/events/<nome_video>_enriched.json

Uso:
    python src/01b_visual_analysis.py --video data/raw/gameplay/match1.mp4
    python src/01b_visual_analysis.py --video data/raw/gameplay/match1.mp4 --limit 200
    python src/01b_visual_analysis.py --video data/raw/gameplay/match1.mp4 --merge features/events/match1.json
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from tqdm import tqdm

import config
from utils import ensure_dir, get_logger, iter_sampled_frames, load_json, save_json

logger = get_logger("fase1b_visivo")


# --------------------------------------------------------------------------- #
# Strutture dati                                                               #
# --------------------------------------------------------------------------- #
class Detection(NamedTuple):
    """Una singola detection YOLO."""
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class BallState(NamedTuple):
    """Stato della palla in un dato istante."""
    timestamp: float
    x: int
    y: int
    speed: float           # pixel/frame
    zone: str              # zona del campo
    direction_deg: float   # angolo di movimento (0=destra, 90=alto)


# --------------------------------------------------------------------------- #
# Detector YOLO                                                                #
# --------------------------------------------------------------------------- #
class YoloDetector:
    """Wrapper leggero per YOLOv8 (ultralytics)."""

    def __init__(
        self,
        model_name: str = config.YOLO_MODEL,
        confidence: float = config.YOLO_CONFIDENCE,
    ) -> None:
        from ultralytics import YOLO  # import locale: pesante

        logger.info("Carico modello YOLO '%s'...", model_name)
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.classes_of_interest = set(config.YOLO_CLASSES_OF_INTEREST.keys())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Esegue la detection su un frame e restituisce le detection filtrate."""
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.classes_of_interest:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=config.YOLO_CLASSES_OF_INTEREST.get(cls_id, "unknown"),
                    confidence=float(box.conf[0]),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        return detections


# --------------------------------------------------------------------------- #
# Ball Tracker (OpenCV-based)                                                  #
# --------------------------------------------------------------------------- #
class BallTracker:
    """
    Traccia la palla tra frame consecutivi e calcola velocita'/traiettoria.

    Usa le detection YOLO della palla come input principale, con un semplice
    filtro di prossimita' per associare le detection tra frame.
    """

    def __init__(self, history_len: int = 10) -> None:
        self.history: deque[tuple[float, int, int]] = deque(maxlen=history_len)
        self.frame_h: int = 0
        self.frame_w: int = 0

    def update(
        self, timestamp: float, detections: list[Detection], frame_shape: tuple[int, ...]
    ) -> BallState | None:
        """Aggiorna lo stato della palla e restituisce BallState o None se non rilevata."""
        self.frame_h, self.frame_w = frame_shape[:2]

        # Filtra solo le detection della palla
        balls = [d for d in detections if d.class_name == "sports_ball"]
        if not balls:
            return None

        # Prendi la palla con confidenza piu' alta
        best = max(balls, key=lambda d: d.confidence)
        cx, cy = best.center

        # Calcola velocita' e direzione
        speed = 0.0
        direction_deg = 0.0
        if self.history:
            _, prev_x, prev_y = self.history[-1]
            dx, dy = cx - prev_x, cy - prev_y
            speed = math.sqrt(dx ** 2 + dy ** 2)
            direction_deg = math.degrees(math.atan2(-dy, dx)) % 360

        self.history.append((timestamp, cx, cy))

        # Classifica la zona del campo
        zone = self._classify_zone(cx, cy)

        return BallState(
            timestamp=timestamp,
            x=cx, y=cy,
            speed=speed,
            zone=zone,
            direction_deg=direction_deg,
        )

    def _classify_zone(self, x: int, y: int) -> str:
        """Determina la zona del campo in base alla posizione normalizzata della palla."""
        nx = x / max(self.frame_w, 1)
        ny = y / max(self.frame_h, 1)

        for zone_name, (x1, y1, x2, y2) in config.FIELD_ZONES.items():
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return zone_name
        return "midfield"

    def get_avg_speed(self, last_n: int = 5) -> float:
        """Velocita' media degli ultimi N punti."""
        if len(self.history) < 2:
            return 0.0
        pts = list(self.history)[-last_n:]
        speeds = []
        for i in range(1, len(pts)):
            _, x0, y0 = pts[i - 1]
            _, x1, y1 = pts[i]
            speeds.append(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
        return sum(speeds) / len(speeds) if speeds else 0.0


# --------------------------------------------------------------------------- #
# Event Classifier (basato su visione)                                         #
# --------------------------------------------------------------------------- #
class VisualEventClassifier:
    """
    Classifica gli eventi basandosi sui dati di detection + ball tracking.

    Complementa l'OCR: l'OCR rileva gol e cambi di possesso dall'HUD, qui
    rileviamo eventi che l'HUD non mostra (tiri, dribbling, ecc.).
    """

    def __init__(self) -> None:
        self.prev_ball: BallState | None = None
        self.prev_players_near_ball: int = 0
        self.tracking = config.BALL_TRACKING

    def classify(
        self,
        timestamp: float,
        ball: BallState | None,
        detections: list[Detection],
    ) -> dict | None:
        """
        Restituisce un dict evento se rileva qualcosa di significativo, altrimenti None.
        """
        if ball is None:
            self.prev_ball = None
            return None

        # Conta i giocatori vicini alla palla (raggio ~100px)
        players = [d for d in detections if d.class_name == "person"]
        nearby = sum(
            1 for p in players
            if math.sqrt((p.center[0] - ball.x) ** 2 + (p.center[1] - ball.y) ** 2) < 100
        )

        event = None

        # --- Tiro in porta: velocita' alta + palla in zona porta ---
        if ball.speed > self.tracking["min_speed_shot"]:
            if "penalty_area" in ball.zone:
                event = {
                    "t": round(timestamp, 2),
                    "type": "shot_on_goal",
                    "importance": config.EVENT_IMPORTANCE.get("shot_on_goal", 0.85),
                    "ball_zone": ball.zone,
                    "ball_speed": "very_high",
                    "players_nearby": nearby,
                    "source": "visual",
                }
            else:
                event = {
                    "t": round(timestamp, 2),
                    "type": "shot_off",
                    "importance": config.EVENT_IMPORTANCE.get("shot_off", 0.60),
                    "ball_zone": ball.zone,
                    "ball_speed": "high",
                    "players_nearby": nearby,
                    "source": "visual",
                }

        # --- Dribbling: molti giocatori vicini + palla in movimento moderato ---
        elif nearby >= 3 and self.tracking["min_speed_pass"] < ball.speed < self.tracking["min_speed_shot"]:
            if self.prev_players_near_ball >= 3:
                event = {
                    "t": round(timestamp, 2),
                    "type": "dribble",
                    "importance": config.EVENT_IMPORTANCE.get("dribble", 0.50),
                    "ball_zone": ball.zone,
                    "ball_speed": "medium",
                    "players_nearby": nearby,
                    "source": "visual",
                }

        # --- Possibile corner/punizione: palla ferma + in zona specifica ---
        elif ball.speed < 5.0 and self.prev_ball and self.prev_ball.speed > self.tracking["min_speed_pass"]:
            if "penalty_area" in ball.zone:
                event = {
                    "t": round(timestamp, 2),
                    "type": "corner",
                    "importance": config.EVENT_IMPORTANCE.get("corner", 0.45),
                    "ball_zone": ball.zone,
                    "ball_speed": "stopped",
                    "players_nearby": nearby,
                    "source": "visual",
                }

        self.prev_ball = ball
        self.prev_players_near_ball = nearby
        return event


# --------------------------------------------------------------------------- #
# Pipeline principale                                                          #
# --------------------------------------------------------------------------- #
def analyze_video(
    video_path: Path,
    limit: int | None = None,
    ocr_events: list[dict] | None = None,
) -> list[dict]:
    """
    Analizza un video con YOLO + ball tracking e restituisce la lista di eventi
    arricchiti. Se `ocr_events` è fornito, li fonde con gli eventi visivi.
    """
    detector = YoloDetector()
    tracker = BallTracker()
    classifier = VisualEventClassifier()

    visual_events: list[dict] = []
    frame_count = 0

    for timestamp, frame in tqdm(
        iter_sampled_frames(video_path, config.FRAMES_PER_SECOND),
        desc="YOLO + Tracking",
    ):
        if limit and frame_count >= limit:
            break
        frame_count += 1

        # 1. Detection YOLO
        detections = detector.detect(frame)

        # 2. Tracking palla
        ball = tracker.update(timestamp, detections, frame.shape)

        # 3. Classificazione evento
        event = classifier.classify(timestamp, ball, detections)
        if event is not None:
            # Aggiungi statistiche generali del frame
            event["n_players"] = sum(1 for d in detections if d.class_name == "person")
            visual_events.append(event)

    # Se abbiamo gli eventi OCR, fondiamoli
    if ocr_events:
        visual_events = merge_events(ocr_events, visual_events)

    # Ordina per timestamp
    visual_events.sort(key=lambda e: e["t"])

    return visual_events


def merge_events(
    ocr_events: list[dict],
    visual_events: list[dict],
    time_tolerance: float = 1.0,
) -> list[dict]:
    """
    Fonde gli eventi OCR e visivi. Priorita':
    - I gol dall'OCR hanno precedenza (l'OCR li rileva in modo piu' affidabile).
    - Gli eventi visivi (tiri, dribbling) arricchiscono gli intervalli senza OCR.
    - Se un evento visivo e uno OCR sono entro `time_tolerance` secondi,
      si fondono prendendo il tipo OCR + i dettagli visivi.
    """
    merged: list[dict] = []

    # Primo: aggiungi tutti gli eventi OCR
    for ocr_ev in ocr_events:
        merged_ev = dict(ocr_ev)
        merged_ev["source"] = "ocr"

        # Cerca un evento visivo vicino per arricchirlo
        for vis_ev in visual_events:
            if abs(vis_ev["t"] - ocr_ev["t"]) <= time_tolerance:
                # Arricchisci l'evento OCR con i dati visivi
                merged_ev["ball_zone"] = vis_ev.get("ball_zone", "unknown")
                merged_ev["ball_speed"] = vis_ev.get("ball_speed", "unknown")
                merged_ev["players_nearby"] = vis_ev.get("players_nearby", 0)
                merged_ev["source"] = "ocr+visual"
                break

        merged.append(merged_ev)

    # Secondo: aggiungi eventi visivi non coperti dall'OCR
    ocr_times = {e["t"] for e in ocr_events}
    for vis_ev in visual_events:
        if not any(abs(vis_ev["t"] - ot) <= time_tolerance for ot in ocr_times):
            merged.append(vis_ev)

    return merged


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1b - Analisi visiva (YOLO + OpenCV)")
    parser.add_argument("--video", required=True, help="Percorso del video di gameplay.")
    parser.add_argument("--limit", type=int, default=None, help="Max frame (debug).")
    parser.add_argument("--merge", type=str, default=None,
                        help="JSON eventi OCR (Fase 1) da fondere con gli eventi visivi.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    ocr_events = None
    if args.merge:
        ocr_path = Path(args.merge)
        ocr_events = load_json(ocr_path)
        logger.info("Caricati %d eventi OCR da %s", len(ocr_events), ocr_path)

    events = analyze_video(video_path, args.limit, ocr_events)

    out_path = config.EVENTS_DIR / f"{video_path.stem}_enriched.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    by_type = {}
    for e in events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    by_source = {}
    for e in events:
        s = e.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1

    logger.info("Estratti %d eventi arricchiti (%s)", len(events), by_type)
    logger.info("Fonti: %s", by_source)
    logger.info("Output -> %s", out_path)
    logger.info("Fase 1b completata.")


if __name__ == "__main__":
    main()
 