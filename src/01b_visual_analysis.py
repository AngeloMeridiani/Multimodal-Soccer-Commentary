"""
01b_visual_analysis.py
=====================
FASE 1b - Modulo Visivo: Object Detection (YOLO) + Event Detection (OpenCV).

Complementa la Fase 1 (OCR HUD) con la comprensione VISIVA del campo:
  - YOLO rileva giocatori e pallone.
  - Si traccia la palla e si stima velocita'/zona del campo.
  - Si classificano eventi che l'HUD non mostra: tiri, dribbling, PARATE, corner.

Lavora su frame GIA' normalizzati (rotazione + crop). L'output puo' essere fuso
con gli eventi OCR della Fase 1.

Output: features/events/<nome_video>_enriched.json

Uso:
    python 01b_visual_analysis.py --video data/raw/gameplay/match1.mp4
    python 01b_visual_analysis.py --video data/raw/gameplay/match1.mp4 --merge features/events/match1.json
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

import config
from utils import ensure_dir, get_logger, iter_sampled_frames, load_json, save_json

logger = get_logger("fase1b_visivo")


class Detection(NamedTuple):
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


class BallState(NamedTuple):
    timestamp: float
    x: int
    y: int
    speed: float
    zone: str
    direction_deg: float


class YoloDetector:
    """Wrapper leggero per YOLOv8 (ultralytics)."""

    def __init__(self, model_name: str = config.YOLO_MODEL,
                 confidence: float = config.YOLO_CONFIDENCE) -> None:
        from ultralytics import YOLO   # import locale: pesante
        logger.info("Carico modello YOLO '%s'...", model_name)
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.classes_of_interest = set(config.YOLO_CLASSES_OF_INTEREST.keys())

    def detect(self, frame) -> list[Detection]:
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


class BallTracker:
    """Traccia la palla tra frame e calcola velocita'/zona/direzione."""

    def __init__(self, history_len: int = 10) -> None:
        self.history: deque[tuple[float, int, int]] = deque(maxlen=history_len)
        self.frame_h = 0
        self.frame_w = 0

    def update(self, timestamp: float, detections: list[Detection], frame_shape) -> BallState | None:
        self.frame_h, self.frame_w = frame_shape[:2]
        balls = [d for d in detections if d.class_name == "sports_ball"]
        if not balls:
            return None
        best = max(balls, key=lambda d: d.confidence)
        cx, cy = best.center

        speed, direction_deg = 0.0, 0.0
        if self.history:
            _, px, py = self.history[-1]
            dx, dy = cx - px, cy - py
            speed = math.hypot(dx, dy)
            direction_deg = math.degrees(math.atan2(-dy, dx)) % 360
        self.history.append((timestamp, cx, cy))

        return BallState(timestamp, cx, cy, speed, self._zone(cx, cy), direction_deg)

    def _zone(self, x: int, y: int) -> str:
        nx, ny = x / max(self.frame_w, 1), y / max(self.frame_h, 1)
        for name, (x1, y1, x2, y2) in config.FIELD_ZONES.items():
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return name
        return "midfield"


class VisualEventClassifier:
    """Classifica eventi da detection + ball tracking (complementare all'OCR)."""

    def __init__(self) -> None:
        self.prev_ball: BallState | None = None
        self.prev_near = 0
        self.t = config.BALL_TRACKING

    @staticmethod
    def _players_near(ball: BallState, detections: list[Detection], radius: float) -> int:
        players = [d for d in detections if d.class_name == "person"]
        return sum(1 for p in players
                   if math.hypot(p.center[0] - ball.x, p.center[1] - ball.y) < radius)

    def classify(self, timestamp: float, ball: BallState | None,
                 detections: list[Detection]) -> dict | None:
        if ball is None:
            self.prev_ball = None
            return None

        nearby = self._players_near(ball, detections, self.t["near_player_px"])
        in_box = "penalty_area" in ball.zone
        event = None

        # --- PARATA: tiro veloce in area, la palla CROLLA di velocita' con un
        #     giocatore (portiere) vicino -> respinta. ---
        if (self.prev_ball is not None and in_box
                and self.prev_ball.speed > self.t["min_speed_shot"]
                and ball.speed < self.prev_ball.speed * self.t["save_drop_ratio"]
                and nearby >= 1):
            event = self._mk(timestamp, "save", ball, nearby, "blocked")

        # --- TIRO: velocita' alta ---
        elif ball.speed > self.t["min_speed_shot"]:
            if in_box:
                event = self._mk(timestamp, "shot_on_goal", ball, nearby, "very_high")
            else:
                event = self._mk(timestamp, "shot_off", ball, nearby, "high")

        # --- DRIBBLING: molti giocatori vicini + palla a velocita' media ---
        elif (nearby >= 3 and self.prev_near >= 3
              and self.t["min_speed_pass"] < ball.speed < self.t["min_speed_shot"]):
            event = self._mk(timestamp, "dribble", ball, nearby, "medium")

        # --- CORNER: palla ferma in area dopo essere arrivata veloce ---
        elif (ball.speed < 5.0 and in_box and self.prev_ball
              and self.prev_ball.speed > self.t["min_speed_pass"]):
            event = self._mk(timestamp, "corner", ball, nearby, "stopped")

        self.prev_ball = ball
        self.prev_near = nearby
        return event

    @staticmethod
    def _mk(t: float, etype: str, ball: BallState, nearby: int, speed_label: str) -> dict:
        return {
            "t": round(t, 2),
            "type": etype,
            "importance": config.EVENT_IMPORTANCE.get(etype, 0.5),
            "ball_zone": ball.zone,
            "ball_speed": speed_label,
            "players_nearby": nearby,
            "source": "visual",
        }


def analyze_video(video_path: Path, limit: int | None = None,
                  ocr_events: list[dict] | None = None) -> list[dict]:
    detector = YoloDetector()
    tracker = BallTracker()
    classifier = VisualEventClassifier()

    visual_events: list[dict] = []
    count = 0
    for timestamp, frame in tqdm(
        iter_sampled_frames(video_path, config.FRAMES_PER_SECOND), desc="YOLO + Tracking"
    ):
        if limit and count >= limit:
            break
        count += 1
        detections = detector.detect(frame)
        ball = tracker.update(timestamp, detections, frame.shape)
        event = classifier.classify(timestamp, ball, detections)
        if event is not None:
            event["n_players"] = sum(1 for d in detections if d.class_name == "person")
            visual_events.append(event)

    if ocr_events:
        visual_events = merge_events(ocr_events, visual_events)
    visual_events.sort(key=lambda e: e["t"])
    return visual_events


def merge_events(ocr_events: list[dict], visual_events: list[dict],
                 time_tolerance: float = 1.0) -> list[dict]:
    """Fonde OCR e visivi: i gol OCR hanno precedenza; i visivi arricchiscono/aggiungono."""
    merged: list[dict] = []
    for ocr_ev in ocr_events:
        m = dict(ocr_ev)
        m.setdefault("source", "ocr")
        for vis in visual_events:
            if abs(vis["t"] - ocr_ev["t"]) <= time_tolerance:
                m["ball_zone"] = vis.get("ball_zone", "unknown")
                m["ball_speed"] = vis.get("ball_speed", "unknown")
                m["players_nearby"] = vis.get("players_nearby", 0)
                m["source"] = "ocr+visual"
                break
        merged.append(m)

    ocr_times = {e["t"] for e in ocr_events}
    for vis in visual_events:
        if not any(abs(vis["t"] - ot) <= time_tolerance for ot in ocr_times):
            merged.append(vis)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1b - Analisi visiva (YOLO + OpenCV)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max frame (debug).")
    parser.add_argument("--merge", type=str, default=None, help="JSON eventi OCR da fondere.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    ocr_events = None
    if args.merge:
        ocr_events = load_json(Path(args.merge))
        logger.info("Caricati %d eventi OCR da %s", len(ocr_events), args.merge)

    events = analyze_video(video_path, args.limit, ocr_events)

    out_path = config.EVENTS_DIR / f"{video_path.stem}_enriched.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for e in events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        by_source[e.get("source", "?")] = by_source.get(e.get("source", "?"), 0) + 1
    logger.info("Estratti %d eventi arricchiti (%s)", len(events), by_type)
    logger.info("Fonti: %s -> %s", by_source, out_path)
    logger.info("Fase 1b completata.")


if __name__ == "__main__":
    main()
