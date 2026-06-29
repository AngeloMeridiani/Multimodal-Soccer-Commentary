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


class TeamClassifier:
    """Classifica la squadra di un giocatore dal COLORE della maglia (HSV)."""

    def __init__(self) -> None:
        self.ranges = config.TEAM_JERSEY_HSV
        self.sample = config.JERSEY_SAMPLE_BOX
        self.min_fill = config.JERSEY_MIN_FILL
        self.min_margin = config.JERSEY_MIN_MARGIN

    def classify(self, frame, det: "Detection") -> tuple[str | None, float]:
        """Ritorna (team, fill) dove team in {'home','away',None}. None = incerto."""
        import cv2
        bw, bh = det.x2 - det.x1, det.y2 - det.y1
        if bw <= 0 or bh <= 0:
            return None, 0.0
        sx1 = max(det.x1 + int(self.sample[0] * bw), 0)
        sy1 = max(det.y1 + int(self.sample[1] * bh), 0)
        sx2 = det.x1 + int(self.sample[2] * bw)
        sy2 = det.y1 + int(self.sample[3] * bh)
        patch = frame[sy1:sy2, sx1:sx2]
        if patch.size == 0:
            return None, 0.0

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        fills: dict[str, float] = {}
        for team, ranges in self.ranges.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            fills[team] = float(mask.sum()) / 255.0 / mask.size

        team = max(fills, key=fills.get)
        fill = fills[team]
        other = max((v for k, v in fills.items() if k != team), default=0.0)
        # serve abbastanza colore E un margine chiaro sull'altra squadra
        if fill < self.min_fill or fill < other * self.min_margin:
            return None, round(fill, 3)
        return team, round(fill, 3)


class PossessionTracker:
    """
    Possesso = squadra del giocatore PIU' VICINO alla palla (entro un raggio),
    classificata dal colore maglia, con debounce anti-flicker.
    """

    def __init__(self) -> None:
        self.max_dist = config.POSSESSION_MAX_DIST_PX
        self.confirm = config.POSSESSION_CONFIRM_FRAMES
        self.team_clf = TeamClassifier()
        self.current: str | None = None
        self.pending: str | None = None
        self.count = 0

    def update(self, frame, ball: "BallState | None",
               detections: list["Detection"]) -> tuple[str | None, float, str | None]:
        """Ritorna (possesso_confermato, fill, carrier_name=None)."""
        if ball is None:
            return self.current, 0.0, None
        players = [d for d in detections if d.class_name == "person"]
        if not players:
            return self.current, 0.0, None

        nearest = min(players, key=lambda p: math.hypot(p.center[0] - ball.x,
                                                         p.center[1] - ball.y))
        if math.hypot(nearest.center[0] - ball.x, nearest.center[1] - ball.y) > self.max_dist:
            return self.current, 0.0, None   # palla in volo / nessuno la controlla

        team, fill = self.team_clf.classify(frame, nearest)
        if team is None:
            return self.current, fill, None

        if team == self.pending:
            self.count += 1
        else:
            self.pending, self.count = team, 1
        if self.count >= self.confirm:
            self.current = self.pending
        return self.current, fill, None



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
                  ocr_events: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    detector = YoloDetector()
    tracker = BallTracker()
    classifier = VisualEventClassifier()
    possession = PossessionTracker()

    visual_events: list[dict] = []
    possession_timeline: list[dict] = []
    count = 0
    for timestamp, frame in tqdm(
        iter_sampled_frames(video_path, config.FRAMES_PER_SECOND), desc="YOLO + Tracking"
    ):
        if limit and count >= limit:
            break
        count += 1
        detections = detector.detect(frame)
        ball = tracker.update(timestamp, detections, frame.shape)

        # --- possesso dal colore maglia del giocatore piu' vicino alla palla ---
        poss_team, poss_fill, _ = possession.update(frame, ball, detections)
        possession_timeline.append({
            "t": round(timestamp, 2),
            "possession": poss_team,
            "confidence": round(poss_fill, 3),
        })

        event = classifier.classify(timestamp, ball, detections)
        if event is not None:
            event["n_players"] = sum(1 for d in detections if d.class_name == "person")
            event["possession"] = poss_team
            visual_events.append(event)

    if ocr_events:
        visual_events = merge_events(ocr_events, visual_events, possession_timeline)
    visual_events.sort(key=lambda e: e["t"])
    return visual_events, possession_timeline


def possession_at(timeline: list[dict], t: float, tol: float = 0.75) -> str | None:
    """Squadra in possesso al tempo t (campione piu' vicino entro tol secondi)."""
    best, best_d = None, tol
    for p in timeline:
        d = abs(p["t"] - t)
        if d <= best_d and p.get("possession"):
            best, best_d = p, d
    return best["possession"] if best else None


def merge_events(ocr_events: list[dict], visual_events: list[dict],
                 possession_timeline: list[dict] | None = None,
                 time_tolerance: float = 1.0) -> list[dict]:
    """
    Fonde OCR e visivi. Inoltre, usando il possesso (colore maglia), CORREGGE il
    portatore: sceglie la targhetta della squadra che ha davvero la palla
    (player_home se possesso=home, player_away se possesso=away).
    """
    possession_timeline = possession_timeline or []
    merged: list[dict] = []
    for ocr_ev in ocr_events:
        m = dict(ocr_ev)
        m.setdefault("source", "ocr")

        # --- correzione del portatore in base al possesso visivo ---
        poss = possession_at(possession_timeline, ocr_ev["t"])
        if poss in ("home", "away"):
            name = m.get("player_home") if poss == "home" else m.get("player_away")
            if name:
                m["player"] = name
                m["player_team"] = poss
                m["possession_certain"] = True
                m["possession_source"] = "visual"

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
    parser.add_argument("--profile", default="auto",
                        help="Profilo HUD/colori: 'auto' o un nome di config.HUD_PROFILES.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    # Applica il profilo (rose/colori maglia/lato) prima di costruire i classificatori.
    from utils import get_frame_at
    probe = get_frame_at(video_path, 0.0)
    pname, prof = config.select_profile(probe.shape[1], probe.shape[0], args.profile)
    config.apply_profile(prof)
    logger.info("Profilo HUD/colori: '%s' (frame %dx%d)", pname, probe.shape[1], probe.shape[0])

    ocr_events = None
    if args.merge:
        ocr_events = load_json(Path(args.merge))
        logger.info("Caricati %d eventi OCR da %s", len(ocr_events), args.merge)

    events, possession_timeline = analyze_video(video_path, args.limit, ocr_events)

    out_path = config.EVENTS_DIR / f"{video_path.stem}_enriched.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    poss_path = config.EVENTS_DIR / f"{video_path.stem}_possession.json"
    save_json(possession_timeline, poss_path)
    held = sum(1 for p in possession_timeline if p["possession"])
    logger.info("Timeline possesso: %d/%d frame con possesso assegnato -> %s",
                held, len(possession_timeline), poss_path)

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