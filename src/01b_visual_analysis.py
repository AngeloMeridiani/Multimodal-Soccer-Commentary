"""
01b_visual_analysis.py
=====================
PHASE 1b - Visual module: Object Detection (YOLO) + Event Detection (OpenCV).

Complements Phase 1 (HUD OCR) with the VISUAL understanding of the field:
  - YOLO detects players and the ball.
  - The ball is tracked and its speed/zone estimated.
  - Events the HUD does not show are classified: shots, SAVES, corners.

Works on ALREADY normalized frames (rotation + crop). The output can be merged
with the OCR events of Phase 1.

Output: features/events/<video_name>_enriched.json

Usage:
    python 01b_visual_analysis.py --video data/raw/gameplay/match1.mp4
    python 01b_visual_analysis.py --video data/raw/gameplay/match1.mp4 --merge features/events/match1.json
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, deque
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

import config
from utils import ensure_dir, get_logger, iter_sampled_frames, load_json, save_json

logger = get_logger("fase1b_visivo")


class Detection(NamedTuple):
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


def _dist(point: tuple[int, int], ball: BallState) -> float:
    """Point->ball distance in px: the module's only distance formula."""
    return math.hypot(point[0] - ball.x, point[1] - ball.y)


class YoloDetector:
    """Lightweight wrapper for YOLOv8 (ultralytics)."""

    def __init__(
        self, model_name: str = config.YOLO_MODEL, confidence: float = config.YOLO_CONFIDENCE
    ) -> None:
        """
        Initialize the YOLOv8 model for object detection.
        """
        from ultralytics import YOLO  # local import: heavy

        logger.info("Carico modello YOLO '%s'...", model_name)
        self.model = YOLO(model_name)
        self.confidence = confidence
        # The ball has a dedicated lower threshold (it is small/blurred):
        # with the players' one it would disappear in most frames.
        self.ball_confidence = config.YOLO_BALL_CONFIDENCE
        self.classes_of_interest = set(config.YOLO_CLASS_MAP.values())

    def detect(self, frame) -> list[Detection]:
        """
        Run inference on a frame and return filtered detections for classes of interest.
        """
        # The model is asked the LOWER of the two thresholds; the per-class
        # filter (ball vs rest) happens later, on the single boxes.
        # imgsz from the config: it must match the model's training (960),
        # otherwise the small ball gets lost at the default resolution.
        results = self.model(
            frame,
            imgsz=config.YOLO_IMGSZ,
            conf=min(self.confidence, self.ball_confidence),
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                orig_name = self.model.names[cls_id]
                mapped_name = config.YOLO_CLASS_MAP.get(orig_name)
                if mapped_name not in self.classes_of_interest:
                    continue
                floor = self.ball_confidence if mapped_name == "sports_ball" else self.confidence
                if float(box.conf[0]) < floor:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(mapped_name, float(box.conf[0]), x1, y1, x2, y2))
        return detections


class BallTracker:
    """Tracks the ball across frames and computes speed/zone/direction."""

    def __init__(self, history_len: int = 10) -> None:
        """
        Initialize the ball tracker with a sliding window of historical positions.
        """
        self.history: deque[tuple[float, int, int]] = deque(maxlen=history_len)
        self.frame_h = 0
        self.frame_w = 0

    def update(
        self, timestamp: float, detections: list[Detection], frame_shape
    ) -> BallState | None:
        """
        Update the ball tracking state using the latest detections.
        Calculates speed and direction if a valid history exists.
        """
        self.frame_h, self.frame_w = frame_shape[:2]
        balls = [d for d in detections if d.class_name == "sports_ball"]
        if not balls:
            return None
        best = max(balls, key=lambda d: d.confidence)
        cx, cy = best.center

        speed, direction_deg = 0.0, 0.0
        max_jump = config.BALL_TRACKING.get("max_ball_jump_px", 400.0)
        if self.history:
            pt, px, py = self.history[-1]
            dx, dy = cx - px, cy - py
            dist = math.hypot(dx, dy)
            dt = timestamp - pt
            # Implausible jump or time gap: unreliable detection.
            # We do NOT invent a speed (it is the cause of the phantom shots).
            if dist <= max_jump and 0 < dt <= 0.75:
                # Speed in px/s (dist/dt), NOT px/sample: this way the
                # BALL_TRACKING thresholds do not depend on the sampling frame rate.
                speed = dist / dt
                direction_deg = math.degrees(math.atan2(-dy, dx)) % 360
        self.history.append((timestamp, cx, cy))

        return BallState(timestamp, cx, cy, speed, self._zone(cx, cy), direction_deg)

    def _zone(self, x: int, y: int) -> str:
        """
        Determine the logical field zone based on normalized coordinates.
        """
        nx, ny = x / max(self.frame_w, 1), y / max(self.frame_h, 1)
        for name, (x1, y1, x2, y2) in config.FIELD_ZONES.items():
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return name
        return "midfield"


class PossessionTracker:
    """
    Reads the ball possession by analyzing the minimap (radar).
    - Masks the team and ball MARKERS with the active profile's colors
      (config.RADAR_HSV: each interface has its own).
    - Assigns the possession to the team nearest the ball's last known
      position, with hysteresis + debounce anti-flicker.
    """

    def __init__(self) -> None:
        """
        Initialize the possession tracker using minimap radar analysis.
        """
        self.confirm = config.POSSESSION_CONFIRM_FRAMES
        self.margin = config.POSSESSION_MARGIN_PX
        self.hysteresis = config.POSSESSION_HYSTERESIS_PX
        self.current: str | None = None
        self.pending: str | None = None
        self.count = 0
        self.last_ball_pos: tuple[int, int] | None = None

    def update(self, frame) -> tuple[str | None, float]:
        """Update the possession by reading the frame's minimap.
        Returns (team, confidence): team is "home"/"away" or None until the
        possession is confirmed; confidence 0.0 = radar/ball not found.
        (Only the frame is needed: ball and YOLO detections are unrelated to the radar.)"""
        import cv2

        minimap_rect = config.HUD_REGIONS.get("minimap")
        if not minimap_rect:
            return self.current, 0.0

        h, w = frame.shape[:2]
        rx1, ry1 = int(minimap_rect[0] * w), int(minimap_rect[1] * h)
        rx2, ry2 = int(minimap_rect[2] * w), int(minimap_rect[3] * h)
        radar = frame[ry1:ry2, rx1:rx2]
        hsv = cv2.cvtColor(radar, cv2.COLOR_BGR2HSV)

        # Ball and team markers, with the ACTIVE profile's colors.
        ball_dots = self._blobs(self._mask(hsv, config.RADAR_HSV["ball"]))
        home_dots = self._blobs(self._mask(hsv, config.RADAR_HSV["home"]))
        away_dots = self._blobs(self._mask(hsv, config.RADAR_HSV["away"]))

        # The ball is the blob of its color with the largest area (the real
        # cross); if it disappears for a few frames the last known position holds.
        if ball_dots:
            self.last_ball_pos = max(ball_dots, key=lambda b: b[1])[0]
        if self.last_ball_pos is None:
            return self.current, 0.0

        # Distance of each team's nearest dot from the ball, with hysteresis
        # in favor of whoever already has possession (anti-flicker).
        dist_home = self._nearest(home_dots, self.last_ball_pos)
        dist_away = self._nearest(away_dots, self.last_ball_pos)
        if self.current == "home":
            dist_home -= self.hysteresis
        elif self.current == "away":
            dist_away -= self.hysteresis

        # MARGIN + HOLD: a team is chosen only if its nearest dot beats the
        # other by at least `margin` px. In the mixed cluster (or if a color is
        # missing) the verdict is ambiguous and the current possession is KEPT,
        # instead of flipping it on 1px of noise.
        if math.isinf(dist_home) and math.isinf(dist_away):
            candidate = self.current
        elif abs(dist_home - dist_away) < self.margin:
            candidate = self.current
        else:
            candidate = "home" if dist_home < dist_away else "away"

        # Debounce: accumulate only towards a candidate DIFFERENT from the
        # current possession; the change fires after `confirm` consistent
        # frames. A candidate equal to the current possession (or ambiguous)
        # resets the accumulation.
        if candidate is None or candidate == self.current:
            self.pending, self.count = self.current, 0
        elif candidate == self.pending:
            self.count += 1
            if self.count >= self.confirm:
                self.current, self.count = candidate, 0
        else:
            self.pending, self.count = candidate, 1
        return self.current, 1.0

    @staticmethod
    def _mask(hsv, ranges) -> "object":
        """Binary mask = OR of the HSV intervals (more than one when the color
        straddles 0/180, like red)."""
        import cv2

        mask = cv2.inRange(hsv, *ranges[0])
        for lo, hi in ranges[1:]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return mask

    @staticmethod
    def _blobs(mask) -> list[tuple[tuple[int, int], float]]:
        """Centers (x, y) and areas of the lit blobs in a binary mask."""
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            m = cv2.moments(cnt) if area > 1 else None
            if m and m["m00"]:
                blobs.append(((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])), area))
        return blobs

    @staticmethod
    def _nearest(blobs: list, ref: tuple[int, int]) -> float:
        """Distance from the blob nearest to ref; inf if there are none."""
        return min(
            (math.hypot(c[0] - ref[0], c[1] - ref[1]) for c, _ in blobs), default=float("inf")
        )


class FrameContext(NamedTuple):
    """Measures of the current frame, computed ONCE in _context() and read by
    all the event rules and the telemetry (no scattered recomputations)."""

    nearby: int  # players (goalkeeper included) within near_player_px of the ball
    in_box: bool  # ball in a penalty_area of the FIELD_ZONES
    gk_dist: float  # goalkeeper seen in THIS frame (SHOT criterion)
    gk_mem: float  # goalkeeper seen now OR remembered within gk_memory_s (SAVE)


class VisualEventClassifier:
    """Classifies events from the ball motion (complementary to the OCR).

    PATTERN: one rule per event = a dedicated method (_is_save, _shot_type,
    _is_corner) that reads the FrameContext; classify() measures the context
    once, tries the rules in priority order and applies the anti-duplicate
    cooldown. The thresholds come from config.BALL_TRACKING (defaults +
    override of the active camera profile).
    """

    def __init__(self) -> None:
        """
        Initialize the visual event classifier state and tracking variables.
        """
        self.t = config.BALL_TRACKING
        self.prev_ball: BallState | None = None
        self.prev_near = 0
        self.prev_gk_dist = float("inf")
        # Last KNOWN goalkeeper position (timestamp, (x, y)): during the save
        # dive YOLO loses it right in the decisive frames.
        self._gk_seen: tuple[float, tuple[int, int]] | None = None
        # Last timestamp per event type (anti-duplicate cooldown: at 8 fps the
        # same action satisfies the condition on several frames).
        self._last_event_t: dict[str, float] = {}
        self.last_ctx: FrameContext | None = None  # exposed for the telemetry

    # ------------------------- context measures ------------------------- #
    def _context(
        self, timestamp: float, ball: BallState, detections: list[Detection]
    ) -> FrameContext:
        """
        Extract the surrounding context (goalkeeper distance, nearby players, etc.) for the current frame.
        """
        gks = [d.center for d in detections if d.class_name == "goalkeeper"]
        if gks:
            self._gk_seen = (timestamp, min(gks, key=lambda c: _dist(c, ball)))
        gk_dist = min((_dist(c, ball) for c in gks), default=float("inf"))
        # Goalkeeper memory: the position stays valid for gk_memory_s.
        gk_mem = float("inf")
        if self._gk_seen and timestamp - self._gk_seen[0] <= self.t["gk_memory_s"]:
            gk_mem = _dist(self._gk_seen[1], ball)
        # The goalkeeper counts as a player in the density around the ball.
        people = [d.center for d in detections if d.class_name in ("person", "goalkeeper")]
        nearby = sum(1 for c in people if _dist(c, ball) < self.t["near_player_px"])
        return FrameContext(nearby, "penalty_area" in ball.zone, gk_dist, gk_mem)

    # ------------------- rules (one per event type) ----------------- #
    def _is_save(self, timestamp: float, ball: BallState, ctx: FrameContext) -> bool:
        """PARRIED SAVE (frame with ball): the ball arrived from a shot and its
        speed DROPS NEAR the goalkeeper. The case of the ball CAUGHT and held
        (disappears in the GK's hands) is handled separately in classify(), on
        the ball disappearing. The criterion is the REAL closeness of the
        keeper, now reliable with the fine-tuned detector -> no more the
        sequential 'still ball' route, which mistook midfield slowdowns for saves."""
        if self.prev_ball is None or self.prev_ball.speed <= self.t["min_speed_shot"]:
            return False
        if ball.speed >= self.prev_ball.speed * self.t["save_drop_ratio"]:
            return False
        return ctx.gk_mem < self.t["near_goalkeeper_px"]

    def _shot_type(self, ball: BallState, ctx: FrameContext) -> str | None:
        """SHOT = CLEAR ball acceleration (shot_accel_ratio) with someone on it
        and the GOALKEEPER really close (the ball goes towards the goal). The GK
        closeness, now reliable, distinguishes the shot from the strong midfield
        pass (same speed, but keeper far ~900px)."""
        # NB: 'goalpost' as a "goal in frame" signal was tried and discarded
        # (false positives at midfield). Only the goalkeeper counts.
        if not (
            self.prev_ball is not None
            and ball.speed > self.t["min_speed_shot"]
            and ball.speed > self.prev_ball.speed * self.t["shot_accel_ratio"]
            and ctx.nearby >= 1
            and min(ctx.gk_dist, self.prev_gk_dist) < self.t["shot_goal_view_px"]
        ):
            return None

        # The shot must always have a dominant horizontal component (towards the goal)
        goalward = abs(math.cos(math.radians(ball.direction_deg))) > 0.5
        if not goalward:
            return None

        if ctx.in_box:
            return "shot_on_goal"

        # Outside the box, through balls or long passes can fool the system.
        # We confirm the shot from outside the box ONLY if the keeper is very close (< 250px).
        if min(ctx.gk_dist, self.prev_gk_dist) < 250.0:
            return "shot_off"

        return None

    def _is_corner(self, ball: BallState, ctx: FrameContext) -> bool:
        """CORNER = ball still in the box after arriving fast."""
        return (
            ball.speed < 5.0
            and ctx.in_box
            and self.prev_ball is not None
            and self.prev_ball.speed > self.t["min_speed_pass"]
        )

    # ------------------------------ dispatch ----------------------------- #
    def classify(
        self, timestamp: float, ball: BallState | None, detections: list[Detection]
    ) -> dict | None:
        """
        Process the current frame state and return a recognized visual event if applicable.
        """
        if ball is None:
            # CAUGHT SAVE: if the ball, just before disappearing, was fast
            # (from a shot) and VERY CLOSE to the keeper, the GK caught it
            # (occluded in his hands -> YOLO no longer sees it). It is the
            # signature of the "caught" save (vs "parried", handled in _is_save
            # on the frames with the ball).
            caught = None
            if (
                self.prev_ball is not None
                and self.prev_ball.speed > self.t["min_speed_shot"]
                and self.prev_gk_dist < self.t["near_goalkeeper_px"]
                and self._cooldown_ok("save", timestamp)
            ):
                caught = self._mk(timestamp, "save", self.prev_ball, self.prev_near, "caught")
            # The ball is forgotten only if the gap exceeds ball_memory_s: the
            # shot's motion blur hides it right in the decisive frames.
            if (
                self.prev_ball is not None
                and timestamp - self.prev_ball.timestamp > self.t["ball_memory_s"]
            ):
                self.prev_ball = None
            self.last_ctx = None
            return caught

        ctx = self.last_ctx = self._context(timestamp, ball, detections)

        # Rules in priority order (the first that fires wins).
        if self._is_save(timestamp, ball, ctx):
            etype, label = "save", "blocked"
        elif shot := self._shot_type(ball, ctx):
            etype, label = shot, "very_high" if shot == "shot_on_goal" else "high"
        elif self._is_corner(ball, ctx):
            etype, label = "corner", "stopped"
        else:
            etype, label = None, ""

        event = None
        if etype and self._cooldown_ok(etype, timestamp):
            event = self._mk(timestamp, etype, ball, ctx.nearby, label)

        self.prev_ball, self.prev_near, self.prev_gk_dist = ball, ctx.nearby, ctx.gk_dist
        return event

    def _cooldown_ok(self, etype: str, timestamp: float) -> bool:
        """The same event type within the cooldown is the same action seen on
        several frames: only the first occurrence is kept."""
        if (
            timestamp - self._last_event_t.get(etype, float("-inf"))
            < config.VISUAL_EVENT_COOLDOWN_S
        ):
            return False
        self._last_event_t[etype] = timestamp
        return True

    @staticmethod
    def _mk(t: float, etype: str, ball: BallState, nearby: int, speed_label: str) -> dict:
        """
        Helper method to format an event dictionary.
        """
        return {
            "t": round(t, 2),
            "type": etype,
            "importance": config.EVENT_IMPORTANCE.get(etype, 0.5),
            "ball_zone": ball.zone,
            "ball_speed": speed_label,
            "players_nearby": nearby,
            "source": "visual",
        }


class ControlledPlayerDetector:
    """
    Detects the white name that appears above the BALL CARRIER on the field:
    identifies who is performing the action (the visual events would be anonymous).
    White text on green background = very high contrast, easy to filter.
    OCR run periodically (every N frames) so as not to slow down the pipeline.
    The result is cached and persisted across frames.
    """

    def __init__(self, ocr_every: int = 1) -> None:
        """
        Initialize the carrier player detector with caching capabilities.
        """
        self._reader = None
        self._ocr_every = ocr_every
        self._frame_n = 0
        self.last_name: str | None = None
        self.last_team: str | None = None
        self.last_seen_frame = 0

    def _get_reader(self):
        """
        Lazy load the EasyOCR reader to save memory until required.
        """
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=False)
        return self._reader

    def detect(self, frame) -> tuple[str | None, str | None]:
        """Look for the white name on the field. Returns (name, team) or cached values."""
        import re

        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        self._frame_n += 1

        # --- 1. Field zone: exclude the HUD above (~12%) and below (~15%) ---
        y_top, y_bot = int(h * 0.12), int(h * 0.85)
        field = frame[y_top:y_bot, :, :]
        fh, fw = field.shape[:2]

        # --- 2. Mask for the white (text) ---
        # Widened thresholds to catch semi-transparent writing or writing blurred by motion blur
        hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Join nearby letters horizontally
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        mask = cv2.dilate(mask, kernel_h, iterations=2)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )

        # --- 3. Find contours with a text shape (wide, short) ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = cw / max(ch, 1)
            # Name: 40-350px wide, 7-30px tall, ratio > 2
            if 40 < cw < 350 and 7 < ch < 30 and ratio > 2.0 and cw < fw * 0.25:
                # We could exclude zones very close to the extreme side edges if needed,
                # but for now we keep all the good candidates.
                candidates.append((x, y, cw, ch))

        if not candidates:
            self._timeout_cache()
            return self.last_name, self.last_team

        # --- 4. OCR only every N frames (for performance) ---
        if self._frame_n % self._ocr_every != 0:
            return self.last_name, self.last_team

        # We sort by decreasing area, but test up to 15 candidates because the
        # advertising boards (e.g. "abu dhabi") are larger than the name.
        candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        candidates = candidates[:15]

        found = False
        for x, y, cw, ch in candidates:
            pad = 5
            crop = field[max(0, y - pad) : y + ch + pad, max(0, x - pad) : x + cw + pad]
            if crop.size == 0:
                continue

            try:
                results = self._get_reader().readtext(crop)
                for _, text, conf in results:
                    if conf < 0.25:
                        continue
                    name = re.sub(r"[^A-Za-zÀ-ɏ\s'-]", "", text).strip().upper()
                    if len(name) < 3:
                        continue
                    team = self._match_roster(name)
                    if team:
                        self.last_name = name
                        self.last_team = team
                        self.last_seen_frame = self._frame_n
                        found = True
                        logger.debug("Nome campo trovato a %d,%d: '%s' -> %s", x, y, name, team)
                        return name, team
            except Exception as exc:
                logger.debug("OCR nome campo fallito: %s", exc)

        if not found:
            self._timeout_cache()

        return self.last_name, self.last_team

    def _timeout_cache(self):
        """
        Clear the cached player name if not detected for several frames.
        """
        # Clear the cache if we do not see the name for 8 frames (~1 second at 8 fps)
        if hasattr(self, 'last_seen_frame') and self._frame_n - self.last_seen_frame > 8:
            self.last_name = None
            self.last_team = None

    @staticmethod
    def _match_roster(name: str) -> str | None:
        """Team of the read name: direct containment (only for long-enough
        fragments), then fuzzy above a threshold. TIGHT thresholds: a garbled
        reading ('NEXE'->NEUER at 0.67, 'NES' substring of 'NUNES') must NOT
        snap to the wrong roster -> better no match (the cached name holds)
        than a wrong one, which would flip possession and attribution."""
        from difflib import SequenceMatcher

        nc = name.replace(" ", "")
        rosters = ((config.ROSTER_HOME, "home"), (config.ROSTER_AWAY, "away"))
        # (a) Direct containment (truncated fragments: "INHOS" in "MARQUINHOS").
        # Only if the fragment is long enough: "NES" is a substring of
        # "NUNES"/"STONES" but it is noise, not a truncated name.
        if len(nc) >= config.CARRIER_NAME_MIN_LEN:
            for roster, team in rosters:
                if any(nc in r.replace(" ", "") or r.replace(" ", "") in nc for r in roster):
                    return team
        # (b) Best fuzzy similarity among ALL the names of the two rosters.
        best_score, best_team = max(
            (SequenceMatcher(None, nc, r.replace(" ", "")).ratio(), team)
            for roster, team in rosters
            for r in roster
        )
        return best_team if best_score >= config.CARRIER_NAME_MIN_FUZZY else None


def analyze_video(
    video_path: Path, limit: int | None = None, ocr_events: list[dict] | None = None, swap_colors: bool = False
) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (enriched_events, possession_timeline, ball_telemetry)."""
    detector = YoloDetector()
    tracker = BallTracker()
    classifier = VisualEventClassifier()
    possession = PossessionTracker()
    # Reads the name above the ball carrier: it is the PRIMARY possession
    # signal (more reliable than the radar). The OCR is costly, so it is done
    # every CARRIER_OCR_INTERVAL_S seconds (the step adapts to the visual frame
    # rate); between two readings the cached name holds.
    ocr_every = max(1, round(config.CARRIER_OCR_INTERVAL_S * config.VISUAL_FRAMES_PER_SECOND))
    controlled = ControlledPlayerDetector(ocr_every=ocr_every)

    visual_events: list[dict] = []
    possession_timeline: list[dict] = []
    # Frame-by-frame ball telemetry: serves to CALIBRATE the BALL_TRACKING
    # thresholds on the real data (true speed of shots/passes/saves) instead of
    # eyeballing them. Saved in <stem>_ball.json.
    ball_timeline: list[dict] = []
    count = 0
    # DENSE sampling (VISUAL_FRAMES_PER_SECOND, not the OCR's 2 fps):
    # needed to see the shots, which between two samples at 2 fps would vanish.
    for timestamp, frame in tqdm(
        iter_sampled_frames(video_path, config.VISUAL_FRAMES_PER_SECOND), desc="YOLO + Tracking"
    ):
        if limit and count >= limit:
            break
        count += 1
        detections = detector.detect(frame)
        ball = tracker.update(timestamp, detections, frame.shape)

        # --- possession: FUSION carrier-name (primary) + radar (fallback) ---
        # The white name above the carrier DIRECTLY identifies who has the
        # ball: it is the most reliable signal. The radar (colors on the
        # minimap) is ambiguous when the teams cluster around the ball, so we
        # use it only when the name is not readable.
        carrier_name, carrier_team = controlled.detect(frame)
        poss_radar, poss_fill = possession.update(frame)
        if carrier_team in ("home", "away"):
            poss_team, poss_conf, poss_src = carrier_team, 1.0, "carrier"
            # Sync the tracker's internal state with the certain visual carrier.
            # If we do not, when the name disappears the radar ignores the hysteresis
            # because it does not know the possession had changed.
            possession.current = carrier_team
            possession.pending = None
            possession.count = 0
        else:
            poss_team, poss_conf, poss_src = poss_radar, round(poss_fill, 3), "radar"
        possession_timeline.append(
            {
                "t": round(timestamp, 2),
                "possession": poss_team,
                "confidence": poss_conf,
                "source": poss_src,
            }
        )

        event = classifier.classify(timestamp, ball, detections)

        # Telemetry: the same measures already computed by classify() (FrameContext),
        # without recomputations. Serves to calibrate BALL_TRACKING on the real numbers.
        if ball is not None and classifier.last_ctx is not None:
            ctx = classifier.last_ctx
            ball_timeline.append(
                {
                    "t": round(timestamp, 2),
                    "speed_px_s": round(ball.speed, 1),
                    "zone": ball.zone,
                    "gk_dist_px": None if math.isinf(ctx.gk_dist) else round(ctx.gk_dist, 1),
                    "gk_mem_px": None if math.isinf(ctx.gk_mem) else round(ctx.gk_mem, 1),
                    "players_near": ctx.nearby,
                }
            )

        if event is not None:
            event["n_players"] = sum(
                1 for d in detections if d.class_name in ("person", "goalkeeper")
            )
            event["possession"] = poss_team
            # Gives a NAME to the visual event (otherwise Phase 2 generates
            # generic sentences like "What a shot by the player!"). The name on
            # the field appears above whoever has the ball, so it is the author of the action.
            if carrier_name:
                event["player"] = carrier_name
                event["player_team"] = carrier_team
            visual_events.append(event)

    # Invert the timeline if the colors were swapped
    if swap_colors:
        for p in possession_timeline:
            if p["possession"] == "home":
                p["possession"] = "away"
            elif p["possession"] == "away":
                p["possession"] = "home"

    if ocr_events:
        visual_events = merge_events(ocr_events, visual_events, possession_timeline)
    # Fill the gaps: inserts turnovers from the visual possession changes
    # and carries when a player carries the ball for a long time without events.
    visual_events = _fill_event_gaps(visual_events, possession_timeline)
    visual_events.sort(key=lambda e: e["t"])
    visual_events = _fix_shot_attribution(visual_events)
    visual_events = _collapse_consecutive(visual_events)
    return visual_events, possession_timeline, ball_timeline


def _collapse_consecutive(events: list[dict]) -> list[dict]:
    """Collapse consecutive pass/carry events with the SAME player, keeping only
    the first. After the possession correction (merge_events) the Phase 1 rule
    'pass = player changed' breaks: the re-labeling by possession can produce
    consecutive passes on the same name (e.g. the nameplate flickering at the
    start of the clip -> 3 identical passes on OLISE). Without this, the
    commentary would repeat 'Olise. Olise. Olise.'. Only routine events
    (pass, carry) are touched: goals/saves/shots/turnovers always stay.
    Exception: if more than 3 seconds pass, we allow the repetition, because
    the player is carrying the ball for a long time and we want to avoid long silences."""
    ROUTINE = {"pass", "carry"}
    collapsed: list[dict] = []
    for ev in events:
        prev = collapsed[-1] if collapsed else None
        if (
            prev is not None
            and ev.get("type") in ROUTINE
            and ev.get("type") == prev.get("type")
            and ev.get("player") == prev.get("player")
            and ev["t"] - prev["t"] <= 3.0
        ):
            continue  # consecutive duplicate same type+player: discard if close in time
        collapsed.append(ev)
    return collapsed


def possession_at(timeline: list[dict], t: float, tol: float = 3.0) -> str | None:
    """Team in possession at time t: the nearest sample within tol seconds,
    otherwise the last known possession BEFORE t."""
    vicini = [p for p in timeline if p.get("possession") and abs(p["t"] - t) <= tol]
    if vicini:
        return min(vicini, key=lambda p: abs(p["t"] - t))["possession"]
    for p in reversed(timeline):
        if p["t"] <= t and p.get("possession"):
            return p["possession"]
    return None


def merge_events(
    ocr_events: list[dict],
    visual_events: list[dict],
    possession_timeline: list[dict] | None = None,
    time_tolerance: float = 1.0,
) -> list[dict]:
    """
    Merges the events extracted from the OCR with those detected visually (YOLO).
    Uses the possession_timeline to establish the team in possession and resolve
    the actor's identity for the purely visual events.
    """
    possession_timeline = possession_timeline or []
    import logging
    logger = logging.getLogger("fase1b_visivo")

    merged: list[dict] = []
    consumed: set[int] = set()  # indices of the visual events already merged
    for ocr_ev in ocr_events:
        m = dict(ocr_ev)
        m.setdefault("source", "ocr")

        # --- carrier correction based on the visual possession ---
        poss = possession_at(possession_timeline, ocr_ev["t"])
        if poss in ("home", "away"):
            name = m.get("player_home") if poss == "home" else m.get("player_away")
            if name:
                m["player"] = name
                m["player_team"] = poss
                m["possession"] = poss  # <-- Also update the possession field!
                m["possession_certain"] = True
                m["possession_source"] = "visual"

        # Each visual event merges with a SINGLE OCR event (the nearest in
        # time): without the "consumed" flag the same save would upgrade all
        # the OCR events within the tolerance -> duplicated saves.
        best_j, best_dt = None, time_tolerance
        for j, vis in enumerate(visual_events):
            if j in consumed:
                continue
            dt = abs(vis["t"] - ocr_ev["t"])
            if dt <= best_dt:
                best_j, best_dt = j, dt
        if best_j is not None:
            vis = visual_events[best_j]
            consumed.add(best_j)
            m["ball_zone"] = vis.get("ball_zone", "unknown")
            m["ball_speed"] = vis.get("ball_speed", "unknown")
            m["players_nearby"] = vis.get("players_nearby", 0)
            m["source"] = "ocr+visual"
            # The MOST IMPORTANT type wins: a save/shot seen by the visual
            # module must not be downgraded to the "pass" that the OCR records
            # at the same instant (the HUD does not see the shots).
            if vis.get("importance", 0.0) > m.get("importance", 0.0):
                m["type"] = vis["type"]
                m["importance"] = vis["importance"]

        merged.append(m)

    # The visual events left orphan (no nearby OCR event) enter the log as they are.
    for j, vis in enumerate(visual_events):
        if j not in consumed:
            merged.append(vis)

    merged.sort(key=lambda x: x["t"])
    return merged


def _fix_shot_attribution(events: list[dict]) -> list[dict]:
    """
    Second pass: Fix the attribution of Shots, Saves and Goals.
    At the moment of a shot or save, the HUD often updates on the goalkeeper
    or the defender. We must recover the REAL shooter by looking backwards
    in the temporally ordered list (which now also includes the turnovers).
    """
    final_merged = []
    for i, m in enumerate(events):
        is_shot = m["type"] in ("shot_on_goal", "shot_off", "shot_post", "shot_blocked", "goal")

        if is_shot or m["type"] == "save":
            poss = m.get("possession")

            # 1) Find the shooter backwards
            shooter_name = m.get("player", "il giocatore")
            shooter_team = m.get("player_team", poss)
            for prev_ev in reversed(final_merged):
                if prev_ev.get("possession") == poss and prev_ev.get("player"):
                    shooter_name = prev_ev["player"]
                    shooter_team = prev_ev.get("player_team", poss)
                    break

            if m["type"] == "save":
                # Add the shot event before the save
                shot_event = dict(m)
                shot_event["type"] = "shot_on_goal"
                shot_event["t"] = round(m["t"] - 0.5, 2)
                shot_event["importance"] = config.EVENT_IMPORTANCE.get("shot_on_goal", 0.85)
                shot_event["player"] = shooter_name
                shot_event["player_team"] = shooter_team
                final_merged.append(shot_event)

                # The save goes to the opposing goalkeeper
                if poss == "away":
                    gk_name = m.get("player_home", "il portiere")
                    m["player"] = gk_name
                    m["player_team"] = "home"
                    m["possession"] = "home"
                elif poss == "home":
                    gk_name = m.get("player_away", "il portiere")
                    m["player"] = gk_name
                    m["player_team"] = "away"
                    m["possession"] = "away"
            else:
                # It is a shot or a goal: we correct the author directly
                m["player"] = shooter_name
                m["player_team"] = shooter_team

        final_merged.append(m)

    return final_merged


def _fill_event_gaps(
    events: list[dict],
    possession_timeline: list[dict],
    carry_gap_s: float = 4.0,
) -> list[dict]:
    """Fill the gaps in the event timeline using the visual possession.

    1) TURNOVER: if the possession changes between two consecutive events (from
       the visual timeline), inserts a turnover event at the change point.
    2) CARRY: if more than carry_gap_s seconds pass between two events of the
       same team, inserts a carry to signal that the player is carrying the
       ball (avoids silences in the commentary).
    """
    if not events or not possession_timeline:
        return events

    extras: list[dict] = []
    sorted_evts = sorted(events, key=lambda e: e["t"])

    for i in range(len(sorted_evts) - 1):
        curr_ev = sorted_evts[i]
        next_ev = sorted_evts[i + 1]
        t_start = curr_ev["t"]
        t_end = next_ev["t"]
        gap = t_end - t_start

        if gap < 1.0:
            continue

        poss_start = curr_ev.get("possession")
        poss_end = next_ev.get("possession")

        # --- 1) Turnover: the possession changes between two events ---
        if poss_start and poss_end and poss_start != poss_end:
            # Find the STABLE moment of the change in the visual timeline.
            # The minimap can flicker: we take the LAST real change towards the
            # new possession (i.e. the moment it switched from another
            # possession to poss_end).
            switch_t = None
            prev_p_poss = None
            for p in possession_timeline:
                # We track the previous possession even outside the range for the first frame
                if p["t"] <= t_start:
                    prev_p_poss = p.get("possession")
                    continue

                if p["t"] <= t_end:
                    curr_p_poss = p.get("possession")
                    if curr_p_poss == poss_end and prev_p_poss != poss_end:
                        switch_t = p["t"]  # Overwrites, so it keeps the last switch
                    prev_p_poss = curr_p_poss

            if switch_t is None:
                switch_t = round((t_start + t_end) / 2, 2)
            else:
                # To avoid the turnover overlapping the next event
                if abs(switch_t - t_end) < 0.1:
                    switch_t -= 0.1

            # Player who RECOVERS (not who receives later): the nameplate of the
            # NEW team in possession at the change instant. curr_ev is the last
            # event before the switch and already carries the nameplate of the
            # team returning to possession (the HUD always shows both nameplates).
            # Before, next_ev.player was used = who touches the ball LATER, not
            # the recoverer -> the turnover ended up on the wrong player.
            recover_plate = (
                curr_ev.get("player_home") if poss_end == "home" else curr_ev.get("player_away")
            )
            turnover_player = recover_plate or next_ev.get("player", "il giocatore")
            turnover_team = poss_end

            extras.append({
                "t": round(switch_t, 2),
                "type": "turnover",
                "player": turnover_player,
                "player_team": turnover_team,
                "possession": poss_end,
                "player_resolved": bool(recover_plate) or next_ev.get("player_resolved", False),
                "name_confidence": next_ev.get("name_confidence", 0.5),
                "player_home": curr_ev.get("player_home", ""),
                "player_away": curr_ev.get("player_away", ""),
                "importance": config.EVENT_IMPORTANCE.get("turnover", 0.55),
                "score": curr_ev.get("score"),
                "clock": None,
                "source": "visual",
                "possession_certain": True,
                "possession_source": "visual",
            })        # --- 2) Carry: long gap without events ---
        if gap > carry_gap_s and poss_start:
            n_carries = int(gap // carry_gap_s)
            for i in range(1, n_carries + 1):
                carry_t = round(t_start + i * (gap / (n_carries + 1)), 2)
                carry_player = curr_ev.get("player", "il giocatore")
                carry_team = curr_ev.get("player_team", poss_start)
                extras.append({
                    "t": carry_t,
                    "type": "carry",
                    "player": carry_player,
                    "player_team": carry_team,
                    "possession": poss_start,
                    "player_resolved": curr_ev.get("player_resolved", False),
                    "name_confidence": curr_ev.get("name_confidence", 0.5),
                    "player_home": curr_ev.get("player_home", ""),
                    "player_away": curr_ev.get("player_away", ""),
                    "importance": config.EVENT_IMPORTANCE.get("carry", 0.2),
                    "score": curr_ev.get("score"),
                    "clock": None,
                    "source": "visual",
                    "possession_certain": True,
                    "possession_source": "visual",
                })

    events.extend(extras)
    events.sort(key=lambda e: e["t"])
    return events



def main() -> None:
    """
    Main entry point for Phase 1b.
    Analyzes the video to extract visual events and optionally merges them with OCR events.
    """
    parser = argparse.ArgumentParser(description="Fase 1b - Analisi visiva (YOLO + OpenCV)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max frame (debug).")
    parser.add_argument("--merge", type=str, default=None, help="JSON eventi OCR da fondere.")
    parser.add_argument(
        "--profile",
        default="auto",
        help="Profilo HUD/colori: 'auto' o un nome di config.HUD_PROFILES.",
    )
    parser.add_argument("--swap-colors", action="store_true", help="Inverte i colori della minimappa (home<->away)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    # Apply the profile (rosters/jersey colors/side) before building the classifiers.
    from utils import get_frame_at

    probe = get_frame_at(video_path, 0.0)
    pname, prof = config.select_profile(probe.shape[1], probe.shape[0], args.profile)
    config.apply_profile(prof)
    logger.info("Profilo HUD/colori: '%s' (frame %dx%d)", pname, probe.shape[1], probe.shape[0])

    ocr_events = None
    if args.merge:
        ocr_events = load_json(Path(args.merge))
        logger.info("Caricati %d eventi OCR da %s", len(ocr_events), args.merge)

    events, possession_timeline, ball_timeline = analyze_video(video_path, args.limit, ocr_events, args.swap_colors)

    out_path = config.EVENTS_DIR / f"{video_path.stem}_enriched.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    # Ball telemetry: speed/zone/keeper distance per frame, to calibrate
    # BALL_TRACKING on the clip's real numbers.
    ball_path = config.EVENTS_DIR / f"{video_path.stem}_ball.json"
    save_json(ball_timeline, ball_path)
    logger.info("Telemetria palla: %d campioni -> %s", len(ball_timeline), ball_path)

    poss_path = config.EVENTS_DIR / f"{video_path.stem}_possession.json"
    save_json(possession_timeline, poss_path)
    held = sum(1 for p in possession_timeline if p["possession"])
    logger.info(
        "Timeline possesso: %d/%d frame con possesso assegnato -> %s",
        held,
        len(possession_timeline),
        poss_path,
    )

    by_type = Counter(e["type"] for e in events)
    by_source = Counter(e.get("source", "?") for e in events)
    logger.info("Estratti %d eventi arricchiti (%s)", len(events), dict(by_type))
    logger.info("Fonti: %s -> %s", dict(by_source), out_path)
    logger.info("Fase 1b completata.")


if __name__ == "__main__":
    main()
