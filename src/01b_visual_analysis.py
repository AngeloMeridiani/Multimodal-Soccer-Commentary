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
from collections import Counter, deque
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

    def __init__(
        self, model_name: str = config.YOLO_MODEL, confidence: float = config.YOLO_CONFIDENCE
    ) -> None:
        from ultralytics import YOLO  # import locale: pesante

        logger.info("Carico modello YOLO '%s'...", model_name)
        self.model = YOLO(model_name)
        self.confidence = confidence
        # La palla ha una soglia dedicata piu' bassa (e' piccola/sfocata):
        # con quella dei giocatori sparirebbe in gran parte dei frame.
        self.ball_confidence = config.YOLO_BALL_CONFIDENCE
        self.classes_of_interest = set(config.YOLO_CLASS_MAP.values())

    def detect(self, frame) -> list[Detection]:
        # Al modello si chiede la soglia PIU' BASSA tra le due; il filtro
        # per-classe (palla vs resto) avviene dopo, sulle singole box.
        results = self.model(frame, conf=min(self.confidence, self.ball_confidence), verbose=False)
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
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=mapped_name,
                        confidence=float(box.conf[0]),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections


class BallTracker:
    """Traccia la palla tra frame e calcola velocita'/zona/direzione."""

    def __init__(self, history_len: int = 10) -> None:
        self.history: deque[tuple[float, int, int]] = deque(maxlen=history_len)
        self.frame_h = 0
        self.frame_w = 0

    def update(
        self, timestamp: float, detections: list[Detection], frame_shape
    ) -> BallState | None:
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
            # Salto implausibile o buco temporale: detection inaffidabile.
            # NON inventiamo una velocita' (e' la causa dei tiri fantasma).
            if dist <= max_jump and 0 < dt <= 0.75:
                # Velocita' in px/s (dist/dt), NON px/campione: cosi' le soglie
                # di BALL_TRACKING non dipendono dal frame rate di campionamento.
                speed = dist / dt
                direction_deg = math.degrees(math.atan2(-dy, dx)) % 360
        self.history.append((timestamp, cx, cy))

        return BallState(timestamp, cx, cy, speed, self._zone(cx, cy), direction_deg)

    def _zone(self, x: int, y: int) -> str:
        nx, ny = x / max(self.frame_w, 1), y / max(self.frame_h, 1)
        for name, (x1, y1, x2, y2) in config.FIELD_ZONES.items():
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return name
        return "midfield"


class PossessionTracker:
    """
    Legge il possesso palla analizzando la minimappa (radar).
    - Maschera i colori delle squadre (giallo = home, rosso/magenta = away)
    - Cerca la palla (arancione intenso)
    - Assegna il possesso alla squadra più vicina all'ultima posizione nota della palla.
    """

    HYSTERESIS_PX = 8.0  # bonus (px) alla squadra gia' in possesso: anti-flicker

    def __init__(self) -> None:
        self.confirm = config.POSSESSION_CONFIRM_FRAMES
        self.current: str | None = None
        self.pending: str | None = None
        self.count = 0
        self.last_ball_pos: tuple[int, int] | None = None

    def update(self, frame) -> tuple[str | None, float]:
        """Aggiorna il possesso leggendo la minimappa del frame.
        Ritorna (squadra, confidenza): squadra e' "home"/"away" o None finche'
        il possesso non e' confermato; confidenza 0.0 = radar/palla non trovati.
        (Serve solo il frame: palla e detections YOLO non c'entrano col radar.)"""
        import cv2

        minimap_rect = config.HUD_REGIONS.get("minimap")
        if not minimap_rect:
            return self.current, 0.0

        h, w = frame.shape[:2]
        rx1, ry1 = int(minimap_rect[0] * w), int(minimap_rect[1] * h)
        rx2, ry2 = int(minimap_rect[2] * w), int(minimap_rect[3] * h)
        radar = frame[ry1:ry2, rx1:rx2]
        hsv = cv2.cvtColor(radar, cv2.COLOR_BGR2HSV)

        # Blob colorati sul radar: palla (arancione), home (giallo),
        # away (rosso, che in HSV sta a cavallo dello 0/180).
        orange = self._blobs(cv2.inRange(hsv, (10, 100, 150), (25, 255, 255)))
        yellow = self._blobs(cv2.inRange(hsv, (25, 30, 150), (45, 255, 255)))
        red = self._blobs(
            cv2.bitwise_or(
                cv2.inRange(hsv, (0, 50, 100), (10, 255, 255)),
                cv2.inRange(hsv, (160, 50, 100), (180, 255, 255)),
            )
        )

        # La palla e' il blob arancione con l'area maggiore (la croce vera);
        # se sparisce per qualche frame vale l'ultima posizione nota.
        if orange:
            self.last_ball_pos = max(orange, key=lambda b: b[1])[0]
        if self.last_ball_pos is None:
            return self.current, 0.0

        # Squadra col giocatore piu' vicino alla palla, con una piccola
        # isteresi a favore di chi ha gia' il possesso (anti-flicker).
        dist_home = self._nearest(yellow, self.last_ball_pos)
        dist_away = self._nearest(red, self.last_ball_pos)
        if self.current == "home":
            dist_home -= self.HYSTERESIS_PX
        elif self.current == "away":
            dist_away -= self.HYSTERESIS_PX
        team = "home" if dist_home < dist_away else "away"

        # Debounce: il cambio vale solo dopo `confirm` frame coerenti.
        if team == self.pending:
            self.count += 1
        else:
            self.pending, self.count = team, 1
        if self.count >= self.confirm:
            self.current = self.pending
        return self.current, 1.0

    @staticmethod
    def _blobs(mask) -> list[tuple[tuple[int, int], float]]:
        """Centri (x, y) e aree dei blob accesi in una maschera binaria."""
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
        """Distanza dal blob piu' vicino a ref; inf se non ce ne sono."""
        return min(
            (math.hypot(c[0] - ref[0], c[1] - ref[1]) for c, _ in blobs), default=float("inf")
        )


class VisualEventClassifier:
    """Classifica eventi da detection + ball tracking (complementare all'OCR)."""

    def __init__(self) -> None:
        self.prev_ball: BallState | None = None
        self.prev_near = 0
        self.prev_gk_dist = float("inf")
        # Ultimo timestamp per tipo di evento: serve al cooldown anti-duplicati
        # (a 8 fps la stessa azione soddisfa la condizione su piu' frame).
        self._last_event_t: dict[str, float] = {}
        self.t = config.BALL_TRACKING

    @staticmethod
    def _players_near(ball: BallState, detections: list[Detection], radius: float) -> int:
        # Il portiere conta come giocatore ai fini della densita' intorno alla palla.
        players = [d for d in detections if d.class_name in ("person", "goalkeeper")]
        return sum(
            1 for p in players if math.hypot(p.center[0] - ball.x, p.center[1] - ball.y) < radius
        )

    @staticmethod
    def _goalkeeper_dist(ball: BallState, detections: list[Detection]) -> float:
        """Distanza (px) dal portiere piu' vicino; inf se YOLO non ne vede."""
        dists = [
            math.hypot(d.center[0] - ball.x, d.center[1] - ball.y)
            for d in detections
            if d.class_name == "goalkeeper"
        ]
        return min(dists) if dists else float("inf")

    def classify(
        self, timestamp: float, ball: BallState | None, detections: list[Detection]
    ) -> dict | None:
        if ball is None:
            # NON si azzera subito la memoria: a 8 fps YOLO perde la palla
            # per qualche frame proprio nei momenti concitati (motion blur
            # del tiro). Si dimentica solo se il buco supera ball_memory_s;
            # il timestamp dell'ultimo avvistamento e' gia' in BallState.
            if (
                self.prev_ball is not None
                and timestamp - self.prev_ball.timestamp > self.t["ball_memory_s"]
            ):
                self.prev_ball = None
            return None

        nearby = self._players_near(ball, detections, self.t["near_player_px"])
        in_box = "penalty_area" in ball.zone
        event = None

        # --- PARATA: la palla arrivava veloce (tiro) e CROLLA di velocita'
        #     addosso al PORTIERE (classe YOLO dedicata) -> presa/respinta.
        #     Vale il portiere vicino ORA o al campione PRECEDENTE: al momento
        #     del crollo la palla puo' essere gia' rimbalzata via dal corpo.
        #     Fallback senza portiere rilevato: crollo in area con un giocatore
        #     vicino (vecchio criterio, per i frame in cui YOLO perde il GK). ---
        gk_dist = self._goalkeeper_dist(ball, detections)
        near_gk = min(gk_dist, self.prev_gk_dist) < self.t["near_goalkeeper_px"]
        if (
            self.prev_ball is not None
            and self.prev_ball.speed > self.t["min_speed_shot"]
            and ball.speed < self.prev_ball.speed * self.t["save_drop_ratio"]
            and (near_gk or (in_box and nearby >= 1))
        ):
            event = self._mk(timestamp, "save", ball, nearby, "blocked")

        # --- TIRO: accelerazione brusca causata da un giocatore, col PORTIERE
        #     in vista (la porta e' inquadrata). Senza questo vincolo i
        #     passaggi forti a centrocampo (380-550 px/s, GK assente)
        #     diventano tiri fantasma. ---
        elif (
            ball.speed > self.t["min_speed_shot"]
            and self.prev_ball is not None
            and self.prev_ball.speed
            < self.t["min_speed_shot"]  # e' un CALCIO, non moto gia' in volo o un salto isolato
            and nearby >= 1  # c'e' qualcuno che l'ha colpita
            and min(gk_dist, self.prev_gk_dist) < self.t["shot_goal_view_px"]
        ):
            if in_box:
                event = self._mk(timestamp, "shot_on_goal", ball, nearby, "very_high")
            else:
                # fuori area accettiamo il tiro solo se va verso una porta (moto ~orizzontale)
                goalward = abs(math.cos(math.radians(ball.direction_deg))) > 0.5
                if goalward:
                    event = self._mk(timestamp, "shot_off", ball, nearby, "high")

        # --- DRIBBLING: molti giocatori vicini + palla a velocita' media ---
        elif (
            nearby >= 3
            and self.prev_near >= 3
            and self.t["min_speed_pass"] < ball.speed < self.t["min_speed_shot"]
        ):
            event = self._mk(timestamp, "dribble", ball, nearby, "medium")

        # --- CORNER: palla ferma in area dopo essere arrivata veloce ---
        elif (
            ball.speed < 5.0
            and in_box
            and self.prev_ball
            and self.prev_ball.speed > self.t["min_speed_pass"]
        ):
            event = self._mk(timestamp, "corner", ball, nearby, "stopped")

        # Cooldown: lo stesso tipo di evento entro VISUAL_EVENT_COOLDOWN_S e'
        # la stessa azione vista su piu' frame -> si tiene solo la prima.
        if event is not None:
            last_t = self._last_event_t.get(event["type"], float("-inf"))
            if timestamp - last_t < config.VISUAL_EVENT_COOLDOWN_S:
                event = None
            else:
                self._last_event_t[event["type"]] = timestamp

        self.prev_ball = ball
        self.prev_near = nearby
        self.prev_gk_dist = gk_dist
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


class ControlledPlayerDetector:
    """
    Rileva il nome bianco che appare sopra il PORTATORE DI PALLA sul campo:
    identifica chi sta facendo l'azione (gli eventi visivi sarebbero anonimi).
    Testo bianco su sfondo verde = contrasto altissimo, facile da filtrare.
    OCR eseguito periodicamente (ogni N frame) per non rallentare la pipeline.
    Il risultato viene cachato e persistito tra i frame.
    """

    def __init__(self, ocr_every: int = 1) -> None:
        self._reader = None
        self._ocr_every = ocr_every
        self._frame_n = 0
        self.last_name: str | None = None
        self.last_team: str | None = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=False)
        return self._reader

    def detect(self, frame) -> tuple[str | None, str | None]:
        """Cerca il nome bianco sul campo. Ritorna (nome, squadra) o valori cachati."""
        import re

        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        self._frame_n += 1

        # --- 1. Zona campo: escludi HUD sopra (~12%) e sotto (~15%) ---
        y_top, y_bot = int(h * 0.12), int(h * 0.85)
        field = frame[y_top:y_bot, :, :]
        fh, fw = field.shape[:2]

        # --- 2. Maschera per il bianco (scritta) ---
        # Soglie allargate per beccare scritte semitrasparenti o sfocate per il motion blur
        hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Unisci lettere vicine orizzontalmente
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        mask = cv2.dilate(mask, kernel_h, iterations=2)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )

        # --- 3. Trova contorni con forma da testo (largo, basso) ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = cw / max(ch, 1)
            # Nome: 40-350px largo, 7-30px alto, ratio > 2
            if 40 < cw < 350 and 7 < ch < 30 and ratio > 2.0 and cw < fw * 0.25:
                # Escludiamo zone vicinissime ai bordi laterali estremi se necessario,
                # ma per ora teniamo tutti i buoni candidati.
                candidates.append((x, y, cw, ch))

        if not candidates:
            return self.last_name, self.last_team

        # --- 4. OCR solo ogni N frame (per performance) ---
        if self._frame_n % self._ocr_every != 0:
            return self.last_name, self.last_team

        # Ordiniamo per area decrescente, ma testiamo i primi 5
        candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        candidates = candidates[:5]

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
                    name = re.sub(r"[^A-Za-z\u00C0-\u024F\s'-]", "", text).strip().upper()
                    if len(name) < 3:
                        continue
                    team = self._match_roster(name)
                    if team:
                        self.last_name = name
                        self.last_team = team
                        logger.debug("Nome campo trovato a %d,%d: '%s' -> %s", x, y, name, team)
                        return name, team
            except Exception as exc:
                logger.debug("OCR nome campo fallito: %s", exc)

        return self.last_name, self.last_team

    @staticmethod
    def _match_roster(name: str) -> str | None:
        """Match del nome letto contro le rose in config."""
        from difflib import SequenceMatcher

        nc = name.replace(" ", "")
        # (a) Contenimento diretto
        for r in config.ROSTER_HOME:
            rc = r.replace(" ", "")
            if nc in rc or rc in nc:
                return "home"
        for r in config.ROSTER_AWAY:
            rc = r.replace(" ", "")
            if nc in rc or rc in nc:
                return "away"
        # (b) Fuzzy
        best_score, best_team = 0.0, None
        for r in config.ROSTER_HOME:
            s = SequenceMatcher(None, nc, r.replace(" ", "")).ratio()
            if s > best_score:
                best_score, best_team = s, "home"
        for r in config.ROSTER_AWAY:
            s = SequenceMatcher(None, nc, r.replace(" ", "")).ratio()
            if s > best_score:
                best_score, best_team = s, "away"

        if best_score > 0.65:
            return best_team
        return None


def analyze_video(
    video_path: Path, limit: int | None = None, ocr_events: list[dict] | None = None
) -> tuple[list[dict], list[dict]]:
    detector = YoloDetector()
    tracker = BallTracker()
    classifier = VisualEventClassifier()
    possession = PossessionTracker()
    # Legge il nome sopra il portatore di palla. L'OCR e' costoso: lo si fa
    # ~1 volta ogni 1.5s di video (il passo si adatta al frame rate visivo);
    # tra una lettura e l'altra vale il nome cachato.
    ocr_every = max(1, round(1.5 * config.VISUAL_FRAMES_PER_SECOND))
    controlled = ControlledPlayerDetector(ocr_every=ocr_every)

    visual_events: list[dict] = []
    possession_timeline: list[dict] = []
    # Telemetria della palla frame per frame: serve a CALIBRARE le soglie di
    # BALL_TRACKING sui dati reali (velocita' vera di tiri/passaggi/parate)
    # invece di stimarle a occhio. Salvata in <stem>_ball.json.
    ball_timeline: list[dict] = []
    count = 0
    # Campionamento FITTO (VISUAL_FRAMES_PER_SECOND, non i 2 fps dell'OCR):
    # serve a vedere i tiri, che tra due campioni a 2 fps sparirebbero.
    for timestamp, frame in tqdm(
        iter_sampled_frames(video_path, config.VISUAL_FRAMES_PER_SECOND), desc="YOLO + Tracking"
    ):
        if limit and count >= limit:
            break
        count += 1
        detections = detector.detect(frame)
        ball = tracker.update(timestamp, detections, frame.shape)

        if ball is not None:
            gk_dist = VisualEventClassifier._goalkeeper_dist(ball, detections)
            ball_timeline.append(
                {
                    "t": round(timestamp, 2),
                    "speed_px_s": round(ball.speed, 1),
                    "zone": ball.zone,
                    "gk_dist_px": None if math.isinf(gk_dist) else round(gk_dist, 1),
                    "players_near": VisualEventClassifier._players_near(
                        ball, detections, config.BALL_TRACKING["near_player_px"]
                    ),
                }
            )

        # --- possesso dai colori squadra sulla minimappa (radar) ---
        poss_team, poss_fill = possession.update(frame)
        possession_timeline.append(
            {
                "t": round(timestamp, 2),
                "possession": poss_team,
                "confidence": round(poss_fill, 3),
            }
        )

        # --- nome del portatore di palla (scritta bianca sopra il giocatore) ---
        carrier_name, carrier_team = controlled.detect(frame)

        event = classifier.classify(timestamp, ball, detections)
        if event is not None:
            event["n_players"] = sum(
                1 for d in detections if d.class_name in ("person", "goalkeeper")
            )
            event["possession"] = poss_team
            # Da' un NOME all'evento visivo (altrimenti la Fase 2 genera frasi
            # generiche tipo "Che tiro di il giocatore!"). Il nome sul campo
            # appare sopra chi ha la palla, quindi e' l'autore dell'azione.
            if carrier_name:
                event["player"] = carrier_name
                event["player_team"] = carrier_team
            visual_events.append(event)

    if ocr_events:
        visual_events = merge_events(ocr_events, visual_events, possession_timeline)
    visual_events.sort(key=lambda e: e["t"])
    return visual_events, possession_timeline, ball_timeline


def possession_at(timeline: list[dict], t: float, tol: float = 3.0) -> str | None:
    """Squadra in possesso al tempo t: il campione piu' vicino entro tol secondi,
    altrimenti l'ultimo possesso noto PRIMA di t."""
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
    Fonde OCR e visivi. Inoltre, usando il possesso (colore maglia), CORREGGE il
    portatore: sceglie la targhetta della squadra che ha davvero la palla
    (player_home se possesso=home, player_away se possesso=away).
    """
    possession_timeline = possession_timeline or []
    merged: list[dict] = []
    consumed: set[int] = set()  # indici degli eventi visivi gia' fusi
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
                m["possession"] = poss  # <-- Aggiorna anche il campo possession!
                m["possession_certain"] = True
                m["possession_source"] = "visual"

        # Ogni evento visivo si fonde con UN SOLO evento OCR (il piu' vicino
        # nel tempo): senza il flag "consumed" la stessa parata upgradava
        # tutti gli eventi OCR entro la tolleranza -> parate duplicate.
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
            # Il tipo PIU' IMPORTANTE vince: una parata/tiro visto dal
            # modulo visivo non deve essere degradato al "pass" che l'OCR
            # registra nello stesso istante (l'HUD non vede i tiri).
            if vis.get("importance", 0.0) > m.get("importance", 0.0):
                m["type"] = vis["type"]
                m["importance"] = vis["importance"]
        merged.append(m)

    # Gli eventi visivi rimasti orfani (nessun evento OCR vicino) entrano
    # nel log cosi' come sono.
    for j, vis in enumerate(visual_events):
        if j not in consumed:
            merged.append(vis)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1b - Analisi visiva (YOLO + OpenCV)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max frame (debug).")
    parser.add_argument("--merge", type=str, default=None, help="JSON eventi OCR da fondere.")
    parser.add_argument(
        "--profile",
        default="auto",
        help="Profilo HUD/colori: 'auto' o un nome di config.HUD_PROFILES.",
    )
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

    events, possession_timeline, ball_timeline = analyze_video(video_path, args.limit, ocr_events)

    out_path = config.EVENTS_DIR / f"{video_path.stem}_enriched.json"
    ensure_dir(out_path)
    save_json(events, out_path)

    # Telemetria palla: velocita'/zona/distanza portiere per frame, per
    # calibrare BALL_TRACKING sui numeri reali della clip.
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
