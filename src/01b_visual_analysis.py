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
    """Distanza punto->palla in px: l'unica formula di distanza del modulo."""
    return math.hypot(point[0] - ball.x, point[1] - ball.y)


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
                detections.append(Detection(mapped_name, float(box.conf[0]), x1, y1, x2, y2))
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
    - Maschera i SEGNALINI delle squadre e della palla coi colori del profilo
      attivo (config.RADAR_HSV: ogni interfaccia ha i suoi).
    - Assegna il possesso alla squadra piu' vicina all'ultima posizione nota
      della palla, con isteresi + debounce anti-flicker.
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

        # Segnalini di palla e squadre, coi colori del PROFILO attivo.
        ball_dots = self._blobs(self._mask(hsv, config.RADAR_HSV["ball"]))
        home_dots = self._blobs(self._mask(hsv, config.RADAR_HSV["home"]))
        away_dots = self._blobs(self._mask(hsv, config.RADAR_HSV["away"]))

        # La palla e' il blob del suo colore con l'area maggiore (la croce
        # vera); se sparisce per qualche frame vale l'ultima posizione nota.
        if ball_dots:
            self.last_ball_pos = max(ball_dots, key=lambda b: b[1])[0]
        if self.last_ball_pos is None:
            return self.current, 0.0

        # Squadra col giocatore piu' vicino alla palla, con una piccola
        # isteresi a favore di chi ha gia' il possesso (anti-flicker).
        dist_home = self._nearest(home_dots, self.last_ball_pos)
        dist_away = self._nearest(away_dots, self.last_ball_pos)
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
    def _mask(hsv, ranges) -> "object":
        """Maschera binaria = OR degli intervalli HSV (piu' di uno quando il
        colore sta a cavallo dello 0/180, come il rosso)."""
        import cv2

        mask = cv2.inRange(hsv, *ranges[0])
        for lo, hi in ranges[1:]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return mask

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


class FrameContext(NamedTuple):
    """Misure del frame corrente, calcolate UNA volta in _context() e lette
    da tutte le regole evento e dalla telemetria (niente ricalcoli sparsi)."""

    nearby: int  # giocatori (portiere incluso) entro near_player_px dalla palla
    in_box: bool  # palla in una penalty_area delle FIELD_ZONES
    gk_dist: float  # portiere visto in QUESTO frame (criterio del TIRO)
    gk_mem: float  # portiere visto ora O ricordato entro gk_memory_s (PARATA)


class VisualEventClassifier:
    """Classifica eventi dal moto della palla (complementare all'OCR).

    PATTERN: una regola per evento = un metodo dedicato (_is_save, _shot_type,
    _is_dribble, _is_corner) che legge il FrameContext; classify() misura il
    contesto una volta, prova le regole in ordine di priorita' e applica il
    cooldown anti-duplicati. Le soglie vengono da config.BALL_TRACKING
    (default + override del profilo camera attivo).
    """

    def __init__(self) -> None:
        self.t = config.BALL_TRACKING
        self.prev_ball: BallState | None = None
        self.prev_near = 0
        self.prev_gk_dist = float("inf")
        # Ultima posizione NOTA del portiere (timestamp, (x, y)): nel tuffo
        # della parata YOLO lo perde proprio nei frame decisivi.
        self._gk_seen: tuple[float, tuple[int, int]] | None = None
        # Ultimo timestamp per tipo di evento (cooldown anti-duplicati: a
        # 8 fps la stessa azione soddisfa la condizione su piu' frame).
        self._last_event_t: dict[str, float] = {}
        self.last_ctx: FrameContext | None = None  # esposto per la telemetria

    # ------------------------- misure di contesto ------------------------ #
    def _context(
        self, timestamp: float, ball: BallState, detections: list[Detection]
    ) -> FrameContext:
        gks = [d.center for d in detections if d.class_name == "goalkeeper"]
        if gks:
            self._gk_seen = (timestamp, min(gks, key=lambda c: _dist(c, ball)))
        gk_dist = min((_dist(c, ball) for c in gks), default=float("inf"))
        # Memoria del portiere: la posizione resta valida per gk_memory_s.
        gk_mem = float("inf")
        if self._gk_seen and timestamp - self._gk_seen[0] <= self.t["gk_memory_s"]:
            gk_mem = _dist(self._gk_seen[1], ball)
        # Il portiere conta come giocatore nella densita' intorno alla palla.
        people = [d.center for d in detections if d.class_name in ("person", "goalkeeper")]
        nearby = sum(1 for c in people if _dist(c, ball) < self.t["near_player_px"])
        return FrameContext(nearby, "penalty_area" in ball.zone, gk_dist, gk_mem)

    # ------------------- regole (una per tipo di evento) ----------------- #
    def _is_save(self, timestamp: float, ball: BallState, ctx: FrameContext) -> bool:
        """PARATA = palla che arrivava a velocita' da tiro e CROLLA. Contesto
        richiesto (uno dei due): portiere vicino, visto o ricordato (il tuffo
        lo nasconde a YOLO); oppure la via SEQUENZIALE tiro-recente + palla
        quasi ferma, che non dipende da distanze in px (robusta al cambio di
        camera) e non promuove a parata i rimbalzi che restano in moto."""
        if self.prev_ball is None or self.prev_ball.speed <= self.t["min_speed_shot"]:
            return False
        if ball.speed >= self.prev_ball.speed * self.t["save_drop_ratio"]:
            return False
        last_shot_t = max(
            self._last_event_t.get("shot_on_goal", float("-inf")),
            self._last_event_t.get("shot_off", float("-inf")),
        )
        after_shot = (
            timestamp - last_shot_t <= self.t["shot_to_save_max_s"]
            and ball.speed < self.t["min_speed_pass"]
        )
        return ctx.gk_mem < self.t["near_goalkeeper_px"] or after_shot

    def _shot_type(self, ball: BallState, ctx: FrameContext) -> str | None:
        """TIRO = calcio improvviso (lento -> veloce) con qualcuno addosso
        alla palla e la porta inquadrata: portiere visto ORA o al campione
        prima (niente memoria lunga: riporterebbe i tiri fantasma)."""
        # NB: il palo (classe goalpost) come segnale alternativo di "porta
        # inquadrata" e' stato PROVATO e SCARTATO: nel retest riportava tiri
        # e parate fantasma sulle clip di solo possesso (falsi positivi del
        # palo a centrocampo). Resta solo il criterio del portiere.
        if not (
            self.prev_ball is not None
            and self.prev_ball.speed < self.t["min_speed_shot"] < ball.speed
            and ctx.nearby >= 1
            and min(ctx.gk_dist, self.prev_gk_dist) < self.t["shot_goal_view_px"]
        ):
            return None
        if ctx.in_box:
            return "shot_on_goal"
        # Fuori area il tiro deve andare verso una porta (moto ~orizzontale).
        goalward = abs(math.cos(math.radians(ball.direction_deg))) > 0.5
        return "shot_off" if goalward else None

    def _is_dribble(self, ball: BallState, ctx: FrameContext) -> bool:
        """DRIBBLING = traffico di giocatori persistente + palla a velocita' media."""
        return (
            ctx.nearby >= 3
            and self.prev_near >= 3
            and self.t["min_speed_pass"] < ball.speed < self.t["min_speed_shot"]
        )

    def _is_corner(self, ball: BallState, ctx: FrameContext) -> bool:
        """CORNER = palla ferma in area dopo essere arrivata veloce."""
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
        if ball is None:
            # La palla si dimentica solo se il buco supera ball_memory_s: il
            # motion blur del tiro la nasconde proprio nei frame decisivi.
            if (
                self.prev_ball is not None
                and timestamp - self.prev_ball.timestamp > self.t["ball_memory_s"]
            ):
                self.prev_ball = None
            self.last_ctx = None
            return None

        ctx = self.last_ctx = self._context(timestamp, ball, detections)

        # Regole in ordine di priorita' (la prima che scatta vince).
        if self._is_save(timestamp, ball, ctx):
            etype, label = "save", "blocked"
        elif shot := self._shot_type(ball, ctx):
            etype, label = shot, "very_high" if shot == "shot_on_goal" else "high"
        elif self._is_dribble(ball, ctx):
            etype, label = "dribble", "medium"
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
        """Lo stesso tipo di evento entro il cooldown e' la stessa azione
        vista su piu' frame: si tiene solo la prima occorrenza."""
        if (
            timestamp - self._last_event_t.get(etype, float("-inf"))
            < config.VISUAL_EVENT_COOLDOWN_S
        ):
            return False
        self._last_event_t[etype] = timestamp
        return True

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
        """Squadra del nome letto: contenimento diretto, poi fuzzy (>0.65)."""
        from difflib import SequenceMatcher

        nc = name.replace(" ", "")
        rosters = ((config.ROSTER_HOME, "home"), (config.ROSTER_AWAY, "away"))
        # (a) Contenimento diretto (frammenti tronchi: "INHOS" in "MARQUINHOS").
        for roster, team in rosters:
            if any(nc in r.replace(" ", "") or r.replace(" ", "") in nc for r in roster):
                return team
        # (b) Miglior somiglianza fuzzy tra TUTTI i nomi delle due rose.
        best_score, best_team = max(
            (SequenceMatcher(None, nc, r.replace(" ", "")).ratio(), team)
            for roster, team in rosters
            for r in roster
        )
        return best_team if best_score > 0.65 else None


def analyze_video(
    video_path: Path, limit: int | None = None, ocr_events: list[dict] | None = None
) -> tuple[list[dict], list[dict], list[dict]]:
    """Ritorna (eventi_arricchiti, timeline_possesso, telemetria_palla)."""
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

        # Telemetria: le stesse misure gia' calcolate da classify() (FrameContext),
        # senza ricalcoli. Serve a calibrare BALL_TRACKING sui numeri reali.
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
