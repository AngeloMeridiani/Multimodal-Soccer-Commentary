"""
01_calibrate_hud.py
===================
Helper to CALIBRATE the HUD regions (Phase 1).

Extracts a frame, normalizes it (rotation + letterbox crop) and draws on top
the config.HUD_REGIONS rectangles, saving an annotated image. This way you see
IF the boxes land where they should (score, clock, the two nameplates) and you
can fix the coordinates in config.py.

With --ocr it also runs EasyOCR on each region and prints what it reads (useful
to check score and names before launching the real extraction).

Usage:
    python 01_calibrate_hud.py --video data/raw/gameplay/match1.mp4
    python 01_calibrate_hud.py --video data/raw/gameplay/match1.mp4 --time 25 --ocr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import config
from utils import ensure_dir, get_frame_at, get_logger

logger = get_logger("calibrazione_hud")

# Color for each region (BGR) — only for the preview.
_COLORS = {
    "score": (60, 200, 60),
    "clock": (60, 200, 200),
    "active_player_home": (220, 120, 60),
    "active_player_away": (120, 60, 220),
}


def draw_regions(frame, regions: dict) -> "cv2.Mat":
    """
    Draw rectangles and labels for the configured HUD regions on the provided frame.
    Returns the annotated frame for visualization.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()
    for name, (x1, y1, x2, y2) in regions.items():
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        color = _COLORS.get(name, (0, 0, 255))
        cv2.rectangle(annotated, p1, p2, color, 2)
        cv2.putText(
            annotated,
            name,
            (p1[0], max(p1[1] - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_ocr(frame, regions: dict) -> dict[str, str]:
    """
    Run EasyOCR on the specified regions of the frame.
    Returns a dictionary mapping region names to the extracted text.
    """
    import easyocr

    reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=False)
    h, w = frame.shape[:2]
    results: dict[str, str] = {}
    for name, (x1, y1, x2, y2) in regions.items():
        crop = frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]
        if crop.size == 0:
            results[name] = ""
            continue
        tokens = [t for _, t, c in reader.readtext(crop) if c >= config.OCR_MIN_CONFIDENCE]
        results[name] = " ".join(tokens).strip()
    return results


def main() -> None:
    """
    Main entry point for the HUD calibration script.
    Extracts a frame, annotates regions, and optionally runs OCR for verification.
    """
    parser = argparse.ArgumentParser(description="Calibrazione regioni HUD")
    parser.add_argument("--video", required=True, help="Percorso del video.")
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Secondo da cui estrarre il frame (default: meta' clip).",
    )
    parser.add_argument("--ocr", action="store_true", help="Esegui anche l'OCR sulle regioni.")
    parser.add_argument(
        "--raw", action="store_true", help="NON normalizzare (mostra il frame grezzo)."
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    # Default time: half of the video.
    t = args.time
    if t is None:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        t = (n / fps) / 2.0 if fps and n else 1.0

    frame = get_frame_at(video_path, t, normalize=not args.raw)
    logger.info("Frame a %.2fs, dimensioni normalizzate: %dx%d", t, frame.shape[1], frame.shape[0])

    annotated = draw_regions(frame, config.HUD_REGIONS)
    out_path = config.NORMALIZED_DIR / f"{video_path.stem}_hud_t{t:.0f}.png"
    ensure_dir(out_path)
    cv2.imwrite(str(out_path), annotated)
    logger.info("Anteprima HUD salvata -> %s", out_path)

    if args.ocr:
        try:
            readings = run_ocr(frame, config.HUD_REGIONS)
            logger.info("=== Letture OCR ===")
            for name, text in readings.items():
                logger.info("  %-20s : %r", name, text)
        except ImportError:
            logger.warning("EasyOCR non installato: salto l'OCR. (pip install easyocr)")

    logger.info(
        "Apri l'immagine e, se le box non combaciano, correggi "
        "config.HUD_REGIONS (coordinate normalizzate sul frame normalizzato)."
    )


if __name__ == "__main__":
    main()
