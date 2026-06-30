import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from utils import get_frame_at
import importlib
visual = importlib.import_module("01b_visual_analysis")
PossessionTracker = visual.PossessionTracker
YoloDetector = visual.YoloDetector
YOLO_CONFIDENCE = 0.5

tracker = PossessionTracker(initial_team="home")

for t in np.arange(1.0, 5.0, 1.0):
    frame = get_frame_at(Path("dataset/possesso_palla_e_parata.mov"), t)
    ball = tracker._find_ball(frame)
    if ball:
        team, conf, _ = tracker.update(frame, ball)
        print(f"t={t:.2f} | team={team} | conf={conf} | colors={tracker.team_colors}")
    else:
        print(f"t={t:.2f} | NO BALL")
