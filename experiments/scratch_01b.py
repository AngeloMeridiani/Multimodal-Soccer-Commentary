import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from utils import get_frame_at

video_path = Path("dataset/possesso_palla_e_parata.mov")
frame = get_frame_at(video_path, 1.01)
h, w = frame.shape[:2]
rx1, ry1 = int(0.35 * w), int(0.82 * h)
rx2, ry2 = int(0.65 * w), int(0.98 * h)
radar = frame[ry1:ry2, rx1:rx2]
cv2.imwrite("outputs/study/radar_1_01.jpg", radar)
