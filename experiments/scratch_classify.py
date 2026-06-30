import cv2
import numpy as np
from src.config import apply_profile, select_profile, auto_detect_teams, TEAM_RADAR_COLORS
from src.utils import get_frame_at

video_path = "dataset/possesso_palla_e_parata.mov"
probe = get_frame_at(video_path, 0.0)
_, prof = select_profile(probe.shape[1], probe.shape[0], "auto")
apply_profile(prof)
auto_detect_teams(probe)

for t in np.arange(31.0, 35.5, 0.5):
    print(f"Reading frame at t={t}")
    frame = get_frame_at(video_path, t)
    if frame is None:
        print("Frame is None")
        break
    print("Frame read successfully")
