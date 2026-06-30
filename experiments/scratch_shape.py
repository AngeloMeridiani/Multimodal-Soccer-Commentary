import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from utils import get_frame_at

video_path = Path("dataset/possesso_palla_e_parata.mov")
frame = get_frame_at(video_path, 0.0)
h, w = frame.shape[:2]
rx1, ry1 = int(0.35 * w), int(0.82 * h)
rx2, ry2 = int(0.65 * w), int(0.98 * h)
radar = frame[ry1:ry2, rx1:rx2]
hsv = cv2.cvtColor(radar, cv2.COLOR_BGR2HSV)
mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
mask_objects = cv2.bitwise_not(mask_green)
contours, _ = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

shapes = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 3 < area < 150:
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius**2)
        ratio = area / circle_area if circle_area > 0 else 0
        
        mask_cnt = np.zeros(mask_objects.shape, dtype=np.uint8)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        mean_color = cv2.mean(radar, mask=mask_cnt)[:3]
        shapes.append({"area": area, "ratio": ratio, "color": mean_color})

shapes.sort(key=lambda s: s["area"])
if len(shapes) > 0: shapes = shapes[1:] # remove ball

for s in shapes:
    print(f"ratio: {s['ratio']:.3f}, area: {s['area']:.1f}, color: {s['color']}")

