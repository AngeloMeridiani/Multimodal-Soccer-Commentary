import cv2
import numpy as np
from src.config import apply_profile, select_profile, auto_detect_teams, TEAM_RADAR_COLORS
from src.utils import get_frame_at

video_path = "dataset/possesso_palla_e_parata.mov"
probe = get_frame_at(video_path, 0.0)
_, prof = select_profile(probe.shape[1], probe.shape[0], "auto")
apply_profile(prof)
auto_detect_teams(probe)

def color_dist(c1, c2): return np.linalg.norm(np.array(c1)-np.array(c2))

for t in np.arange(0.0, 5.0, 0.5):
    frame = get_frame_at(video_path, t)
    if frame is None: break
    h, w = frame.shape[:2]
    rx1, ry1 = int(0.35*w), int(0.82*h)
    rx2, ry2 = int(0.65*w), int(0.98*h)
    radar = frame[ry1:ry2, rx1:rx2]
    
    hsv = cv2.cvtColor(radar, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    mask_objects = cv2.bitwise_not(mask_green)
    contours, _ = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 3 < area < 150:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                mask_cnt = np.zeros(mask_objects.shape, dtype=np.uint8)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                color = cv2.mean(radar, mask=mask_cnt)[:3]
                objects.append({"pos": (cx, cy), "color": color, "area": area})
                
    if not objects: continue
    
    objects.sort(key=lambda o: o["area"])
    smallest = objects[0]
    
    print(f"t={t:.1f}: {len(objects)} objects. Smallest: area={smallest['area']:.1f}, color={smallest['color']}")
    for obj in objects:
        if obj['area'] < 10:
            print(f"  Small obj: area={obj['area']:.1f}, color={obj['color']}")

