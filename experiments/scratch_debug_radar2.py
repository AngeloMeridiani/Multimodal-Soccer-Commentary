import cv2
import numpy as np

frame = cv2.imread("test_frame.png")
minimap_rect = (0.35, 0.82, 0.65, 0.98)
h, w = frame.shape[:2]
rx1, ry1 = int(minimap_rect[0] * w), int(minimap_rect[1] * h)
rx2, ry2 = int(minimap_rect[2] * w), int(minimap_rect[3] * h)
radar = frame[ry1:ry2, rx1:rx2]

hsv = cv2.cvtColor(radar, cv2.COLOR_BGR2HSV)
mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
mask_objects = cv2.bitwise_not(mask_green)

contours, _ = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
home_c = np.array([168, 220, 210])
away_c = np.array([89, 101, 142])
ball_c = np.array([0, 140, 255])
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 3 < area < 150:
        mask_cnt = np.zeros(mask_objects.shape, dtype=np.uint8)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        color = cv2.mean(radar, mask=mask_cnt)[:3]
        c = np.array(color)
        dh = np.linalg.norm(c - home_c)
        da = np.linalg.norm(c - away_c)
        db = np.linalg.norm(c - ball_c)
        print(f"Area {area:5.1f} | BGR {int(c[0]):3},{int(c[1]):3},{int(c[2]):3} | dh={dh:5.1f}, da={da:5.1f}, db={db:5.1f} | {'BALL' if db < min(dh, da) else 'HOME' if dh < da else 'AWAY'}")
