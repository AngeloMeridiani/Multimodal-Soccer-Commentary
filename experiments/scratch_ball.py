import cv2
import numpy as np
frame = cv2.imread("test_frame.png")
minimap_rect = (0.35, 0.82, 0.65, 0.98)
h, w = frame.shape[:2]
radar = frame[int(minimap_rect[1]*h):int(minimap_rect[3]*h), int(minimap_rect[0]*w):int(minimap_rect[2]*w)]

# Trova i pixel più "arancioni" (alto G e R, basso B)
# Oppure converti in HSV e cerca hue ~ 15-25, sat alta
hsv = cv2.cvtColor(radar, cv2.COLOR_BGR2HSV)
mask_orange = cv2.inRange(hsv, (10, 100, 150), (25, 255, 255))
res = cv2.bitwise_and(radar, radar, mask=mask_orange)
cv2.imwrite("test_orange.png", res)
cnts, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Orange contours: {len(cnts)}")
for c in cnts:
    print(f"Area: {cv2.contourArea(c)}, Pos: {cv2.boundingRect(c)}")
