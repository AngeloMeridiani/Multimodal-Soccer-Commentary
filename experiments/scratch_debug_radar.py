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
print(f"Found {len(contours)} contours")
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

print(f"Filtered to {len(objects)} objects")
for obj in objects:
    print(obj)
