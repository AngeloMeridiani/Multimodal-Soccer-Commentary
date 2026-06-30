import cv2
import numpy as np

img = cv2.imread('radar_crop.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Calculate histogram of hues to find dominant non-green colors
# Field is mostly green, so we ignore Hue between 30 and 80.
mask_non_green = cv2.bitwise_or(cv2.inRange(hsv, (0, 50, 50), (35, 255, 255)), cv2.inRange(hsv, (85, 50, 50), (180, 255, 255)))
# White/gray/black also exist
mask_white = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
mask_all = cv2.bitwise_or(mask_non_green, mask_white)

pixels = img[mask_all > 0]
if len(pixels) > 0:
    # Kmeans to find 3 dominant colors (home, away, ball)
    pixels32 = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(pixels32, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    print("Dominant BGR colors of non-green pixels in radar:")
    for c in center:
        print(c)
else:
    print("No non-green pixels found.")
