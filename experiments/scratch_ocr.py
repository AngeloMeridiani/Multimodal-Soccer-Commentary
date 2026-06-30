import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path("src").resolve()))

import cv2
import easyocr
from utils import get_frame_at

video_path = Path("dataset/possesso_palla_e_parata.mov")
frame = get_frame_at(video_path, 0.0)

h, w = frame.shape[:2]
# y: 2% to 15%, x: 2% to 30%
x1, y1 = int(w * 0.02), int(h * 0.02)
x2, y2 = int(w * 0.30), int(h * 0.15)
crop = frame[y1:y2, x1:x2]

reader = easyocr.Reader(['en'])
results = reader.readtext(crop)
print("OCR sul tabellone:")
for res in results:
    print(res[1])
