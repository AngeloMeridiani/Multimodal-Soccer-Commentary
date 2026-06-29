from inference_sdk import InferenceHTTPClient
import cv2

cap = cv2.VideoCapture("dataset/possesso_palla_e_parata.mov")
cap.set(cv2.CAP_PROP_POS_MSEC, 15000)
ret, frame = cap.read()
cv2.imwrite("test_frame.png", frame)
cap.release()
print("Frame estratto")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YsYDu03mt7GfKqIqg82f"
)
result = client.infer("test_frame.png", model_id="fifa-yolo-project/1")

for pred in result["predictions"]:
    print(f"  {pred['class']:12} conf={pred['confidence']:.2f}")