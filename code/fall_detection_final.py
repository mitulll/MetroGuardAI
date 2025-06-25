from ultralytics import YOLO
import cv2
import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask server URL (replace with your laptop IP)
#API_URL = "http://192.168.1.25:5000/log_incident"  # ← update IP
API_URL = os.getenv("API_URL", "http://localhost:5000/log_incident")
VIDEO_STREAM_URL = os.getenv("VIDEO_STREAM_URL", "video_2.avi")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load YOLO model
model = YOLO("fall_det_1.pt")

# Video input
video_path = "video_2.avi"
cap = cv2.VideoCapture(VIDEO_STREAM_URL)

# Detection parameters
THRESHOLD = 5  # seconds
RESIZE_DIM = (320, 240)
FRAME_SKIP = 3   # Run detection every 3rd frame
DISPLAY = False  # Toggle for GUI (cv2.imshow)
DISPLAY = os.getenv("DISPLAY","False")
fall_detections = {}
frame_counter = 0

def log_incident(train_number, coach_number, incident_type):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data = {
        "train_number": train_number,
        "coach_number": coach_number,
        "incident_type": incident_type,
        "timestamp": timestamp
    }

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            print(f"✅ Incident logged: {incident_type}")
        else:
            print("❌ Failed to log incident.")
    except Exception as e:
        print(f"🚨 Error sending incident: {e}")

# Processing loop
while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, RESIZE_DIM)
    frame_counter += 1

    # Skip detection on some frames
    if frame_counter % FRAME_SKIP != 0:
        if DISPLAY:
            cv2.imshow("YOLOv8 Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        continue

    # Run YOLO detection
    results = model.track(frame, persist=True, conf=0.5)
    annotated_frame = results[0].plot()

    for detection in results[0].boxes.data:
        obj_id = int(detection[4])  # Object ID

        if obj_id not in fall_detections:
            fall_detections[obj_id] = time.time()
        elif isinstance(fall_detections[obj_id], float):
            duration = time.time() - fall_detections[obj_id]
            if duration >= THRESHOLD:
                if fall_detections[obj_id] != "logged":
                    print("🚨 Fall detected for 15s. Logging incident...")
                    log_incident("1023", "4", "Falling Person")
                    fall_detections[obj_id] = "logged"

    # Optional display
    if DISPLAY:
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Optional: show FPS
    fps = 1 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

# Cleanup
cap.release()
if DISPLAY:
    cv2.destroyAllWindows()