import cv2
import mediapipe as mp
import numpy as np
import time
import requests
from ultralytics import YOLO
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask server URL
API_URL = os.getenv("API_URL", "http://localhost:5000/log_incident")
VIDEO_STREAM_URL = os.getenv("VIDEO_STREAM_URL", "vio.mp4")

# Performance tuning
RESIZE_DIM = (416, 416)
FRAME_SKIP = 3
DISPLAY = False
DISPLAY = os.getenv("DISPLAY","False")

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video source
video_path = "vio.mp4"
cap = cv2.VideoCapture(video_path)

# Config
THRESHOLD = 5
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
FRAMES_THRESHOLD = int(FPS * THRESHOLD)
COOLDOWN_PERIOD = 10

# Persistent state tracking
INCIDENT_STATE_FILE = "violence_state.json"
if os.path.exists(INCIDENT_STATE_FILE):
    with open(INCIDENT_STATE_FILE, "r") as f:
        incident_states = json.load(f)
else:
    incident_states = {}

def save_incident_states():
    with open(INCIDENT_STATE_FILE, "w") as f:
        json.dump(incident_states, f)

def log_incident(train_number, coach_number, incident_type):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data = {
        "train_number": train_number,
        "coach_number": coach_number,
        "incident_type": incident_type,
        "timestamp": timestamp
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=data, timeout=5)
            if response.status_code == 200:
                print(f"✅ Incident logged: {incident_type} at {timestamp}")
                return True
            else:
                print(f"❌ Failed to log incident. Status: {response.status_code}")
        except Exception as e:
            print(f"⚠️ API request failed (Attempt {attempt + 1}/{max_retries}): {e}")
        time.sleep(2)

    return False

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Main video loop
frame_count = 0
violence_detected = False
last_logged_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No more frames. Exiting...")
        break

    frame_count += 1
    current_time = time.time()

    frame = cv2.resize(frame, RESIZE_DIM)

    # Skip frames for performance
    if frame_count % FRAME_SKIP != 0:
        if DISPLAY:
            cv2.imshow("Violence Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue

    # Convert to RGB for pose estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose estimation
    results = pose.process(rgb_frame)

    # YOLO detection
    detections = model(frame, verbose=False)[0].boxes.data
    print(f"🟢 Frame {frame_count}: {len(detections)} objects detected")

    violence_detected = False

    for box in detections:
        x1, y1, x2, y2, conf, cls = box.tolist()
        if int(cls) == 0:
            print(f"  - Person detected at: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if left_arm_angle > 160 or right_arm_angle > 160:
            violence_detected = True
            print("❗ Arm raised – possible aggression detected")

        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if violence_detected:
        if current_time - last_logged_time >= COOLDOWN_PERIOD:
            print("🚨 Violence detected! Logging incident...")
            if log_incident("1023", "5", "Fight Inside Coach"):
                last_logged_time = current_time
                incident_states["last_logged_time"] = current_time
                save_incident_states()

    if violence_detected and DISPLAY:
        cv2.putText(frame, "🚨 Violence Detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if DISPLAY:
        cv2.imshow("Violence Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if DISPLAY:
    cv2.destroyAllWindows()

save_incident_states()