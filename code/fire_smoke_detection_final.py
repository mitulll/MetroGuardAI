
import cv2
import time
import json
import os
import logging
import requests
from ultralytics import YOLO
from requests.exceptions import RequestException  # For better error handling
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# === USER CONFIGURATION ===
DISPLAY = False  # 👈 Toggle this to True or False as needed
DISPLAY = os.getenv("DISPLAY","False")
# --- Configuration ---
MODEL_PATH = "best.pt"
VIDEO_PATH = "fire_new.mp4"
API_URL = os.getenv("API_URL", "http://localhost:5000/log_incident")
VIDEO_STREAM_URL = os.getenv("VIDEO_STREAM_URL", "fire_new.mp4")
RESIZE_DIM = (320, 240)
PERSISTENCE_THRESHOLD = 5
COOLDOWN_FRAMES = 100
LOG_HISTORY_FRAMES = 10

# --- Logger Setup ---
def setup_logger():
    logger = logging.getLogger("fire4_pi_display_flag")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

logger = setup_logger()

class FireSmokeLogger:
    def __init__(self):
        self.frame_count = 0
        self.persistence = {}
        self.cooldowns = {}
        self.recent_logs = {}
        self.logged_this_frame = set()
        self.recent_saved = set()
        self.incident_history = set()
        self.existing_incidents = self.load_existing_incidents()

    def load_existing_incidents(self):
        file_path = "incidents.json"
        incidents = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    incidents = json.load(f)
                    for incident in incidents:
                        key = (incident["object_id"], incident["timestamp"])
                        self.incident_history.add(key)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load existing incidents: {e}. Starting fresh.")
        return incidents

    def start_frame(self):
        self.frame_count += 1
        self.logged_this_frame.clear()

    def update(self, object_id, cls_name, train="12345", coach="1"):
        if object_id in self.logged_this_frame:
            return
        if object_id in self.cooldowns and self.frame_count - self.cooldowns[object_id] < COOLDOWN_FRAMES:
            return
        if object_id in self.recent_logs and self.frame_count - self.recent_logs[object_id] < LOG_HISTORY_FRAMES:
            return

        self.persistence[object_id] = self.persistence.get(object_id, 0) + 1

        if self.persistence[object_id] >= PERSISTENCE_THRESHOLD:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            incident_key = (object_id, timestamp)
            if incident_key in self.incident_history:
                logger.info(f"⚠️ Already logged: {cls_name} at {object_id} ({timestamp})")
                return
            self.log_incident(object_id, cls_name, train, coach, timestamp)
            self.cooldowns[object_id] = self.frame_count
            self.recent_logs[object_id] = self.frame_count
            self.persistence[object_id] = 0
            self.logged_this_frame.add(object_id)
            self.incident_history.add(incident_key)

    def log_incident(self, object_id, cls_name, train, coach, timestamp):
        data = {
            "object_id": object_id,
            "train_number": train,
            "coach_number": coach,
            "incident_type": cls_name,
            "timestamp": timestamp
        }
        logger.info(f"🔥 {cls_name} detected at {object_id}")
        self.save_to_json(data)
        self.send_to_api(data)

    def save_to_json(self, data):
        file_path = "incidents.json"
        updated_incidents = self.existing_incidents.copy()
        log_key = (data["object_id"], data["timestamp"])
        if log_key in self.recent_saved:
            logger.info("⚠️ Already saved this incident recently. Skipping JSON write.")
            return
        updated_incidents.append(data)
        try:
            with open(file_path, "w") as f:
                json.dump(updated_incidents, f, indent=4)
            self.existing_incidents = updated_incidents
            self.recent_saved.add(log_key)
        except Exception as e:
            logger.error(f"Failed to save to JSON: {e}")

    def send_to_api(self, data):
        try:
            response = requests.post(API_URL, json=data, timeout=5)
            if response.status_code != 200:
                logger.warning(f"API post failed with status {response.status_code}")
        except Exception as e:
            logger.warning(f"API post failed: {e}")

def main():
    model = YOLO(MODEL_PATH)
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    logger_instance = FireSmokeLogger()

    if not cap.isOpened():
        print("❌ Error opening video.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        logger_instance.start_frame()
        resized = cv2.resize(frame, RESIZE_DIM)
        results = model(resized, verbose=False)

        fire_boxes = []
        smoke_boxes = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            if cls_name == "Fire":
                fire_boxes.append((box, cls_name))
            elif cls_name == "Smoke":
                smoke_boxes.append((box, cls_name))

        boxes_to_process = fire_boxes if fire_boxes else smoke_boxes

        for box, cls_name in boxes_to_process:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_id = f"{cls_name}_{x1}_{y1}"
            color = (0, 0, 255) if cls_name == "Fire" else (255, 165, 0)
            label = f"{cls_name}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            logger_instance.update(object_id, cls_name)

        if DISPLAY:
            cv2.imshow("Fire/Smoke Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if DISPLAY:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
