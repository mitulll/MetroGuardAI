import os
import time
import requests
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
THRESHOLD = 15
COOLDOWN = 300
API_URL = os.getenv("API_URL", "http://localhost:5000/log_incident")
VIDEO_STREAM_URL = os.getenv("VIDEO_STREAM_URL", "object.mp4")
TARGET_CLASSES = {'suitcase', 'backpack', 'handbag'}
PERSON_CLASS = 'person'
PROXIMITY_THRESHOLD = 150
MAX_RETRIES = 3
RETRY_DELAY = 2
DISPLAY = False  # 👈 Toggle display on/off
DISPLAY = os.getenv("DISPLAY","False")
def setup_logger():
    logger = logging.getLogger("unattended_object")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

class ObjectTracker:
    def __init__(self):
        self.active_detections = {}
        self.last_incident_time = 0
        self.logged_incidents = set()

    def _boxes_overlap(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        return intersection > (0.2 * min(box1_area, box2_area))

    def _get_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _is_near_person(self, obj_box, person_boxes):
        obj_center = self._get_center(obj_box)
        for person_box in person_boxes:
            person_center = self._get_center(person_box)
            distance = np.sqrt((obj_center[0] - person_center[0])**2 + 
                               (obj_center[1] - person_center[1])**2)
            if distance < PROXIMITY_THRESHOLD or self._boxes_overlap(obj_box, person_box):
                return True
        return False

    def process_frame(self, results, frame_width, frame_height):
        current_time = time.time()
        person_boxes = []
        object_detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            box_coords = box.xyxy[0].tolist()

            if cls_name == PERSON_CLASS:
                person_boxes.append(box_coords)
            elif cls_name in TARGET_CLASSES:
                object_detections.append((cls_name, box_coords))

        for cls_name, box_coords in object_detections:
            object_id = "%s_%d_%d" % (cls_name, box_coords[0], box_coords[1])
            if current_time - self.last_incident_time < COOLDOWN:
                continue

            is_attended = self._is_near_person(box_coords, person_boxes)
            if object_id not in self.active_detections:
                self.active_detections[object_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'attended': is_attended,
                    'logged': False
                }
                continue

            self.active_detections[object_id]['last_seen'] = current_time
            self.active_detections[object_id]['attended'] = is_attended

            if (not self.active_detections[object_id]['logged'] and 
                not is_attended and
                current_time - self.active_detections[object_id]['first_seen'] >= THRESHOLD):
                self._log_incident(object_id, "1023", "4", "Unattended %s" % cls_name, current_time)

        self._clean_expired_detections(current_time)

    def _log_incident(self, object_id, train, coach, incident_type, timestamp):
        incident_key = "%s_%s_%s" % (object_id, train, coach)
        if incident_key in self.logged_incidents:
            return

        logger.info("Incident detected: %s (Object: %s)" % (incident_type, object_id))
        data = {
            "train_number": train,
            "coach_number": coach,
            "incident_type": incident_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(API_URL, json=data, timeout=5)
                if response.status_code == 200:
                    logger.info("Incident logged successfully")
                    self.active_detections[object_id]['logged'] = True
                    self.last_incident_time = timestamp
                    self.logged_incidents.add(incident_key)
                    return
                logger.warning("Attempt %d: Failed to log incident" % (attempt+1))
            except Exception as e:
                logger.error("Attempt %d: Error: %s" % (attempt+1, str(e)))
            time.sleep(RETRY_DELAY)

    def _clean_expired_detections(self, current_time):
        expired = [
            k for k, v in self.active_detections.items()
            if current_time - v['last_seen'] > THRESHOLD * 2
        ]
        for key in expired:
            del self.active_detections[key]

        if current_time - self.last_incident_time > COOLDOWN * 2:
            self.logged_incidents.clear()

def main():
    tracker = ObjectTracker()
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO_STREAM_URL)

    if not cap.isOpened():
        logger.error("Failed to open video stream.")
        return

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, verbose=False)
            tracker.process_frame(results, frame.shape[1], frame.shape[0])

            if DISPLAY:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = results[0].names[cls_id]
                    if cls_name in TARGET_CLASSES or cls_name == PERSON_CLASS:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = (0, 255, 0) if cls_name == PERSON_CLASS else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        logger.error("Error: %s" % str(e))
    finally:
        cap.release()
        if DISPLAY:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()