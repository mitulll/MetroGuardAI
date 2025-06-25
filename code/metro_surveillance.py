import cv2
import os  # Added missing import
from ultralytics import YOLO
import requests
from requests.exceptions import RequestException  # For better error handling
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Load YOLOv8 pose model
    model = YOLO("yolov8n-pose.pt")

    # Define overcrowding thresholds
    GREEN_THRESHOLD = 10
    YELLOW_THRESHOLD = 35
    RED_THRESHOLD = 45

    # Performance tuning
    RESIZE_DIM = (416, 416)
    FRAME_SKIP = 3
    DISPLAY = False
    DISPLAY = os.getenv("DISPLAY","False")
    

    # Web server API endpoint
    API_URL = os.getenv("API_URL2", "http://localhost:5000/update")
    VIDEO_STREAM_URL = os.getenv("VIDEO_STREAM_URL", "metro2.mp4")

    try:
        cap = cv2.VideoCapture(VIDEO_STREAM_URL)
        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, RESIZE_DIM)
            frame_counter += 1

            if frame_counter % FRAME_SKIP != 0:
                if DISPLAY:
                    cv2.imshow("Metro Passenger Density", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # Run inference
            results = model(frame, verbose=False)

            # Count people
            person_count = 0
            for box in results[0].boxes:
                if int(box.cls.cpu().numpy()) == 0:  # Class 0 = "person"
                    person_count += 1

            # Determine crowd status (now using all three thresholds)
            if person_count < GREEN_THRESHOLD:
                status = "Safe"
                color = (0, 255, 0)
            elif person_count < YELLOW_THRESHOLD:
                status = "Moderate Crowd"
                color = (0, 255, 255)
            elif person_count < RED_THRESHOLD:
                status = "High Crowd"
                color = (0, 165, 255)
            else:
                status = "Overcrowded"
                color = (0, 0, 255)

            print(f"Frame {frame_counter}: Detected {person_count} people → Status: {status}")

            # Send data to web server
            try:
                requests.post(API_URL, json={"count": person_count, "status": status}, timeout=2)
            except RequestException as e:
                print(f"⚠️  Web server error: {str(e)}")

            if DISPLAY:
                cv2.putText(frame, f"Passenger Count: {person_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                cv2.imshow("Metro Passenger Density", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if DISPLAY:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()