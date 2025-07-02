# 🚇 AI-Powered Metro Surveillance System

This project is an advanced real-time surveillance solution for metro trains using computer vision and AI. It uses multiple YOLOv8-based modules to monitor onboard incidents such as:

- 🚶‍♂️ Fall detection
- 🔥 Fire & smoke detection
- 🎒 Unattended object detection
- 🥊 Violence detection
- 🚦 Overcrowding analysis

All detections are logged and sent to a central Flask-based API for storage, analysis, and alerting.

---

## 📁 Modules Overview

### 1. `fall_detection_final.py`
Detects if a person remains in a fallen state for more than 15 seconds.

- **Model**: YOLOv8 (`fall_det_1.pt`)
- **Input**: Pre-recorded video stream (`video_2.avi`)
- **Logging**: Sends a `Falling Person` alert to the central API
- **Env variables**: `API_URL`, `VIDEO_STREAM_URL`

---

### 2. `fire_smoke_detection_final_c.py`
Identifies persistent fire or smoke with cooldowns and history tracking.

- **Model**: YOLOv8 (`best.pt`)
- **Classes**: `"Fire"` and `"Smoke"`
- **Persistence logic**: Requires consistent detection over frames
- **Storage**: Logs to `incidents.json` and POSTs to API

---

### 3. `object_detection_final.py`
Detects unattended objects like bags that are far from any person.

- **Classes Tracked**: `"suitcase"`, `"backpack"`, `"handbag"`
- **Person proximity check**: Based on Euclidean distance and bounding box overlap
- **Cooldown**: Prevents redundant logging within a 5-minute window

---

### 4. `violence_detection.py`
Combines YOLO and MediaPipe Pose to detect raised arms suggesting potential violence.

- **Model**: YOLOv8 + MediaPipe
- **Condition**: Arms raised above 160° angle
- **Event**: Logs `"Fight Inside Coach"` incident

---

### 5. `metro_surveillance.py`
Estimates crowd density inside the coach using YOLOv8 pose model.

- **Crowd status**:
  - <10: ✅ Safe
  - 10–34: ⚠️ Moderate
  - 35–44: 🚨 High
  - 45+: 🔴 Overcrowded
- **Logs**: Crowd count and status to central web API

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO (`pip install ultralytics`)
- `mediapipe`
- `python-dotenv`
- Flask server (for incident logging)

---

## ⚙️ Usage

Update `.env`:
```env
API_URL=http://localhost:5000/log_incident
VIDEO_STREAM_URL=video_2.avi
```


## 🧠 Future Enhancements

- Integration with SMS or Email services
- Multi-camera coordination for full metro coverage
- ML-based violence classification using sequence modeling (e.g., LSTM)

---

## 👨‍💻 Authors & Contributors
- Adithya Anand
- Mitul Chitkara
- Josh Ethan N
