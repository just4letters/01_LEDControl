"""
Demos/object_detection_mp_demo.py

Description:
    Modern MediaPipe Tasks API Object Detection Demo.
    Detects 80+ everyday object categories in real-time.
    Objects detected at >50% confidence persist in the telemetry HUD for 5 seconds.
"""

import os
import sys
import time

# --- ENVIRONMENT OVERRIDES (Clean Terminal Output) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# --- PATH & MODULE IMPORTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.config import FACE_CAMERA_INDEX

print("🎬 Starting Modern MediaPipe Tasks API Object Detection Demo (5s Memory)...")

# 1. Download Object Detector Model (EfficientDet-Lite0) if missing
OBJ_MODEL_PATH = os.path.join(ROOT_DIR, "models", "efficientdet_lite0.tflite")

if not os.path.exists(OBJ_MODEL_PATH):
    print(f"⏳ Downloading efficientdet_lite0.tflite model to: {OBJ_MODEL_PATH}...")
    os.makedirs(os.path.dirname(OBJ_MODEL_PATH), exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
    urllib.request.urlretrieve(url, OBJ_MODEL_PATH)
    print("✅ Download complete!")

# 2. Configure Object Detector Options for Live Video
base_options = python.BaseOptions(model_asset_path=OBJ_MODEL_PATH)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.35,                              # Minimum threshold for initial detection
    max_results=8,                                     # Detect up to 8 objects at once
    running_mode=vision.RunningMode.VIDEO
)

# 3. Create Detector Instance
detector = vision.ObjectDetector.create_from_options(options)

# 4. Open Webcam with Fallback
print(f"📷 Attempting to open camera at index {FACE_CAMERA_INDEX}...")
cap = cv2.VideoCapture(FACE_CAMERA_INDEX)

if not cap.isOpened():
    print(f"⚠️ Warning: Could not open camera at index {FACE_CAMERA_INDEX}. Falling back to default camera (index 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Fatal Error: Could not open camera at index 0 either. Check camera connections and permissions.")
        sys.exit(1)

print("✅ Camera initialized successfully. Press 'ESC' to exit.")

# Palette for bounding box drawing
COLOR_PALETTE = [
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 128, 0),  # Orange
    (255, 0, 255),  # Magenta
    (0, 165, 255),  # Deep Orange
    (255, 255, 0),  # Cyan
    (128, 0, 255),  # Purple
    (0, 255, 128)   # Bright Teal
]

# Memory dictionary to persist objects > 50% confidence for 5 seconds
# Format: { category_name: {"score": float, "last_seen": float, "color": tuple} }
tracked_memory = {}
HOLD_DURATION_SEC = 5.0

start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    now = time.time()

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms = int((now - start_time) * 1000)

    # Detect objects
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Flip frame for natural mirror view
    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    # Set of categories detected in the CURRENT frame
    current_frame_categories = set()

    # --- 1. PROCESS CURRENT FRAME DETECTIONS & UPDATE MEMORY ---
    if detection_result.detections:
        for idx, detection in enumerate(detection_result.detections):
            category = detection.categories[0]
            category_name = category.category_name
            score = category.score
            bbox = detection.bounding_box

            current_frame_categories.add(category_name)
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

            # Draw bounding box on active detections
            box_x = w - (bbox.origin_x + bbox.width)
            box_y = bbox.origin_y
            box_w = bbox.width
            box_h = bbox.height

            cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 3)

            # Draw active label tag
            label_text = f"{category_name.upper()} {score*100:.0f}%"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (box_x, box_y - text_h - 10), (box_x + text_w + 10, box_y), color, -1)
            cv2.putText(display_frame, label_text, (box_x + 5, box_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- 50% CONFIDENCE MEMORY RULE ---
            if score >= 0.50:
                tracked_memory[category_name] = {
                    "score": score,
                    "last_seen": now,
                    "color": color
                }

    # --- 2. PRUNE EXPIRED MEMORY (OLDER THAN 5 SECONDS) ---
    expired_keys = [k for k, data in tracked_memory.items() if (now - data["last_seen"]) > HOLD_DURATION_SEC]
    for k in expired_keys:
        del tracked_memory[k]

    # --- 3. DRAW DYNAMIC HUD TELEMETRY PANEL ---
    # Adjust overlay box height based on active memory count
    hud_height = max(180, 110 + (len(tracked_memory) * 32))
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (15, 15), (515, hud_height), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
    cv2.rectangle(display_frame, (15, 15), (515, hud_height), (0, 255, 255), 2)

    cv2.putText(display_frame, "OBJECT TELEMETRY (5s PERSISTENCE)", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.line(display_frame, (30, 60), (500, 60), (70, 70, 70), 1)

    if tracked_memory:
        cv2.putText(display_frame, f"Tracked Items (>=50%): {len(tracked_memory)}", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset = 120
        # Sort tracked items by most recently seen
        sorted_memory = sorted(tracked_memory.items(), key=lambda item: item[1]["last_seen"], reverse=True)

        for category_name, data in sorted_memory:
            elapsed = now - data["last_seen"]
            remaining = max(0.0, HOLD_DURATION_SEC - elapsed)
            is_active = category_name in current_frame_categories

            if is_active:
                status_str = f"LIVE ({data['score']*100:.0f}%)"
                text_color = data["color"]
            else:
                status_str = f"HELD ({remaining:.1f}s left)"
                text_color = (180, 180, 180)

            hud_line = f"• {category_name.title():<15} : {status_str}"
            cv2.putText(display_frame, hud_line, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_offset += 30

    else:
        cv2.putText(display_frame, "No items >= 50% in memory", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)

    cv2.imshow('MediaPipe Tasks - Object Detection Demo', display_frame)

    # Exit on 'ESC'
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Object detection demo closed cleanly.")