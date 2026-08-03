"""
Demos/hand_mp_demo.py

Description:
    Modern MediaPipe Tasks API Hand Landmarker Demo.
    Tracks 21 3D hand keypoints per hand, detects Left/Right handedness,
    measures pinch distance, and classifies real-time hand gestures.
"""

import os
import sys
import time
import math

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

from core.config import FACE_CAMERA_INDEX  # Uses main project camera setting

print("🎬 Starting Modern MediaPipe Tasks API Hand Tracking Demo...")

# 1. Download Hand Landmarker Model if missing
HAND_MODEL_PATH = os.path.join(ROOT_DIR, "models", "hand_landmarker.task")

if not os.path.exists(HAND_MODEL_PATH):
    print(f"⏳ Downloading hand_landmarker.task model to: {HAND_MODEL_PATH}...")
    os.makedirs(os.path.dirname(HAND_MODEL_PATH), exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, HAND_MODEL_PATH)
    print("✅ Download complete!")

# 2. Configure Hand Landmarker Options for Live Video
base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                                       # Track up to 2 hands simultaneously
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)

# 3. Create Landmarker Instance
detector = vision.HandLandmarker.create_from_options(options)

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

# Standard Hand Skeleton Connections (21 Landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm Base
]


def classify_hand_gesture(landmarks, handedness_label):
    """
    Analyzes finger extension & tip positions to classify gestures:
    Pinch, Open Palm, Fist, Victory/Peace, Thumbs Up, Pointing.
    """
    # Landmark indices for finger tips and PIP/MCP joints
    # 4: Thumb Tip, 8: Index Tip, 12: Middle Tip, 16: Ring Tip, 20: Pinky Tip
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Calculate Pinch Distance (Thumb Tip to Index Tip)
    pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    # Check which fingers are extended (Tip Y is higher/smaller than PIP joint Y)
    index_ext = index_tip.y < landmarks[6].y
    middle_ext = middle_tip.y < landmarks[10].y
    ring_ext = ring_tip.y < landmarks[14].y
    pinky_ext = pinky_tip.y < landmarks[18].y
    thumb_ext = thumb_tip.y < landmarks[2].y

    # 1. Pinch Gesture
    if pinch_dist < 0.06:
        return f"Pinch 👌 (Dist: {pinch_dist:.2f})", (0, 255, 255)

    # 2. Open Palm
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "Open Palm ✋", (0, 255, 0)

    # 3. Fist
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        return "Fist ✊", (0, 0, 255)

    # 4. Peace / Victory
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "Peace / Victory ✌️", (255, 0, 255)

    # 5. Pointing Finger
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "Pointing ☝️", (255, 255, 0)

    # 6. Thumbs Up
    if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "Thumbs Up 👍", (0, 255, 150)

    return "Tracking Hand 🖐️", (200, 200, 200)


start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert frame BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # Detect hand landmarks & handedness
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Flip frame for natural mirror view
    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    # Large dark overlay panel for telemetry
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (15, 15), (515, 220), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
    cv2.rectangle(display_frame, (15, 15), (515, 220), (0, 255, 255), 2)

    cv2.putText(display_frame, "HAND TRACKING TELEMETRY", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(display_frame, (30, 60), (500, 60), (70, 70, 70), 1)

    if detection_result.hand_landmarks:
        num_detected = len(detection_result.hand_landmarks)
        
        for idx in range(num_detected):
            hand_landmarks = detection_result.hand_landmarks[idx]
            
            # Extract Handedness (Left/Right) & Confidence Score
            handedness_info = detection_result.handedness[idx][0]
            # When frame is mirrored via cv2.flip, swap Left/Right label for user perception
            raw_label = handedness_info.category_name
            label = "Right" if raw_label == "Left" else "Left"
            score = handedness_info.score

            # Classify Gesture
            gesture_text, gesture_color = classify_hand_gesture(hand_landmarks, label)

            # Draw Hand Skeleton Connections
            for p1, p2 in HAND_CONNECTIONS:
                x1, y1 = int((1.0 - hand_landmarks[p1].x) * w), int(hand_landmarks[p1].y * h)
                x2, y2 = int((1.0 - hand_landmarks[p2].x) * w), int(hand_landmarks[p2].y * h)
                cv2.line(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Draw 21 Landmark Nodes
            for lm in hand_landmarks:
                cx, cy = int((1.0 - lm.x) * w), int(lm.y * h)
                cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.circle(display_frame, (cx, cy), 2, (0, 0, 0), -1)

            # Highlight Index & Thumb Tips for Pinching Visual
            t_x, t_y = int((1.0 - hand_landmarks[4].x) * w), int(hand_landmarks[4].y * h)
            i_x, i_y = int((1.0 - hand_landmarks[8].x) * w), int(hand_landmarks[8].y * h)
            cv2.line(display_frame, (t_x, t_y), (i_x, i_y), (0, 255, 255), 2)

            # Print Info Panel for each detected hand
            y_pos = 95 + (idx * 60)
            cv2.putText(display_frame, f"Hand {idx+1} [{label} ({score*100:.0f}%)]:", (30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
            cv2.putText(display_frame, gesture_text, (260, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)

    else:
        cv2.putText(display_frame, "No Hands Detected", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 2)

    cv2.imshow('MediaPipe Tasks - Hand Tracking Demo', display_frame)

    # Exit on 'ESC'
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Hand tracking demo closed cleanly.")