"""
Demos/pose_mp_demo.py

Description:
    Modern MediaPipe Tasks API Pose Landmarker Demo.
    Tracks 33 3D body keypoints, calculates joint angles (elbows, knees),
    and estimates posture metrics in real-time.
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

from core.config import FACE_CAMERA_INDEX

print("🎬 Starting Modern MediaPipe Tasks API Pose Tracking Demo...")

# 1. Download Pose Landmarker Model if missing
POSE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "pose_landmarker.task")

if not os.path.exists(POSE_MODEL_PATH):
    print(f"⏳ Downloading pose_landmarker.task model to: {POSE_MODEL_PATH}...")
    os.makedirs(os.path.dirname(POSE_MODEL_PATH), exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, POSE_MODEL_PATH)
    print("✅ Download complete!")

# 2. Configure Pose Landmarker Options for Live Video
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)

# 3. Create Landmarker Instance
detector = vision.PoseLandmarker.create_from_options(options)

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

# Standard 33 Full-Body Pose Connections
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Right Arm
    (12, 14), (14, 16),
    # Left Arm
    (11, 13), (13, 15),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31)
]


def calculate_angle(a, b, c):
    """
    Calculates 2D angle (in degrees) at joint point 'b' between vectors (a->b) and (c->b).
    Points a, b, c are tuples (x, y).
    """
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


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

    # Detect Pose
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Flip frame horizontally for mirror view
    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    # Telemetry Panel Overlay (500px wide, 240px tall)
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (15, 15), (515, 255), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
    cv2.rectangle(display_frame, (15, 15), (515, 255), (0, 255, 255), 2)

    cv2.putText(display_frame, "BODY POSE TELEMETRY", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(display_frame, (30, 60), (500, 60), (70, 70, 70), 1)

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]

        # Draw Skeletal Lines
        for p1, p2 in POSE_CONNECTIONS:
            x1, y1 = int((1.0 - landmarks[p1].x) * w), int(landmarks[p1].y * h)
            x2, y2 = int((1.0 - landmarks[p2].x) * w), int(landmarks[p2].y * h)
            cv2.line(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw 33 Keypoint Nodes
        for lm in landmarks:
            cx, cy = int((1.0 - lm.x) * w), int(lm.y * h)
            cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1)

        # Helper to convert landmark index to mirrored pixel tuple
        def get_pt(idx):
            return (int((1.0 - landmarks[idx].x) * w), int(landmarks[idx].y * h))

        # --- CALCULATE JOINT ANGLES ---
        # 1. Right Arm Angle (Shoulder 12, Elbow 14, Wrist 16)
        r_elbow_angle = calculate_angle(get_pt(12), get_pt(14), get_pt(16))

        # 2. Left Arm Angle (Shoulder 11, Elbow 13, Wrist 15)
        l_elbow_angle = calculate_angle(get_pt(11), get_pt(13), get_pt(15))

        # 3. Right Leg Angle (Hip 24, Knee 26, Ankle 28)
        r_knee_angle = calculate_angle(get_pt(24), get_pt(26), get_pt(28))

        # 4. Left Leg Angle (Hip 23, Knee 25, Ankle 27)
        l_knee_angle = calculate_angle(get_pt(23), get_pt(25), get_pt(27))

        # Render Angle Text on Screen Joint Locations
        cv2.putText(display_frame, f"{int(r_elbow_angle)}deg", get_pt(14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(display_frame, f"{int(l_elbow_angle)}deg", get_pt(13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Render Data in Telemetry Panel
        cv2.putText(display_frame, f"Right Elbow Flex: {r_elbow_angle:.1f} deg", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
        cv2.putText(display_frame, f"Left  Elbow Flex: {l_elbow_angle:.1f} deg", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
        cv2.putText(display_frame, f"Right Knee Flex : {r_knee_angle:.1f} deg", (30, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
        cv2.putText(display_frame, f"Left  Knee Flex : {l_knee_angle:.1f} deg", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)

        # Posture Status
        posture = "Standing / Straight"
        if r_knee_angle < 120 or l_knee_angle < 120:
            posture = "Squatting / Bending 🧘"
        cv2.putText(display_frame, f"Posture: {posture}", (30, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    else:
        cv2.putText(display_frame, "No Pose Detected", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 2)

    cv2.imshow('MediaPipe Tasks - Pose Tracking Demo', display_frame)

    # Exit on 'ESC'
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Pose tracking demo closed cleanly.")