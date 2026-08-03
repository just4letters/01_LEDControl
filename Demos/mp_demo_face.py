"""
Demos/mp_demo.py

Description:
    Modern MediaPipe Tasks API Face Landmarker Demo.
    Displays 478 3D landmarks, high-visibility Blendshape meters, 
    3D Head Pose Angles (Pitch, Yaw, Roll), and a real-time Emotion Classifier.
"""

import os
import sys
import time
import math
import numpy as np

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

from core.config import FACE_MODEL_PATH, FACE_CAMERA_INDEX

print("🎬 Starting Modern MediaPipe Tasks API Demo with Large Data Panel...")

# 1. Download Model File if missing
model_path = FACE_MODEL_PATH

if not os.path.exists(model_path):
    print(f"⏳ Downloading face_landmarker.task model to: {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("✅ Download complete!")

# 2. Configure Options WITH Blendshapes and 3D Matrix enabled
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,                      # Enables 52 ARKit micro-expressions
    output_facial_transformation_matrixes=True,        # Enables 4x4 3D Head Pose Matrix
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO
)

# 3. Create Landmarker Instance
detector = vision.FaceLandmarker.create_from_options(options)

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


def rotation_matrix_to_euler_angles(R):
    """Calculates Pitch, Yaw, and Roll (in degrees) from a 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return math.degrees(x), math.degrees(y), math.degrees(z)


def classify_emotion(scores):
    """Derived emotion classification using raw blendshape thresholds."""
    jaw_open = scores.get("jawOpen", 0.0)
    eye_wide_l = scores.get("eyeWideLeft", 0.0)
    eye_wide_r = scores.get("eyeWideRight", 0.0)
    smile_l = scores.get("mouthSmileLeft", 0.0)
    smile_r = scores.get("mouthSmileRight", 0.0)
    brow_down_l = scores.get("browDownLeft", 0.0)
    brow_down_r = scores.get("browDownRight", 0.0)
    brow_up_l = scores.get("browOuterUpLeft", 0.0)
    brow_up_r = scores.get("browOuterUpRight", 0.0)
    nose_sneer = max(scores.get("noseSneerLeft", 0.0), scores.get("noseSneerRight", 0.0))
    pucker = scores.get("mouthPucker", 0.0)

    # 1. Surprise
    if jaw_open > 0.35 and (eye_wide_l > 0.25 or eye_wide_r > 0.25):
        return "Surprised / Shocked 😮", (0, 255, 255)

    # 2. Happiness
    avg_smile = (smile_l + smile_r) / 2.0
    if avg_smile > 0.35:
        return "Happy / Smiling 😊", (0, 255, 0)

    # 3. Disgust
    if nose_sneer > 0.35:
        return "Disgusted 😖", (0, 100, 255)

    # 4. Skeptical (Asymmetrical eyebrows)
    if abs(brow_up_l - brow_up_r) > 0.35 or abs(brow_down_l - brow_down_r) > 0.35:
        return "Skeptical / Curious 🤔", (255, 165, 0)

    # 5. Anger / Deep Focus
    if brow_down_l > 0.4 and brow_down_r > 0.4:
        return "Angry / Focused 😠", (0, 0, 255)

    # 6. Pucker / Kissy
    if pucker > 0.4:
        return "Pucker / Whistling 😗", (255, 190, 200)

    return "Neutral 😐", (200, 200, 200)


start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert frame BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Frame timestamp
    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # Detect face landmarks, blendshapes, and transformation matrices
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Flip frame horizontally for natural selfie view
    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    # --- 1. DRAW FACE LANDMARK MESH ---
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            for lm in face_landmarks:
                # Mirror X for flipped view
                cx = int((1.0 - lm.x) * w)
                cy = int(lm.y * h)
                cv2.circle(display_frame, (cx, cy), 1, (0, 255, 0), -1)

    # Enlarged semi-transparent overlay panel (Width: 500px, Height: 480px)
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (15, 15), (515, 495), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
    cv2.rectangle(display_frame, (15, 15), (515, 495), (0, 255, 255), 2)

    # --- 2. EMOTION CLASSIFIER PANEL ---
    if detection_result.face_blendshapes:
        blendshapes = detection_result.face_blendshapes[0]
        scores = {b.category_name: b.score for b in blendshapes}

        # Classify emotion state
        emotion_text, emotion_color = classify_emotion(scores)

        # Header: Detected Emotion
        cv2.putText(display_frame, "STATE:", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)
        cv2.putText(display_frame, emotion_text, (130, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, emotion_color, 2)

        cv2.line(display_frame, (30, 65), (500, 65), (70, 70, 70), 1)

        # Sub-header: Blendshape meters
        cv2.putText(display_frame, "FACIAL BLENDSHAPES", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        tracked = [
            ("Jaw Open", scores.get("jawOpen", 0.0)),
            ("Blink Left", scores.get("eyeBlinkLeft", 0.0)),
            ("Blink Right", scores.get("eyeBlinkRight", 0.0)),
            ("Smile Left", scores.get("mouthSmileLeft", 0.0)),
            ("Smile Right", scores.get("mouthSmileRight", 0.0)),
            ("Brow Down L", scores.get("browDownLeft", 0.0)),
            ("Pucker", scores.get("mouthPucker", 0.0)),
        ]

        y_offset = 130
        for name, val in tracked:
            # Larger text label
            text = f"{name:<12}: {val:.2f}"
            cv2.putText(display_frame, text, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Larger progress bar (Width: 180px, Height: 14px)
            bar_w = int(val * 180)
            cv2.rectangle(display_frame, (280, y_offset - 14), (280 + bar_w, y_offset + 2), (0, 255, 0), -1)
            cv2.rectangle(display_frame, (280, y_offset - 14), (460, y_offset + 2), (120, 120, 120), 1)
            y_offset += 32

    # --- 3. 3D HEAD POSE PANEL ---
    if detection_result.facial_transformation_matrixes:
        matrix_raw = detection_result.facial_transformation_matrixes[0]
        matrix_4x4 = np.array(matrix_raw).reshape(4, 4)

        R = matrix_4x4[:3, :3]
        pitch, yaw, roll = rotation_matrix_to_euler_angles(R)

        cv2.line(display_frame, (30, y_offset), (500, y_offset), (70, 70, 70), 1)

        cv2.putText(display_frame, "3D HEAD POSE", (30, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(display_frame, f"Pitch (Up/Down) : {pitch:+.1f} deg", (30, y_offset + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
        cv2.putText(display_frame, f"Yaw   (Left/Rgt): {yaw:+.1f} deg", (30, y_offset + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
        cv2.putText(display_frame, f"Roll  (Tilt)    : {roll:+.1f} deg", (30, y_offset + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    cv2.imshow('MediaPipe Tasks - Blendshapes & 3D Matrix', display_frame)

    # Exit on 'ESC'
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Demo closed cleanly.")