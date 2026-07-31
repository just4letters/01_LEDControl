import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import os
import json

# --- 1. TRY TO LOAD YOUR CALIBRATION ---
homography_matrix = None
for config_file in ['led_config.json', 'saved_states.json']:
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                # Look for homography keys
                if isinstance(data, dict):
                    for key in ['homography_matrix', 'homography', 'matrix']:
                        if key in data:
                            homography_matrix = np.array(data[key], dtype=np.float32)
                            print(f"✅ Loaded matrix from {config_file}!")
                            break
        except Exception as e:
            pass

if homography_matrix is None:
    print("⚠️ No calibration matrix found. Running in RAW mode (no perspective mapping).")
    homography_matrix = np.eye(3, dtype=np.float32)

# --- 2. INITIALIZE MEDIAPIPE ---
MP_MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MP_MODEL_PATH):
    print("❌ Error: MediaPipe AI model missing (hand_landmarker.task).")
    exit()

base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = mp_vision.HandLandmarker.create_from_options(options)

# --- 3. INITIALIZE CAMERA ---
cap = cv2.VideoCapture(0) # Change to 1 if you want the external webcam

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

# Set to HD (we'll see if your Mac accepts this or reverts to default!)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 Camera Resolution Active: {cam_w}x{cam_h}")
print("🚀 Starting Debug Window. Click on the camera window and press 'q' to quit.")

# --- 4. VISUAL LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        print("❌ Error: Failed to read frame.")
        break

    # Flip horizontally for natural mirror behavior
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        index_finger = detection_result.hand_landmarks[0][8]
        
        # 1. Raw camera coordinates
        raw_x = int(index_finger.x * cam_w)
        raw_y = int(index_finger.y * cam_h)
        
        # Draw green circle on raw finger position
        cv2.circle(frame, (raw_x, raw_y), 12, (0, 255, 0), -1)
        cv2.putText(frame, f"Raw Cam: {raw_x}, {raw_y}", (raw_x + 20, raw_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 2. Scaled perspective coordinates (Simulated 640x480 mapping for homography)
        scale_x = index_finger.x * 640
        scale_y = index_finger.y * 480
        
        pt = np.array([[[scale_x, scale_y]]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(pt, homography_matrix)
        
        proj_x = int(warped_pt[0][0][0])
        proj_y = int(warped_pt[0][0][1])
        
        # Display the projection data on screen
        cv2.putText(frame, f"Projected LED Matrix Target: {proj_x}, {proj_y}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Webcam Tracking Debug - Press 'q' to exit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()