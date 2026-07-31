"""
nose_calibration.py

Description:
    Generates a homography matrix for 3D head-pose tracking.
    Reads MediaPipe's facial_transformation_matrixes to calculate the Pitch and Yaw 
    of the user's head, mapping these vectors to the 4 corners of the LED canvas.
    Outputs the matrix to 'nose_config.json' in the root directory.

Usage:
    Run standalone or via main.py. Point nose at red targets and press SPACE to lock.
    Requires 'led_config.json' and 'models/face_landmarker.task' in the root directory.
"""

import os
import cv2
import json
import numpy as np
import pygame
import urllib.request
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def run_calibration():
    print("🎬 Initializing 3D Nose-Pointing Calibrator...")

    # --- BULLETPROOF PATH RESOLUTION ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

    LED_CONFIG_PATH = os.path.join(ROOT_DIR, "led_config.json")
    NOSE_CONFIG_PATH = os.path.join(ROOT_DIR, "nose_config.json")
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "face_landmarker.task")

    # 1. Load Configurations (Including Offsets)
    try:
        with open(LED_CONFIG_PATH, "r") as f:
            led_cfg = json.load(f)
        CANVAS_W = int(led_cfg.get("width", 256))
        CANVAS_H = int(led_cfg.get("height", 256))
        OFFSET_X = int(led_cfg.get("offset_x", 0))
        OFFSET_Y = int(led_cfg.get("offset_y", 0))
    except Exception:
        print(f"⚠️ Could not find {LED_CONFIG_PATH}, using defaults.")
        CANVAS_W, CANVAS_H, OFFSET_X, OFFSET_Y = 256, 256, 0, 0

    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe model to {MODEL_PATH}...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    pygame.init()
    OS_W, OS_H = 1920, 1080

    try:
        screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
    except Exception:
        screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    # Create the virtual LED canvas
    canvas = pygame.Surface((CANVAS_W, CANVAS_H))

    padding = max(15, int(CANVAS_W * 0.1))
    targets_canvas = [
        (padding, padding),                        
        (CANVAS_W - padding, padding),             
        (CANVAS_W - padding, CANVAS_H - padding),  
        (padding, CANVAS_H - padding)              
    ]

    nose_points = []
    current_target = 0
    calibrating, testing = True, False
    matrix = None
    smooth_x, smooth_y = CANVAS_W // 2, CANVAS_H // 2
    alpha = 0.35 
    last_input_time = 0.0
    
    success_flag = False

    print("\n🎯 3D NOSE CALIBRATION STARTED")

    while calibrating or testing:
        success, frame = cap.read()
        if not success: continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
        
        nose_vector_x, nose_vector_y = 0.0, 0.0
        face_detected = False
        preview_frame = frame.copy()

        if results.facial_transformation_matrixes:
            transform_matrix = results.facial_transformation_matrixes[0]
            R = transform_matrix[:3, :3] 
            
            head_yaw = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2)))
            head_pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            nose_vector_x = head_yaw
            nose_vector_y = head_pitch
            face_detected = True

            cv2.putText(preview_frame, f"Yaw: {head_yaw:.1f} | Pitch: {head_pitch:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Desktop Camera Preview", cv2.flip(preview_frame, 1))

        current_time = time.time()
        if (current_time - last_input_time) > 0.5:
            cv_key = cv2.waitKey(1) & 0xFF
            if cv_key == 27: # ESC
                break
            elif cv_key == 32 and calibrating and face_detected: # SPACE
                last_input_time = current_time
                nose_points.append([nose_vector_x, nose_vector_y])
                current_target += 1
                if current_target >= 4:
                    pts_src = np.array(nose_points, dtype=np.float32)
                    pts_dst = np.array(targets_canvas, dtype=np.float32)
                    matrix, _ = cv2.findHomography(pts_src, pts_dst)
                    calibrating, testing = False, True
            elif cv_key == 32 and testing: 
                success_flag = True
                break
            elif cv_key in [ord('r'), ord('R')]: 
                # Reset
                nose_points = []
                current_target = 0
                calibrating, testing = True, False
                matrix = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): 
                    calibrating, testing = False, False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if calibrating and face_detected:
                        last_input_time = current_time
                        nose_points.append([nose_vector_x, nose_vector_y])
                        current_target += 1
                        if current_target >= 4:
                            pts_src = np.array(nose_points, dtype=np.float32)
                            pts_dst = np.array(targets_canvas, dtype=np.float32)
                            matrix, _ = cv2.findHomography(pts_src, pts_dst)
                            calibrating, testing = False, True
                    elif testing:
                        success_flag = True
                        calibrating, testing = False, False

        # --- RENDER TO VIRTUAL CANVAS ---
        canvas.fill((0, 0, 0))
        if calibrating:
            pygame.draw.circle(canvas, (255, 0, 0), targets_canvas[current_target], 8)
            pygame.draw.circle(canvas, (255, 255, 255), targets_canvas[current_target], 8, 1)
        elif testing and matrix is not None and face_detected:
            src_pt = np.array([[[nose_vector_x, nose_vector_y]]], dtype=np.float32)
            dst_pt = cv2.perspectiveTransform(src_pt, matrix)
            smooth_x = int(alpha * dst_pt[0][0][0] + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * dst_pt[0][0][1] + (1 - alpha) * smooth_y)
            pygame.draw.circle(canvas, (255, 150, 0), (max(0, min(CANVAS_W, smooth_x)), max(0, min(CANVAS_H, smooth_y))), 12, 1)

        # --- BLIT CANVAS TO SCREEN ---
        screen.fill((0, 0, 0)) # Ensure OS bounds are black
        screen.blit(canvas, (OFFSET_X, OFFSET_Y))
        pygame.display.flip()

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

    if success_flag and matrix is not None:
        with open(NOSE_CONFIG_PATH, "w") as f:
            json.dump({"homography_matrix": matrix.tolist(), "canvas_width": CANVAS_W, "canvas_height": CANVAS_H}, f, indent=4)
        print(f"\n🎉 CALIBRATION SAVED to {NOSE_CONFIG_PATH}!")

if __name__ == "__main__":
    run_calibration()