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

print("🎬 Initializing 3D Nose-Pointing Calibrator...")

try:
    with open("led_config.json", "r") as f:
        led_cfg = json.load(f)
    CANVAS_W = int(led_cfg.get("width", 256))
    CANVAS_H = int(led_cfg.get("height", 256))
except Exception:
    CANVAS_W, CANVAS_H = 256, 256

model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_facial_transformation_matrixes=True, # Back to 3D tracking!
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

padding = max(15, int(CANVAS_W * 0.1))
targets_canvas = [
    (padding, padding),                        
    (CANVAS_W - padding, padding),             
    (CANVAS_W - padding, CANVAS_H - padding),  
    (padding, CANVAS_H - padding)              
]

def run_calibration_and_test():
    nose_points = []
    current_target = 0
    calibrating, testing = True, False
    matrix = None
    smooth_x, smooth_y = CANVAS_W // 2, CANVAS_H // 2
    alpha = 0.35 
    last_input_time = 0.0

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
            
            # The Rock-Solid 3D Matrix Math
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
            if cv_key == 27: return False, None
            elif cv_key == 32 and calibrating and face_detected:
                last_input_time = current_time
                nose_points.append([nose_vector_x, nose_vector_y])
                current_target += 1
                if current_target >= 4:
                    pts_src = np.array(nose_points, dtype=np.float32)
                    pts_dst = np.array(targets_canvas, dtype=np.float32)
                    matrix, _ = cv2.findHomography(pts_src, pts_dst)
                    calibrating, testing = False, True
            elif cv_key == 32 and testing: return True, matrix
            elif cv_key in [ord('r'), ord('R')]: return run_calibration_and_test()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): return False, None
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
                    elif testing: return True, matrix

        screen.fill((0, 0, 0))
        if calibrating:
            pygame.draw.circle(screen, (255, 0, 0), targets_canvas[current_target], 8)
            pygame.draw.circle(screen, (255, 255, 255), targets_canvas[current_target], 8, 1)
        elif testing and matrix is not None and face_detected:
            src_pt = np.array([[[nose_vector_x, nose_vector_y]]], dtype=np.float32)
            dst_pt = cv2.perspectiveTransform(src_pt, matrix)
            smooth_x = int(alpha * dst_pt[0][0][0] + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * dst_pt[0][0][1] + (1 - alpha) * smooth_y)
            pygame.draw.circle(screen, (255, 150, 0), (max(0, min(CANVAS_W, smooth_x)), max(0, min(CANVAS_H, smooth_y))), 12, 1)

        pygame.display.flip()

    return False, None

success, matrix = run_calibration_and_test()
cap.release()
cv2.destroyAllWindows()
pygame.quit()

if success and matrix is not None:
    with open("nose_config.json", "w") as f:
        json.dump({"homography_matrix": matrix.tolist(), "canvas_width": CANVAS_W, "canvas_height": CANVAS_H}, f, indent=4)
    print("\n🎉 CALIBRATION SAVED!")