import os
import cv2
import json
import math
import numpy as np
import pygame
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("🚀 Starting Nose Sandbox...")

# 1. Load Configurations
try:
    with open("led_config.json", "r") as f:
        led_cfg = json.load(f)
    CANVAS_W = int(led_cfg.get("width", 256))
    CANVAS_H = int(led_cfg.get("height", 256))
except Exception:
    CANVAS_W, CANVAS_H = 256, 256

try:
    with open("nose_config.json", "r") as f:
        nose_cfg = json.load(f)
    matrix = np.array(nose_cfg["homography_matrix"], dtype=np.float32)
    print("✅ Configs loaded successfully.")
except Exception as e:
    print("❌ Failed to load nose_config.json. Please run nose_calibration.py first!")
    exit()

# 2. Setup MediaPipe
model_path = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_facial_transformation_matrixes=True, # Need head pose
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.FaceLandmarker.create_from_options(options)

# 3. Init Camera and Pygame
cap = cv2.VideoCapture(0)
pygame.init()
OS_W, OS_H = 1920, 1080
try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except Exception:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)

# --- STATE VARIABLES ---
shapes = []
grabbed_shape = None  
grab_radius = max(30, int(CANVAS_W * 0.15))

smooth_x, smooth_y = CANVAS_W // 2, CANVAS_H // 2
alpha = 0.35 

print("Ready! Press 'C' to create, 'SPACE' to grab/drop, 'X' to delete. ESC to quit.")

running = True
while running:
    success, frame = cap.read()
    if not success: continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    # 2. CALCULATE NOSE POINTER
    if results.facial_transformation_matrixes:
        transform_matrix = results.facial_transformation_matrixes[0]
        R = transform_matrix[:3, :3] 
        
        head_yaw = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2)))
        head_pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))

        src_pt = np.array([[[head_yaw, head_pitch]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(src_pt, matrix)
        raw_x, raw_y = dst_pt[0][0][0], dst_pt[0][0][1]

        smooth_x = int(alpha * raw_x + (1 - alpha) * smooth_x)
        smooth_y = int(alpha * raw_y + (1 - alpha) * smooth_y)

    nose_x = max(0, min(CANVAS_W, smooth_x))
    nose_y = max(0, min(CANVAS_H, smooth_y))

    # 3. EVENT HANDLING (Keyboard UI)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            
            elif event.key == pygame.K_c:
                shapes.append({"x": nose_x, "y": nose_y, "color": (255, 105, 180), "radius": 20})
                
            elif event.key == pygame.K_x and grabbed_shape is None:
                for s in shapes:
                    if math.hypot(s["x"] - nose_x, s["y"] - nose_y) < grab_radius:
                        shapes.remove(s)
                        break
                        
            elif event.key == pygame.K_SPACE:
                if grabbed_shape is not None:
                    grabbed_shape = None
                else:
                    closest_shape = None
                    closest_dist = float('inf')
                    for s in shapes:
                        dist = math.hypot(s["x"] - nose_x, s["y"] - nose_y)
                        if dist < grab_radius and dist < closest_dist:
                            closest_dist = dist
                            closest_shape = s
                    if closest_shape: grabbed_shape = closest_shape

    # 4. UPDATE PHYSICS
    if grabbed_shape is not None:
        grabbed_shape["x"] = nose_x
        grabbed_shape["y"] = nose_y

    # 5. RENDER
    screen.fill((15, 15, 15))

    for s in shapes:
        x, y = int(s["x"]), int(s["y"])
        dist = math.hypot(x - nose_x, y - nose_y)
        is_hovered = (dist < grab_radius) and (grabbed_shape is None)
        
        if s == grabbed_shape:
            pygame.draw.circle(screen, s["color"], (x, y), s["radius"] + 5)
            pygame.draw.circle(screen, (255, 255, 255), (x, y), s["radius"] + 5, 3)
        elif is_hovered:
            pygame.draw.circle(screen, (100, 100, 100), (x, y), s["radius"] + 10)
            pygame.draw.circle(screen, s["color"], (x, y), s["radius"])
        else:
            pygame.draw.circle(screen, s["color"], (x, y), s["radius"])

    # Draw Nose Cursor (Orange to differentiate from the Cyan gaze cursor)
    pygame.draw.circle(screen, (255, 150, 0), (int(nose_x), int(nose_y)), 6, 1)

    pygame.display.flip()

cap.release()
pygame.quit()