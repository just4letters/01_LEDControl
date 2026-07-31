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

print("🚀 Starting All-in-One Gaze Sandbox...")

# 1. Load Configurations
try:
    with open("led_config.json", "r") as f:
        led_cfg = json.load(f)
    CANVAS_W = int(led_cfg.get("width", 256))
    CANVAS_H = int(led_cfg.get("height", 256))
except Exception:
    CANVAS_W, CANVAS_H = 256, 256

try:
    with open("gaze_config.json", "r") as f:
        gaze_cfg = json.load(f)
    matrix = np.array(gaze_cfg["homography_matrix"], dtype=np.float32)
    print("✅ Configs loaded successfully.")
except Exception as e:
    print("❌ Failed to load gaze_config.json. Please run calibration first!")
    sys.exit()

# 2. Setup MediaPipe (Using the True 3D Blendshape tracking)
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("⏳ Downloading model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=True,
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
clock = pygame.time.Clock()

# --- STATE VARIABLES ---
shapes = []
grabbed_shape = None  
grab_radius = max(30, int(CANVAS_W * 0.15))

# Basic Exponential Smoothing (matches the calibrator!)
smooth_x, smooth_y = CANVAS_W // 2, CANVAS_H // 2
alpha = 0.35 

print("Ready! Press 'C' to create, 'SPACE' to grab/drop, 'X' to delete. ESC to quit.")

running = True
while running:
    # 1. READ CAMERA & AI
    success, frame = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    # 2. CALCULATE RAW GAZE
    if results.face_blendshapes:
        bs = {cat.category_name: cat.score for cat in results.face_blendshapes[0]}
        
        # 3D Eye Rotation Math
        r_yaw = bs.get('eyeLookOutRight', 0) - bs.get('eyeLookInRight', 0)
        l_yaw = bs.get('eyeLookInLeft', 0) - bs.get('eyeLookOutLeft', 0)
        eye_yaw = (r_yaw + l_yaw) / 2.0
        
        r_pitch = bs.get('eyeLookDownRight', 0) - bs.get('eyeLookUpRight', 0)
        l_pitch = bs.get('eyeLookDownLeft', 0) - bs.get('eyeLookUpLeft', 0)
        eye_pitch = (r_pitch + l_pitch) / 2.0

        # Matrix Transform
        src_pt = np.array([[[eye_yaw, eye_pitch]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(src_pt, matrix)
        raw_x, raw_y = dst_pt[0][0][0], dst_pt[0][0][1]

        # Basic Smoothing
        smooth_x = int(alpha * raw_x + (1 - alpha) * smooth_x)
        smooth_y = int(alpha * raw_y + (1 - alpha) * smooth_y)

    # Clamp to edges
    gaze_x = max(0, min(CANVAS_W, smooth_x))
    gaze_y = max(0, min(CANVAS_H, smooth_y))

    # 3. EVENT HANDLING (Keyboard UI)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            
            # Make Shape
            elif event.key == pygame.K_c:
                shapes.append({"x": gaze_x, "y": gaze_y, "color": (255, 105, 180), "radius": 20})
                
            # Delete Shape
            elif event.key == pygame.K_x and grabbed_shape is None:
                for s in shapes:
                    if math.hypot(s["x"] - gaze_x, s["y"] - gaze_y) < grab_radius:
                        shapes.remove(s)
                        break
                        
            # Grab / Drop
            elif event.key == pygame.K_SPACE:
                if grabbed_shape is not None:
                    grabbed_shape = None
                else:
                    closest_shape = None
                    closest_dist = float('inf')
                    for s in shapes:
                        dist = math.hypot(s["x"] - gaze_x, s["y"] - gaze_y)
                        if dist < grab_radius and dist < closest_dist:
                            closest_dist = dist
                            closest_shape = s
                    if closest_shape:
                        grabbed_shape = closest_shape

    # 4. UPDATE PHYSICS
    if grabbed_shape is not None:
        grabbed_shape["x"] = gaze_x
        grabbed_shape["y"] = gaze_y

    # 5. RENDER
    screen.fill((15, 15, 15))

    for s in shapes:
        x, y = int(s["x"]), int(s["y"])
        dist = math.hypot(x - gaze_x, y - gaze_y)
        is_hovered = (dist < grab_radius) and (grabbed_shape is None)
        
        if s == grabbed_shape:
            pygame.draw.circle(screen, s["color"], (x, y), s["radius"] + 5)
            pygame.draw.circle(screen, (255, 255, 255), (x, y), s["radius"] + 5, 3)
        elif is_hovered:
            pygame.draw.circle(screen, (100, 100, 100), (x, y), s["radius"] + 10)
            pygame.draw.circle(screen, s["color"], (x, y), s["radius"])
        else:
            pygame.draw.circle(screen, s["color"], (x, y), s["radius"])

    # Draw Gaze Cursor
    pygame.draw.circle(screen, (0, 255, 255), (int(gaze_x), int(gaze_y)), 6, 1)

    pygame.display.flip()
    
    # We remove clock.tick(60) so Pygame simply runs as fast as the webcam allows (~30fps).
    # This prevents artificial stuttering.

cap.release()
pygame.quit()