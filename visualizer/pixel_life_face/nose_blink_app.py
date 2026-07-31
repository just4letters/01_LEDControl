import os
import cv2
import json
import math
import random
import numpy as np
import pygame
import urllib.request
import collections
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("🚀 Starting Refined Swarm Sandbox...")

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
        matrix = np.array(json.load(f)["homography_matrix"], dtype=np.float32)
except Exception:
    print("❌ Run nose_calibration.py first!")
    exit()

# 2. Setup MediaPipe
model_path = 'face_landmarker.task'
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    num_faces=1,
    output_face_blendshapes=True,
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

# --- STATE VARIABLES ---
shapes = []
is_blinking = False
blink_threshold = 0.5 

HISTORY_SIZE = 8
history_x = collections.deque([CANVAS_W // 2]*HISTORY_SIZE, maxlen=HISTORY_SIZE)
history_y = collections.deque([CANVAS_H // 2]*HISTORY_SIZE, maxlen=HISTORY_SIZE)

smooth_radius = 0.5

# --- INIT SWARM RETICLE PARTICLES ---
swarm = []
# Doubled the particle count to 100
for _ in range(100):
    swarm.append({
        "angle": random.uniform(0, 2 * math.pi),
        # Doubled the orbital rotation speed
        "speed": random.uniform(-0.06, 0.06),       
        # Doubled the wobble (bounce) speed
        "wobble_speed": random.uniform(0.004, 0.012), 
        "phase": random.uniform(0, 2 * math.pi)       
    })
torus_thickness = 8 

running = True
while running:
    success, frame = cap.read()
    if not success: continue

    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    current_time_ms = pygame.time.get_ticks()
    current_blink_score = 0.0

    if results.facial_transformation_matrixes and results.face_blendshapes:
        bs = {cat.category_name: cat.score for cat in results.face_blendshapes[0]}
        current_blink_score = (bs.get('eyeBlinkLeft', 0) + bs.get('eyeBlinkRight', 0)) / 2.0
        
        # --- MOUTH SCALING (Deadzone to 80px Diameter) ---
        jaw_open = bs.get('jawOpen', 0.0)
        
        # Deadzone: Ignore the first bit of mouth opening so resting face = 1px diameter
        jaw_deadzone = 0.04
        if jaw_open <= jaw_deadzone:
            target_radius = 0.5
        else:
            # Map the remaining movement to a max radius of 40 (80px diameter)
            active_jaw = jaw_open - jaw_deadzone
            target_radius = min(40.0, 0.5 + (active_jaw * 60.0))

        smooth_radius = (0.3 * target_radius) + (0.7 * smooth_radius)

        # --- 3D HEAD MATRIX ---
        transform_matrix = results.facial_transformation_matrixes[0]
        R = transform_matrix[:3, :3] 
        head_yaw = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2)))
        head_pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))

        src_pt = np.array([[[head_yaw, head_pitch]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(src_pt, matrix)
        raw_x, raw_y = dst_pt[0][0][0], dst_pt[0][0][1]

        # BLINK FREEZE
        if current_blink_score < 0.2:
            history_x.append(raw_x)
            history_y.append(raw_y)

    nose_x = max(0, min(CANVAS_W, int(sum(history_x) / len(history_x))))
    nose_y = max(0, min(CANVAS_H, int(sum(history_y) / len(history_y))))

    # --- BLINK LOGIC ---
    if current_blink_score > blink_threshold and not is_blinking:
        is_blinking = True
        
        particles = []
        for _ in range(16):
            angle = random.uniform(0, 2 * math.pi)
            dist = smooth_radius + random.uniform(2, 6)
            ox, oy = math.cos(angle) * dist, math.sin(angle) * dist
            
            speed = random.uniform(1.5, 3.5)
            vx = -math.sin(angle) * speed + random.uniform(-0.5, 0.5)
            vy = math.cos(angle) * speed + random.uniform(-0.5, 0.5)
            particles.append([ox, oy, vx, vy])

        shapes.append({
            "x": nose_x, "y": nose_y, 
            "color": (255, 50, 150), 
            "radius": smooth_radius, 
            "created_at": current_time_ms,
            "particles": particles
        })
    elif current_blink_score < (blink_threshold - 0.1) and is_blinking:
        is_blinking = False

    # --- EVENT HANDLING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_x: shapes.clear()

    # --- RENDER ---
    screen.fill((0, 0, 0))

    # 1. Draw Dropped Shapes and Physics Particles
    for s in shapes:
        pulse = (math.sin((current_time_ms - s["created_at"]) * 0.004) + 1.0) / 2.0 
        intensity = 0.2 + (0.8 * pulse)
        r, g, b = int(s["color"][0]*intensity), int(s["color"][1]*intensity), int(s["color"][2]*intensity)
        
        if s["radius"] < 1.0:
            pygame.draw.rect(screen, (r, g, b), (int(s["x"]), int(s["y"]), 1, 1))
        else:
            pygame.draw.circle(screen, (r, g, b), (int(s["x"]), int(s["y"])), int(s["radius"]))

        for p in s["particles"]:
            ox, oy, vx, vy = p
            ox += vx
            oy += vy
            dist_to_center = math.hypot(ox, oy)
            
            if dist_to_center > 0:
                gx = -(ox / dist_to_center) * 0.15
                gy = -(oy / dist_to_center) * 0.15
                vx += gx
                vy += gy
            
            if dist_to_center < s["radius"] + 1:
                if dist_to_center == 0: dist_to_center = 1
                nx, ny = ox / dist_to_center, oy / dist_to_center
                dot = vx * nx + vy * ny
                if dot < 0:
                    vx -= 2 * dot * nx
                    vy -= 2 * dot * ny
                ox = nx * (s["radius"] + 1)
                oy = ny * (s["radius"] + 1)

            vx *= 0.98
            vy *= 0.98
            p[0], p[1], p[2], p[3] = ox, oy, vx, vy
            
            pygame.draw.rect(screen, (50, 50, 50), (int(s["x"] + ox), int(s["y"] + oy), 1, 1))

    # 2. Draw Torus Swarm Reticle (Half Brightness)
    # 255 -> 127 for white (blinking)
    # (0, 200, 255) -> (0, 100, 127) for cyan (idle)
    swarm_color = (127, 127, 127) if is_blinking else (0, 100, 127)
    
    for p in swarm:
        p["angle"] += p["speed"]
        wobble = ((math.sin(current_time_ms * p["wobble_speed"] + p["phase"]) + 1.0) / 2.0)
        current_offset = wobble * torus_thickness
        
        distance_from_center = smooth_radius + 2 + current_offset
        
        dx = int(nose_x + math.cos(p["angle"]) * distance_from_center)
        dy = int(nose_y + math.sin(p["angle"]) * distance_from_center)
        
        pygame.draw.rect(screen, swarm_color, (dx, dy, 1, 1))

    pygame.display.flip()

cap.release()
pygame.quit()