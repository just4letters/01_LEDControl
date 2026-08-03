"""
pixel_life_face/app.py

Description:
    Main interactive application for the LED canvas. 
    Combines 3D head pose tracking, facial blendshapes, and blink detection 
    to drive a 2D physics-based "Black Hole" UI.
"""

print("🚀 Starting Black Hole Physics Sandbox...")

import os
import sys

# 1. Point Python to the root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 2. Import global paths & core services
from core.config import LED_CONFIG_PATH, NOSE_CONFIG_PATH
from core.vision import FaceTracker
from core.voice import VoiceListener

import json
import math
import random
import queue
import collections
import numpy as np
import pygame

# --- LOAD CONFIGURATIONS ---
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

try:
    with open(NOSE_CONFIG_PATH, "r") as f:
        matrix = np.array(json.load(f)["homography_matrix"], dtype=np.float32)
except Exception:
    print(f"❌ Run nose_calibration.py first! Could not find: {NOSE_CONFIG_PATH}")
    sys.exit()


# =========================================================================
# 🧠 INTERACTION MODULE (Decoupled Math)
# =========================================================================
def apply_face_interactions(
    raw_x, raw_y, blink_score, jaw_open,
    history_x, history_y, smooth_radius, 
    is_blinking, shapes, current_time_ms,
    reticle_scale
):
    # 1. Position & Blink-Freeze Logic
    if blink_score < 0.2:
        history_x.append(raw_x)
        history_y.append(raw_y)

    nose_x = max(0, min(CANVAS_W, int(sum(history_x) / len(history_x))))
    nose_y = max(0, min(CANVAS_H, int(sum(history_y) / len(history_y))))

    # 2. Corrected Linear Mouth Scaling Logic (Now up to 150)
    if jaw_open <= 0.1:
        target_radius = 0.5 
    else:
        normalized_jaw = min(1.0, (jaw_open - 0.1) / 0.8)
        target_radius = 0.5 + (149.5 * normalized_jaw)  # Scales to 150 max

    # HEAVIER SMOOTHING: Absorbs AI micro-jitters
    new_smooth_radius = (0.1 * target_radius) + (0.9 * smooth_radius)

    # 3. Blink Trigger (Drop a shape)
    blink_threshold = 0.5
    new_is_blinking = is_blinking

    if blink_score > blink_threshold and not is_blinking:
        new_is_blinking = True
        
        # ONLY create objects if the reticle is mostly visible
        if reticle_scale > 0.5: 
            shape_particles = []
            for _ in range(16):
                angle = random.uniform(0, 2 * math.pi)
                dist = new_smooth_radius + random.uniform(2, 6)
                ox, oy = math.cos(angle) * dist, math.sin(angle) * dist
                
                speed = random.uniform(1.5, 3.5)
                vx = -math.sin(angle) * speed + random.uniform(-0.5, 0.5)
                vy = math.cos(angle) * speed + random.uniform(-0.5, 0.5)
                shape_particles.append([ox, oy, vx, vy])

            shapes.append({
                "x": nose_x, "y": nose_y, 
                "color": (255, 50, 150), 
                "radius": new_smooth_radius, 
                "created_at": current_time_ms,
                "particles": shape_particles
            })
        
    elif blink_score < (blink_threshold - 0.1) and is_blinking:
        new_is_blinking = False

    return nose_x, nose_y, new_smooth_radius, new_is_blinking

# =========================================================================


# --- QUEUES & VOICE SETUP ---
sentence_queue = queue.Queue()
action_queue = queue.Queue()

voice_listener = VoiceListener(sentence_queue, action_queue)
voice_listener.start(prompt_message="[VOICE] Black Hole Sandbox listening (Say 'shut down' or 'clear screen')")

# --- INITIALIZE TRACKER & PYGAME ---
tracker = FaceTracker(matrix)
tracker.start()

pygame.init()
OS_W, OS_H = 1920, 1080
try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except Exception:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

canvas = pygame.Surface((CANVAS_W, CANVAS_H))

# --- STATE VARIABLES ---
shapes = []
is_blinking = False

HISTORY_SIZE = 8
history_x = collections.deque([CANVAS_W // 2]*HISTORY_SIZE, maxlen=HISTORY_SIZE)
history_y = collections.deque([CANVAS_H // 2]*HISTORY_SIZE, maxlen=HISTORY_SIZE)

smooth_radius = 0.5
reticle_particles = [] 

# IDLE TRACKING VARIABLES
last_nose_x, last_nose_y = CANVAS_W // 2, CANVAS_H // 2
last_move_time = 0
reticle_scale = 1.0  # 1.0 = fully visible, 0.0 = sucked into center

# --- MAIN LOOP ---
running = True
while running:
    # --- VOICE & ACTION QUEUE ---
    while not action_queue.empty():
        payload = action_queue.get()
        action = payload.get("action")
        if action == "quit":
            running = False
        elif action == "clear_screen":
            print("🧹 Voice Command: Clearing screen...")
            shapes.clear()

    # 1. Fetch raw state from Vision Engine
    raw_x, raw_y, current_blink_score, jaw_open = tracker.get_state()
    current_time_ms = pygame.time.get_ticks()
    
    if last_move_time == 0:
        last_move_time = current_time_ms

    # 2. RUN DECOUPLED FACE MATH
    nose_x, nose_y, smooth_radius, is_blinking = apply_face_interactions(
        raw_x, raw_y, current_blink_score, jaw_open,
        history_x, history_y, smooth_radius, 
        is_blinking, shapes, current_time_ms,
        reticle_scale
    )

    # --- 3. IDLE SUCK-IN LOGIC ---
    # User-adjusted sensitivity: ONLY X-Axis, > 5.0 pixels
    dist_moved_x = abs(nose_x - last_nose_x)
    if dist_moved_x > 5.0:
        last_move_time = current_time_ms
        last_nose_x = nose_x
        last_nose_y = nose_y

    is_idle = (current_time_ms - last_move_time) > 3000  # 3 seconds of no movement

    if is_idle:
        # Sucks in very quickly over ~12 frames
        reticle_scale = max(0.0, reticle_scale - 0.08)
    else:
        # Pops back out instantly when movement is detected
        reticle_scale = min(1.0, reticle_scale + 0.15)


    # --- EVENT HANDLING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): 
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_x: 
            shapes.clear()

    # --- RENDER TO VIRTUAL CANVAS ---
    canvas.fill((0, 0, 0))

    # Draw Dropped Shapes
    for s in shapes:
        pulse = (math.sin((current_time_ms - s["created_at"]) * 0.004) + 1.0) / 2.0 
        intensity = 0.2 + (0.8 * pulse)
        r, g, b = int(s["color"][0]*intensity), int(s["color"][1]*intensity), int(s["color"][2]*intensity)
        
        if s["radius"] <= 1.0:
            pygame.draw.rect(canvas, (r, g, b), (int(s["x"]), int(s["y"]), 1, 1))
        else:
            pygame.draw.circle(canvas, (r, g, b), (int(s["x"]), int(s["y"])), int(s["radius"]))

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
            
            pygame.draw.rect(canvas, (50, 50, 50), (int(s["x"] + ox), int(s["y"] + oy), 1, 1))

    # --- DRAW BLACK HOLE RETICLE ---
    dim_color = (0, 80, 120) if not is_blinking else (100, 100, 100)
    
    # Scale radiuses based on idle animation
    display_outer = (smooth_radius + 24) * reticle_scale
    display_inner = smooth_radius * reticle_scale

    # 1. THE CENTER DOT (Always visible and pulsing, regardless of idle state)
    pulse = (math.sin(current_time_ms * 0.008) + 1.0) / 2.0
    c_val = int(100 + 155 * pulse)  # Notice we removed `* reticle_scale` here
    pygame.draw.rect(canvas, (c_val, c_val, c_val), (int(nose_x), int(nose_y), 1, 1))


    # 2. THE OUTER RINGS AND PARTICLES (Only if not fully idle)
    if reticle_scale > 0.01:
        # Moderate Outer Spin
        for i in range(36):
            a = (i / 36.0) * math.pi * 2 + (current_time_ms * 0.002)
            dx = int(nose_x + math.cos(a) * display_outer)
            dy = int(nose_y + math.sin(a) * display_outer)
            pygame.draw.rect(canvas, dim_color, (dx, dy, 1, 1))

        # PREVENT GLITCHY CENTER RENDERING
        if display_inner > 3.0:
            for i in range(24):
                a = (i / 24.0) * math.pi * 2 - (current_time_ms * 0.0025)
                dx = int(nose_x + math.cos(a) * display_inner)
                dy = int(nose_y + math.sin(a) * display_inner)
                pygame.draw.rect(canvas, dim_color, (dx, dy, 1, 1))

        # Spawn reticle particles (Using the dynamically shrinking outer radius)
        for _ in range(4):
            a = random.uniform(0, 2 * math.pi)
            px = nose_x + math.cos(a) * display_outer
            py = nose_y + math.sin(a) * display_outer
            
            tangent = random.uniform(1.5, 3.0)  
            inward = random.uniform(0.1, 0.5)
            vx = -math.sin(a) * tangent - math.cos(a) * inward
            vy = math.cos(a) * tangent - math.sin(a) * inward

            reticle_particles.append({
                "x": px, "y": py,
                "vx": vx, "vy": vy,
                "passed_inner": False
            })

    # Alive particles continue processing even when fully collapsed
    alive_particles = []
    for p in reticle_particles:
        dx = nose_x - p["x"]
        dy = nose_y - p["y"]
        dist = math.hypot(dx, dy)

        # STRONGER GRAVITY PULL
        if dist > 0:
            force = 45.0 / (dist + 5.0)  
            p["vx"] += (dx / dist) * force * 0.2
            p["vy"] += (dy / dist) * force * 0.2

        p["vx"] *= 0.96
        p["vy"] *= 0.96
        p["x"] += p["vx"]
        p["y"] += p["vy"]

        if dist < smooth_radius and not p["passed_inner"]:
            p["passed_inner"] = True
            if random.random() < 0.65:
                continue 
            else:
                p["vx"] *= 0.25 
                p["vy"] *= 0.25 

        # Die if too close to center
        if dist < 2.0:
            continue

        p_color = (255, 255, 255) if p["passed_inner"] else (0, 160, 220)
        if is_blinking: p_color = (150, 150, 150)
        
        # Fade out particles as the reticle collapses to hide stragglers
        if reticle_scale < 1.0:
            p_color = (int(p_color[0] * reticle_scale), int(p_color[1] * reticle_scale), int(p_color[2] * reticle_scale))
            
        pygame.draw.rect(canvas, p_color, (int(p["x"]), int(p["y"]), 1, 1))
        alive_particles.append(p)

    reticle_particles = alive_particles

    # --- BLIT CANVAS TO SCREEN ---
    screen.fill((0, 0, 0))
    screen.blit(canvas, (OFFSET_X, OFFSET_Y))
    pygame.display.flip()
    
    clock.tick(60)

# --- CLEAN SHUTDOWN ---
voice_listener.stop()
tracker.stop()
pygame.quit()
sys.exit()