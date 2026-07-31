"""
nose_calibration.py

Description:
    Generates a homography matrix for 3D head-pose tracking.
    Uses FaceTracker from core.vision to calculate head vectors, mapping 
    them to the 4 corners of the LED canvas.
    Outputs the matrix to 'nose_config.json' in the root directory.

Usage:
    Run standalone or via main.py. Point nose at red targets and say 'looks good'
    or press SPACE to lock.
"""

import os
import sys
import json
import time
import queue
import numpy as np
import pygame
import cv2

# --- BULLETPROOF PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import central modules
from core import config
from core.vision import FaceTracker
from core.voice import VoiceListener


def create_calib_parser():
    """Parser function to allow voice-driven targets and calibration actions."""
    def nose_calib_commands(text_lower):
        if "looks good" in text_lower or "drop it" in text_lower or "place it" in text_lower or "lock" in text_lower:
            return {"action": "confirm"}
        elif "reset" in text_lower or "start over" in text_lower or "try again" in text_lower:
            return {"action": "reset"}
        return None
    return nose_calib_commands


def run_calibration():
    print("🎬 Initializing 3D Nose-Pointing Calibrator...")

    # Load configurations via central paths
    try:
        with open(config.LED_CONFIG_PATH, "r") as f:
            led_cfg = json.load(f)
        CANVAS_W = int(led_cfg.get("width", 256))
        CANVAS_H = int(led_cfg.get("height", 256))
        OFFSET_X = int(led_cfg.get("offset_x", 0))
        OFFSET_Y = int(led_cfg.get("offset_y", 0))
    except Exception:
        print(f"⚠️ Could not find {config.LED_CONFIG_PATH}, using defaults.")
        CANVAS_W, CANVAS_H, OFFSET_X, OFFSET_Y = 256, 256, 0, 0

    # Dummy identity matrix passed initially to initialize raw head angle readings
    identity_matrix = np.eye(3, dtype=np.float32)
    tracker = FaceTracker(identity_matrix)
    tracker.start()

    # Voice & Action setup
    sentence_queue = queue.Queue()
    action_queue = queue.Queue()
    voice_listener = VoiceListener(sentence_queue, action_queue, custom_parser=create_calib_parser())
    voice_listener.start(prompt_message="🎤 Voice active! Point nose at target and say 'Looks good' or press SPACE.")

    pygame.init()
    OS_W, OS_H = 1920, 1080
    try:
        screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
    except Exception:
        screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

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
    
    success_flag = False

    print("\n🎯 3D NOSE CALIBRATION STARTED")

    while calibrating or testing:
        # Get head pose vector from centralized FaceTracker
        nose_vector_x, nose_vector_y, _, _ = tracker.get_state()
        face_detected = (nose_vector_x != -100 and nose_vector_y != -100)

        # Voice action handling
        while not action_queue.empty():
            payload = action_queue.get()
            act = payload.get("action")
            if act == "quit":
                calibrating, testing = False, False
            elif act == "confirm":
                if calibrating and face_detected:
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
            elif act == "reset":
                nose_points = []
                current_target = 0
                calibrating, testing = True, False
                matrix = None

        # Keyboard Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): 
                calibrating, testing = False, False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if calibrating and face_detected:
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
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_r, pygame.K_RESET):
                nose_points = []
                current_target = 0
                calibrating, testing = True, False
                matrix = None

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
        screen.fill((0, 0, 0))
        screen.blit(canvas, (OFFSET_X, OFFSET_Y))
        pygame.display.flip()
        time.sleep(0.016) # ~60 FPS update rate

    voice_listener.stop()
    tracker.stop()
    cv2.destroyAllWindows()
    pygame.quit()

    if success_flag and matrix is not None:
        with open(config.NOSE_CONFIG_PATH, "w") as f:
            json.dump({"homography_matrix": matrix.tolist(), "canvas_width": CANVAS_W, "canvas_height": CANVAS_H}, f, indent=4)
        print(f"\n🎉 CALIBRATION SAVED to {config.NOSE_CONFIG_PATH}!")


if __name__ == "__main__":
    run_calibration()