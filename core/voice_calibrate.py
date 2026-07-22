import sys
import os

# Make the script aware of the main root folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core import config
import cv2
import numpy as np
import pygame
import json
import time
import queue
import threading
import subprocess
import speech_recognition as sr
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- 1. AI BRAIN SETUP (GEMINI) ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

calib_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "action": types.Schema(type=types.Type.STRING, enum=["confirm", "move_edge", "quit", "ignore"]),
        "edge": types.Schema(type=types.Type.STRING, enum=["top", "bottom", "left", "right"]),
        "direction": types.Schema(type=types.Type.STRING, enum=["up", "down", "left", "right"]),
        "amount": types.Schema(type=types.Type.INTEGER)
    },
    required=["action"]
)

sentence_queue = queue.Queue()
action_queue = queue.Queue()
state = "GRID" # Global state for AI context

# --- 2. LOCAL WHISPER LISTENER ---
def listen_loop():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 600 
    while True:
        try:
            with sr.Microphone(sample_rate=16000) as source:
                while True:
                    try:
                        audio = recognizer.listen(source, phrase_time_limit=5)
                        text = recognizer.recognize_whisper(audio, model="base.en").strip()
                        if text:
                            print(f"\n[🎤 Heard]: '{text}'")
                            sentence_queue.put(text)
                    except sr.WaitTimeoutError: continue
                    except Exception: pass
        except Exception:
            time.sleep(2)

# --- 3. AI WORKER LOOP ---
def ai_worker_loop():
    fallback_models = ['gemini-3.1-flash-lite', 'gemini-flash-lite-latest', 'gemini-3.5-flash']
    instructions_path = os.path.join(current_dir, "calib_instructions.txt")
    
    while True:
        text = sentence_queue.get()
        try:
            with open(instructions_path, "r") as f:
                base_prompt = f.read()
        except FileNotFoundError:
            continue 
            
        dynamic_prompt = base_prompt.format(state=state)
        
        success = False
        for model_name in fallback_models:
            if success: break 
            try:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=f"{dynamic_prompt}\nUser: {text}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=calib_schema,
                    ),
                )
                payload = json.loads(response.text)
                print(f"🤖 AI Action: {payload}")
                action_queue.put(payload)
                success = True 
            except Exception: continue 

# --- 4. INITIALIZE CALIBRATION RESOURCES ---
pygame.init()
pygame.key.set_repeat(250, 50) 
pygame.font.init()

# Setup fonts for the error message
font_error_lg = pygame.font.SysFont(None, 120)
font_error_sm = pygame.font.SysFont(None, 80)

OS_W, OS_H = 1920, 1080
try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except Exception:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Generate Smart Grid
GRID_MARKER_SIZE = 80 
GRID_QUIET_ZONE = 10
GRID_INNER = GRID_MARKER_SIZE - (GRID_QUIET_ZONE * 2)

bg_surface = pygame.Surface((OS_W, OS_H))
bg_surface.fill((100, 100, 100))

marker_map = {} 
current_id = 10 

for y in range(0, OS_H, GRID_MARKER_SIZE):
    for x in range(0, OS_W, GRID_MARKER_SIZE):
        if current_id < 1000:
            img = cv2.aruco.generateImageMarker(aruco_dict, current_id, GRID_INNER)
            img[img == 255] = 100
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            bg_surface.blit(surf, (x + GRID_QUIET_ZONE, y + GRID_QUIET_ZONE))
            marker_map[current_id] = {"x": x, "y": y}
            current_id += 1
            
TUNE_QUIET_ZONE = 10
def get_corner_markers(size):
    inner = size - (TUNE_QUIET_ZONE * 2)
    markers = []
    for i in range(4):
        img = cv2.aruco.generateImageMarker(aruco_dict, i, inner)
        img[img == 255] = 100
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        bg = pygame.Surface((size, size))
        bg.fill((100, 100, 100))
        bg.blit(surf, (TUNE_QUIET_ZONE, TUNE_QUIET_ZONE))
        markers.append(bg)
    return markers

cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)

roi_x, roi_y, roi_w, roi_h = 0, 0, 256, 256 
error_display_timer = 0  # This tracks how long to show the camera error on the LED wall

# --- START AI THREADS ---
threading.Thread(target=listen_loop, daemon=True).start()
threading.Thread(target=ai_worker_loop, daemon=True).start()

print("\n--- PHASE 1: AUTO-DETECT ROUGH BOUNDS ---")
print("Say 'Looks Good' or press [SPACE] to lock rough bounds.")

running = True
while running:
    
    # --- PROCESS AI COMMANDS ---
    while not action_queue.empty():
        payload = action_queue.get()
        action = payload.get("action")
        
        if action == "quit":
            running = False
            
        elif action == "confirm":
            if state == "GRID":
                state = "TUNE"
                print("\n\n--- PHASE 2: PIXEL-PERFECT TUNE ---")
                print("Say 'Move top down 5 pixels', or use WASD/Arrows.")
                print("Say 'Looks Good' or press [ENTER] to save and launch system.")
            elif state == "TUNE":
                # Faking the ENTER key press to trigger the save routine
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                
        elif action == "move_edge" and state == "TUNE":
            amt = payload.get("amount", 1)
            edge = payload.get("edge")
            direction = payload.get("direction")
            
            if edge == "top":
                if direction == "up": roi_y -= amt; roi_h += amt
                elif direction == "down": roi_y += amt; roi_h -= amt
            elif edge == "bottom":
                if direction == "up": roi_h -= amt
                elif direction == "down": roi_h += amt
            elif edge == "left":
                if direction == "left": roi_x -= amt; roi_w += amt
                elif direction == "right": roi_x += amt; roi_w -= amt
            elif edge == "right":
                if direction == "left": roi_w -= amt
                elif direction == "right": roi_w += amt

    # --- PROCESS KEYBOARD EVENTS ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False
                
            if state == "GRID" and event.key == pygame.K_SPACE:
                state = "TUNE"
                print("\n\n--- PHASE 2: PIXEL-PERFECT TUNE ---")
                print("Say 'Move top down 5 pixels', or use WASD/Arrows.")
                print("Say 'Looks Good' or press [ENTER] to save and launch system.")

            elif state == "TUNE":
                mods = pygame.key.get_mods()
                step = 10 if mods & pygame.KMOD_SHIFT else 1
                
                if event.key == pygame.K_w: roi_y -= step; roi_h += step
                elif event.key == pygame.K_s: roi_y += step; roi_h -= step
                elif event.key == pygame.K_UP: roi_h -= step
                elif event.key == pygame.K_DOWN: roi_h += step
                elif event.key == pygame.K_a: roi_x -= step; roi_w += step
                elif event.key == pygame.K_d: roi_x += step; roi_w -= step
                elif event.key == pygame.K_LEFT: roi_w -= step
                elif event.key == pygame.K_RIGHT: roi_w += step
                
                elif event.key == pygame.K_RETURN:
                    success, frame = cap.read()
                    if success:
                        blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
                        corners, ids, rejected = detector.detectMarkers(blurred_frame)
                        
                        if ids is not None and all(i in ids for i in [0, 1, 2, 3]):
                            print("\nCalculating Perspective Matrix...")
                            cam_pts = []
                            for i in range(4):
                                idx = np.where(ids == i)[0][0]
                                center_x = np.mean(corners[idx][0][:, 0])
                                center_y = np.mean(corners[idx][0][:, 1])
                                cam_pts.append([center_x, center_y])
                            
                            tune_marker_size = max(50, min(80, roi_w // 3, roi_h // 3))
                            half = tune_marker_size / 2
                            
                            pygame_pts = [
                                [roi_x + half, roi_y + half],
                                [roi_x + roi_w - half, roi_y + half],
                                [roi_x + roi_w - half, roi_y + roi_h - half],
                                [roi_x + half, roi_y + roi_h - half]
                            ]
                            
                            matrix, _ = cv2.findHomography(np.float32(cam_pts), np.float32(pygame_pts))
                            
                            save_data = {
                                "roi_x": roi_x, "roi_y": roi_y, 
                                "roi_w": roi_w, "roi_h": roi_h,
                                "homography_matrix": matrix.tolist()
                            }
                            # Save back to root directory
                            config_path = os.path.join(parent_dir, "led_config.json")
                            with open(config_path, "w") as f:
                                json.dump(save_data, f, indent=4)
                                
                            print(f"\n✅ SUCCESS! Bounds ({roi_w}x{roi_h}) saved.")
                            
                            # CLEANUP BEFORE LAUNCHING MAIN
                            cap.release()
                            cv2.destroyAllWindows()
                            pygame.quit()
                            
                            print("🚀 Auto-launching main.py...")
                            main_path = os.path.join(parent_dir, "main.py")
                            
                            # Use Popen so this script fully detaches and dies
                            subprocess.Popen([sys.executable, main_path], cwd=parent_dir)
                            sys.exit(0)
                        else:
                            print("\nError: Camera cannot see all 4 corner markers clearly. Adjust the camera and say 'looks good' again.")
                            # Set the timer to display the error on the LED wall for 5 seconds
                            error_display_timer = time.time() + 5.0

                roi_w, roi_h = max(100, roi_w), max(100, roi_h)

    # --- CONSTANT WEBCAM FEED ---
    success, frame = cap.read()

    if success:
        if state == "GRID":
            screen.fill((0, 0, 0)) 
            screen.blit(bg_surface, (0, 0))
            
            blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
            corners, ids, rejected = detector.detectMarkers(blurred_frame)
            
            valid_ids = []
            if ids is not None:
                for marker_id in ids.flatten():
                    if marker_id in marker_map:
                        valid_ids.append(marker_id)
            
            if len(valid_ids) > 0:
                min_x = min([marker_map[i]["x"] for i in valid_ids])
                min_y = min([marker_map[i]["y"] for i in valid_ids])
                max_x = max([marker_map[i]["x"] + GRID_MARKER_SIZE for i in valid_ids])
                max_y = max([marker_map[i]["y"] + GRID_MARKER_SIZE for i in valid_ids])
                
                roi_x, roi_y = min_x, min_y
                roi_w, roi_h = max_x - min_x, max_y - min_y
            
            pygame.draw.rect(screen, (0, 255, 0), (roi_x, roi_y, roi_w, roi_h), 4)

        elif state == "TUNE":
            screen.fill((0, 0, 0))
            
            tune_marker_size = max(80, min(80, roi_w // 3, roi_h // 3))
            tune_markers = get_corner_markers(tune_marker_size)
            
            pygame.draw.rect(screen, (0, 255, 0), (roi_x, roi_y, roi_w, roi_h), 2)
            
            screen.blit(tune_markers[0], (roi_x, roi_y))
            screen.blit(tune_markers[1], (roi_x + roi_w - tune_marker_size, roi_y))
            screen.blit(tune_markers[2], (roi_x + roi_w - tune_marker_size, roi_y + roi_h - tune_marker_size))
            screen.blit(tune_markers[3], (roi_x, roi_y + roi_h - tune_marker_size))
            
            # Draw the Camera Error if the timer is active
            if time.time() < error_display_timer:
                # Calculate the dead-center of your physical LED area
                led_center_x = roi_x + (roi_w // 2)
                led_center_y = roi_y + (roi_h // 2)
                
                # Big red warning text
                err_text = font_error_lg.render("ERROR: ADJUST CAMERA", True, (255, 50, 50))
                err_rect = err_text.get_rect(center=(led_center_x, led_center_y - 50))
                
                # Smaller context text
                sub_text = font_error_sm.render("Cannot see all 4 corner markers", True, (255, 100, 200))
                sub_rect = sub_text.get_rect(center=(led_center_x, led_center_y + 50))
                
                # Draw black backgrounds behind the text so it covers up the grid perfectly
                pygame.draw.rect(screen, (0, 0, 0), err_rect.inflate(60, 40))
                pygame.draw.rect(screen, (0, 0, 0), sub_rect.inflate(60, 40))
                
                # Draw the actual text
                screen.blit(err_text, err_rect)
                screen.blit(sub_text, sub_rect)
            
            sys.stdout.write(f"\rFine-Tuning: X:{roi_x} Y:{roi_y} | Resolution: {roi_w}x{roi_h}  ")
            sys.stdout.flush()

        pygame.display.flip()

        display_frame = cv2.flip(frame, 1) 
        if state == "GRID":
            display_corners, display_ids, _ = detector.detectMarkers(cv2.GaussianBlur(display_frame, (7, 7), 0))
            if display_ids is not None:
                cv2.aruco.drawDetectedMarkers(display_frame, display_corners, display_ids)
            
        cv2.imshow("Webcam View", display_frame)
        
        key = cv2.waitKeyEx(1)
        if key != -1:
            if key == 27 or key == ord('q'): running = False
            elif key == 32 and state == "GRID": pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            elif key == 13 and state == "TUNE": pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
            elif state == "TUNE":
                step = 10 if key in [ord('W'), ord('S'), ord('A'), ord('D')] else 1
                if key in [ord('w'), ord('W')]: roi_y -= step; roi_h += step 
                elif key in [ord('s'), ord('S')]: roi_y += step; roi_h -= step 
                elif key in [ord('a'), ord('A')]: roi_x -= step; roi_w += step 
                elif key in [ord('d'), ord('D')]: roi_x += step; roi_w -= step 
                elif key == 63232: roi_h -= step 
                elif key == 63233: roi_h += step 
                elif key == 63234: roi_w -= step 
                elif key == 63235: roi_w += step 

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()