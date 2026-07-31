# --- VERSION 6.11 (Unified Architecture) ---
import sys
import os

# 1. Point Python to the root directory so it can find the 'core' folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 2. Import global paths from the traffic controller
from core.config import LED_CONFIG_PATH, FINGER_CONFIG_PATH, NOSE_CONFIG_PATH, HAND_MODEL_PATH, FACE_MODEL_PATH

# 2. NOW WE CAN SAFELY IMPORT CUSTOM MODULES
from core.vision import HandTracker
from core import config
from physics import spawn_particles, update_physics
from core.voice import VoiceListener

# 3. IMPORT EVERYTHING ELSE
import numpy as np
import pygame
import json
import threading
import math
import queue
import random
import speech_recognition as sr
import time
from PIL import Image  # Required for waterfall_evolve
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- 1. AI BRAIN SETUP (GEMINI) ---

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
physics_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "action": types.Schema(
            type=types.Type.STRING, 
            enum=[
                "undo", "save_state", "load_state", "clear_all", "quit", "ignore",
                "create_obstacle", "modify_obstacle", "remove_obstacle", "rain", "burst", "set_global"
            ]
        ),
        "state_name": types.Schema(type=types.Type.STRING),
        "color_rgb": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
        "color_rgb_2": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)), 
        "gravity": types.Schema(type=types.Type.NUMBER),
        "wind": types.Schema(type=types.Type.NUMBER),
        "wind_swirl": types.Schema(type=types.Type.NUMBER),
        "rain_rate": types.Schema(type=types.Type.INTEGER),
        "max_particles": types.Schema(type=types.Type.INTEGER), 
        "particle_size": types.Schema(type=types.Type.INTEGER),
        "obstacle_size": types.Schema(type=types.Type.INTEGER),
        "emitter_source": types.Schema(type=types.Type.STRING, enum=["sky", "finger"]), 
        "move_x": types.Schema(type=types.Type.INTEGER),
        "move_y": types.Schema(type=types.Type.INTEGER),
        "reply_message": types.Schema(type=types.Type.STRING)
    },
    required=["action"]
)

sentence_queue = queue.Queue()
action_queue = queue.Queue()


# --- 3. AI WORKER LOOP ---
def ai_worker_loop():
    print("[DEBUG] ai_worker_loop thread has started successfully.")
    fallback_models = ['gemini-3.1-flash-lite', 'gemini-flash-lite-latest', 'gemini-3.5-flash', 'gemini-2.5-flash']
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    instructions_path = os.path.join(SCRIPT_DIR, "instructions.txt")
    
    while True:
        text = sentence_queue.get()
        print(f"🧠 AI is thinking about: '{text}'...")
        
        try:
            with open(instructions_path, "r") as f:
                base_prompt = f.read()
        except FileNotFoundError:
            print(f"❌ Error: Missing {instructions_path}")
            continue 
            
        # --- STATE AWARENESS PIPELINE ---
        obs_context = "None"
        if selected_obstacle is not None:
            obs_context = json.dumps(selected_obstacle)
            
        states_list = ", ".join(saved_states.keys()) if saved_states else "None"
            
        try:
            dynamic_prompt = base_prompt.format(
                gravity=global_state['gravity'],
                wind=global_state['wind'],
                swirl=global_state['wind_swirl'],
                selected_obstacle_data=obs_context,
                saved_states_list=states_list
            )
        except KeyError as e:
            print(f"Prompt formatting error: missing {e}")
            continue
        
        success = False
        for model_name in fallback_models:
            if success: break 
            try:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=f"{dynamic_prompt}\nUser: {text}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=physics_schema,
                    ),
                )
                payload = json.loads(response.text)
                print(f"🤖 AI Decision ({model_name}): {payload}")
                action_queue.put(payload)
                success = True 
            except Exception: continue 
        
        if not success: print("❌ All AI models failed. Check Network.")

# --- 4. LOAD CONFIGS & STATES ---
try:
    # 1. Load the screen boundaries
    with open(LED_CONFIG_PATH, "r") as f:
        led_cfg = json.load(f)
    CANVAS_W = int(led_cfg.get("width", 256))
    CANVAS_H = int(led_cfg.get("height", 256)) + 25
    
    # 2. Load the camera homography matrix
    with open(FINGER_CONFIG_PATH, "r") as f:
        finger_cfg = json.load(f)
    homography_matrix = np.array(finger_cfg["homography_matrix"], dtype=np.float32)
    
except Exception as e:
    print(f"❌ Error loading configurations: {e}")
    sys.exit()

saves_path = os.path.join(SCRIPT_DIR, "saves_waterfall.json")
saved_states = {}
if os.path.exists(saves_path):
    try:
        with open(saves_path, "r") as f:
            saved_states = json.load(f)
    except: pass

def waterfall_commands(text_lower): #loading custom waterfall commands
    """Parses voice commands specific to the Waterfall app."""
    if "clear the screen" in text_lower:
        return {"action": "clear_all"}
        
    # Return None if no specific commands matched, so the AI handles it
    return None

# --- 5. INITIALIZE HARDWARE (Pygame & OpenCV) ---
pygame.init()
OS_W, OS_H = 1920, 1080
try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)

# --- 6. APP STATE (Unified Architecture) ---
particles = [] 
obstacles = [] 
selected_obstacle = None
cursor_x, cursor_y = -100, -100 
rain_active = False

global_state = {
    'gravity': 0.04,
    'wind': 0.0,
    'wind_swirl': 0.0,
    'rain_rate': 2,
    'max_particles': 1000,
    'particle_size': 1,
    'emitter_source': 'sky',
    'rain_color': (0, 150, 255),
    'rain_color_2': None
}

ui_message = ""
ui_message_timer = 0

# --- 7. START THREADS ---
voice_listener = VoiceListener(
    sentence_queue, 
    action_queue, 
    custom_parser=waterfall_commands
)
voice_listener.start(prompt_message="[VOICE] Waterfall active! (Say 'Make a rushing waterfall' or 'Make it drizzle')")

threading.Thread(target=ai_worker_loop, daemon=True).start()
tracker = HandTracker(homography_matrix)
tracker.start()

# --- 8. MAIN LOOP ---
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)
running = True

while running:
    current_time = time.time() 
    cursor_x, cursor_y = tracker.get_cursor()

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False
            
    screen.fill((0, 0, 0))

    # Selection Logic
    selected_obstacle = None
    if cursor_x > 0 and cursor_y > 0:
        for obs in obstacles:
            half_s = obs['size'] / 2
            if obs['x'] - half_s <= cursor_x <= obs['x'] + half_s and obs['y'] - half_s <= cursor_y <= obs['y'] + half_s:
                selected_obstacle = obs
                break

    while not action_queue.empty():
        payload = action_queue.get()
        action = payload.get("action")
        
        # Update Globals Safely
        if "gravity" in payload and payload["gravity"] is not None: global_state['gravity'] = float(payload["gravity"])
        if "wind" in payload and payload["wind"] is not None: global_state['wind'] = float(payload["wind"])
        if "wind_swirl" in payload and payload["wind_swirl"] is not None: global_state['wind_swirl'] = float(payload["wind_swirl"])
        if "rain_rate" in payload and payload["rain_rate"] is not None: global_state['rain_rate'] = max(1, int(payload["rain_rate"])) 
        if "max_particles" in payload and payload["max_particles"] is not None: global_state['max_particles'] = max(50, int(payload["max_particles"])) 
        if "particle_size" in payload and payload["particle_size"] is not None: global_state['particle_size'] = max(1, int(payload["particle_size"])) 
        if "emitter_source" in payload and payload["emitter_source"]: global_state['emitter_source'] = payload["emitter_source"]
        
        # Action Handling
        if action in ["rain", "modify", "burst"]:
            if "color_rgb" in payload and payload["color_rgb"]:
                c = payload["color_rgb"]
                if sum(c) < 30: c = [0, 150, 255] 
                global_state['rain_color'] = tuple(c)
                global_state['rain_color_2'] = None 
                for p in particles: p['color'] = global_state['rain_color']
                    
            if "color_rgb_2" in payload and payload["color_rgb_2"]:
                global_state['rain_color_2'] = tuple(payload["color_rgb_2"])
                for p in particles: p['color'] = random.choice([global_state['rain_color'], global_state['rain_color_2']])
                
        if action == "ignore": continue
        elif action == "quit": running = False
            
        elif action == "save_state":
            state_name = payload.get("state_name", "custom").lower()
            saved_states[state_name] = {
                "global_state": global_state.copy(),
                "obstacles": obstacles.copy()
            }
            try:
                with open(saves_path, "w") as f:
                    json.dump(saved_states, f, indent=4)
                ui_message = f"Saved: '{state_name.upper()}'"
                ui_message_timer = time.time()
            except: pass

        elif action == "load_state":
            state_name = payload.get("state_name", "custom").lower()
            if state_name in saved_states:
                st = saved_states[state_name]
                global_state.update(st.get("global_state", {}))
                obstacles = st.get("obstacles", []).copy()
                rain_active = True
                ui_message = f"Loaded: '{state_name.upper()}'"
                ui_message_timer = time.time()
                for p in particles:
                    if global_state['rain_color_2']:
                        p['color'] = random.choice([global_state['rain_color'], global_state['rain_color_2']])
                    else:
                        p['color'] = global_state['rain_color']
            
        elif action == "create_obstacle":
            obs_color = tuple(payload["color_rgb"]) if "color_rgb" in payload else (255, 255, 255)
            obs_size = payload.get("obstacle_size", 3) 
            obstacles.append({
                "x": cursor_x if cursor_x > 0 else CANVAS_W // 2,
                "y": cursor_y if cursor_y > 0 else CANVAS_H // 2,
                "size": obs_size,
                "color": obs_color
            })
            
        elif action == "modify_obstacle" and selected_obstacle is not None:
            if "color_rgb" in payload: selected_obstacle['color'] = tuple(payload["color_rgb"])
            if "obstacle_size" in payload: selected_obstacle['size'] = payload["obstacle_size"]
            if "move_x" in payload: selected_obstacle['x'] += payload["move_x"]
            if "move_y" in payload: selected_obstacle['y'] += payload["move_y"]
            
        elif action == "clear_all":
            obstacles.clear()
            particles.clear()
            rain_active = False

        elif action == "remove_obstacle" and selected_obstacle is not None:
            if selected_obstacle in obstacles:
                obstacles.remove(selected_obstacle)
            selected_obstacle = None
            
        elif action == "burst":
            for _ in range(30):
                particles.append({
                    "x": cursor_x if cursor_x > 0 else CANVAS_W // 2, 
                    "y": cursor_y if cursor_y > 0 else CANVAS_H // 2,
                    "vx": random.uniform(-1, 1), 
                    "vy": random.uniform(-1, 1), 
                    "color": global_state['rain_color']
                })
            
        elif action == "rain":
            rain_active = True

    # Run Physics using the unified global_state dict
    if rain_active:
        spawn_particles(particles, cursor_x, cursor_y, CANVAS_W, CANVAS_H, global_state)
        
    update_physics(particles, obstacles, current_time, cursor_x, cursor_y, CANVAS_W, CANVAS_H, global_state)
    
    # Render
    for obs in obstacles:
        half_s = obs['size'] / 2
        rect = (int(obs['x'] - half_s), int(obs['y'] - half_s), obs['size'], obs['size'])
        pygame.draw.rect(screen, obs['color'], rect)
        if obs == selected_obstacle:
            pygame.draw.rect(screen, (255, 255, 255), rect, 1)

    for p in particles:
        pygame.draw.rect(screen, p['color'], (int(p['x']), int(p['y']), global_state['particle_size'], global_state['particle_size']))
        
    pygame.draw.circle(screen, (255, 255, 255), (cursor_x, cursor_y), 6, 1)

    if time.time() - ui_message_timer < 4.0: 
        msg_text = font.render(ui_message, True, (0, 255, 255)) 
        screen.blit(msg_text, (10, 40))

    pygame.display.flip()
    clock.tick(60)

tracker.stop()
pygame.quit()
os._exit(0) # Force OS-level termination to kill hanging audio/camera threads