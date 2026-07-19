# --- VERSION 6.11 (Unified Architecture) ---
from core.vision import HandTracker
from core import config
from physics import spawn_particles, update_physics
import numpy as np
import pygame
import sys
import os
import json
import threading
import math
import queue
import random
import speech_recognition as sr
import time
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

# --- 2. LOCAL WHISPER LISTENER ---
def listen_loop():
    print("[DEBUG] listen_loop thread has started successfully.")
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 600 
    recognizer.dynamic_energy_threshold = False 
    
    while True:
        try:
            with sr.Microphone(sample_rate=16000) as source:
                print("\n[VOICE] Local Whisper active! (Say 'Make a rushing waterfall' or 'Make it drizzle')")
                
                while True:
                    try:
                        audio = recognizer.listen(source, phrase_time_limit=5)
                        text = recognizer.recognize_whisper(audio, model="base.en").strip()
                        
                        if text:
                            text_lower = text.lower()
                            print(f"\n> Heard: '{text}'")
                            
                            if "shut down" in text_lower or "quit" in text_lower:
                                action_queue.put({"action": "quit"})
                                return 
                            if "clear the screen" in text_lower:
                                action_queue.put({"action": "clear_all"})
                                continue
                            
                            while not sentence_queue.empty():
                                try: sentence_queue.get_nowait()
                                except queue.Empty: break
                                    
                            sentence_queue.put(text)
                            
                    except sr.WaitTimeoutError: continue
                    except sr.UnknownValueError: pass 
                    except Exception as inner_e: print(f"[Whisper Loop Error]: {inner_e}")
                        
        except Exception as outer_e:
            print(f"⚠️ Hardware Microphone Error: {outer_e}. Rebooting in 2s...")
            time.sleep(2)

# --- 3. AI WORKER LOOP ---
def ai_worker_loop():
    print("[DEBUG] ai_worker_loop thread has started successfully.")
    fallback_models = ['gemini-3.1-flash-lite', 'gemini-flash-lite-latest', 'gemini-3.5-flash', 'gemini-2.5-flash']
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    instructions_path = os.path.join(current_dir, "instructions.txt")
    
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
            
        try:
            dynamic_prompt = base_prompt.format(
                gravity=global_state['gravity'],
                wind=global_state['wind'],
                swirl=global_state['wind_swirl'],
                selected_obstacle_data=obs_context
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
    with open("led_config.json", "r") as f:
        config_data = json.load(f)
    homography_matrix = np.array(config_data["homography_matrix"], dtype=np.float32)
    CANVAS_W = int(config_data.get("width", 256))
    CANVAS_H = int(config_data.get("height", 256)) + 25
except Exception as e:
    print("Error loading led_config.json.")
    sys.exit()

saved_states = {}
if os.path.exists("saved_states.json"):
    try:
        with open("saved_states.json", "r") as f:
            saved_states = json.load(f)
    except: pass

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
threading.Thread(target=listen_loop, daemon=True).start()
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
                with open("saved_states.json", "w") as f:
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
sys.exit()