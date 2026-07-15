# --- VERSION 6.10 ---
from vision import HandTracker
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

# --- 1. AI BRAIN SETUP (GEMINI) ---
client = genai.Client(api_key="YOUR_API_KEY_HERE")

physics_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "action": types.Schema(
            type=types.Type.STRING, 
            enum=["burst", "rain", "clear", "modify", "quit", "ignore", "add_obstacle", "modify_obstacle", "clear_obstacles", "save_state", "load_state"]
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
                                print("⚡ Local Override: Shutting down instantly!")
                                action_queue.put({"action": "quit"})
                                return 
                                
                            if "clear the screen" in text_lower:
                                print("⚡ Local Override: Clearing canvas instantly!")
                                action_queue.put({"action": "clear"})
                                continue
                            
                            while not sentence_queue.empty():
                                try:
                                    sentence_queue.get_nowait()
                                except queue.Empty:
                                    break
                                    
                            sentence_queue.put(text)
                            
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        pass 
                    except Exception as inner_e:
                        print(f"[Whisper Loop Error]: {inner_e}")
                        
        except Exception as outer_e:
            print(f"⚠️ Hardware Microphone Error: {outer_e}")
            print("🔄 Rebooting the microphone in 2 seconds...")
            time.sleep(2)

# --- 3. AI WORKER LOOP ---
def ai_worker_loop():
    print("[DEBUG] ai_worker_loop thread has started successfully.")
    fallback_models = [
        'gemini-3.1-flash-lite',
        'gemini-flash-lite-latest',
        'gemini-3.5-flash',
        'gemini-2.5-flash'
    ]
    
    while True:
        text = sentence_queue.get()
        print(f"🧠 AI is thinking about: '{text}'...")
        
        # Read the prompt from the text file safely
        try:
            with open("ai_instructions.txt", "r") as f:
                base_prompt = f.read()
        except FileNotFoundError:
            print("❌ Error: ai_instructions.txt is missing!")
            time.sleep(2)
            continue 
            
        # Inject the live physics variables into the placeholders
        dynamic_prompt = base_prompt.format(
            gravity=global_gravity,
            wind=global_wind,
            swirl=global_wind_swirl
        )
        
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
                
            except Exception:
                continue 
        
        if not success:
            print("❌ All AI models failed (503/404). Check Network.")



# --- 5. LOAD CONFIGS & STATES ---
try:
    with open("led_config.json", "r") as f:
        config = json.load(f)
    homography_matrix = np.array(config["homography_matrix"], dtype=np.float32)
    
    if "width" in config and "height" in config:
        CANVAS_W = int(config["width"])
        CANVAS_H = int(config["height"])
    elif "canvas_w" in config and "canvas_h" in config:
        CANVAS_W = int(config["canvas_w"])
        CANVAS_H = int(config["canvas_h"])
    elif "matrix_size" in config:
        CANVAS_W, CANVAS_H = int(config["matrix_size"][0]), int(config["matrix_size"][1])
    elif "dst_points" in config:
        CANVAS_W = int(max(pt[0] for pt in config["dst_points"]))
        CANVAS_H = int(max(pt[1] for pt in config["dst_points"]))
    else:
        CANVAS_W, CANVAS_H = 256, 256
        
    CANVAS_H += 25 
    print(f"[DEBUG] Physical Canvas Mapped To: {CANVAS_W}x{CANVAS_H}")
        
except Exception as e:
    print("Error loading led_config.json. Run calibrate.py first.")
    sys.exit()

saved_states = {}
if os.path.exists("saved_states.json"):
    try:
        with open("saved_states.json", "r") as f:
            saved_states = json.load(f)
        print(f"[DEBUG] Loaded {len(saved_states)} saved configurations.")
    except Exception as e:
        pass

# --- 6. INITIALIZE HARDWARE (Pygame & OpenCV) ---
pygame.init()
OS_W, OS_H = 1920, 1080

try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)

# --- 7. START BACKGROUND THREADS ---
threading.Thread(target=listen_loop, daemon=True).start()
threading.Thread(target=ai_worker_loop, daemon=True).start()
tracker = HandTracker(homography_matrix)
tracker.start()

# --- 8. APP STATE (PARTICLES & PHYSICS) ---
particles = [] 
obstacles = [] 
cursor_x, cursor_y = -100, -100 

rain_active = False
global_rain_color = (0, 150, 255) 
global_rain_color_2 = None 
global_gravity = 0.04  
global_wind = 0.0
global_wind_swirl = 0.0 
global_rain_rate = 2
global_max_particles = 1000 # Increased default capacity
global_particle_size = 1
global_emitter_source = "sky" 

ui_message = ""
ui_message_timer = 0

# --- 9. MAIN LOOP ---
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)
running = True

while running:
    current_time = time.time() 
    
    cursor_x, cursor_y = tracker.get_cursor()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
            
    screen.fill((0, 0, 0))

    hovered_obstacle = None
    if cursor_x > 0 and cursor_y > 0:
        for obs in obstacles:
            half_s = obs['size'] / 2
            if obs['x'] - half_s <= cursor_x <= obs['x'] + half_s and obs['y'] - half_s <= cursor_y <= obs['y'] + half_s:
                hovered_obstacle = obs
                break

    while not action_queue.empty():
        payload = action_queue.get()
        action = payload.get("action")
        
        try:
            if "gravity" in payload and payload["gravity"] is not None: 
                global_gravity = float(payload["gravity"])
            if "wind" in payload and payload["wind"] is not None: 
                global_wind = float(payload["wind"])
            if "wind_swirl" in payload and payload["wind_swirl"] is not None: 
                global_wind_swirl = float(payload["wind_swirl"])
            if "rain_rate" in payload and payload["rain_rate"] is not None: 
                global_rain_rate = max(1, int(payload["rain_rate"])) 
            if "max_particles" in payload and payload["max_particles"] is not None: 
                global_max_particles = max(50, int(payload["max_particles"])) 
            if "particle_size" in payload and payload["particle_size"] is not None: 
                global_particle_size = max(1, int(payload["particle_size"])) 
            if "emitter_source" in payload and payload["emitter_source"]: 
                global_emitter_source = payload["emitter_source"]
        except Exception as e:
            print(f"⚠️ Ignored bad AI math: {e}")
        
        if action in ["rain", "modify", "burst"]:
            if "color_rgb" in payload and payload["color_rgb"]:
                c = payload["color_rgb"]
                if sum(c) < 30: c = [0, 150, 255] 
                global_rain_color = tuple(c)
                global_rain_color_2 = None 
                for p in particles:
                    p['color'] = global_rain_color
                    
            if "color_rgb_2" in payload and payload["color_rgb_2"]:
                global_rain_color_2 = tuple(payload["color_rgb_2"])
                for p in particles:
                    p['color'] = random.choice([global_rain_color, global_rain_color_2])
                
        if action == "ignore":
            continue
            
        elif action == "save_state":
            state_name = payload.get("state_name", "custom").lower()
            saved_states[state_name] = {
                "rain_color": global_rain_color,
                "rain_color_2": global_rain_color_2,
                "gravity": global_gravity,
                "wind": global_wind,
                "wind_swirl": global_wind_swirl,
                "rain_rate": global_rain_rate,
                "max_particles": global_max_particles,
                "particle_size": global_particle_size,
                "emitter_source": global_emitter_source,
                "obstacles": obstacles.copy()
            }
            try:
                with open("saved_states.json", "w") as f:
                    json.dump(saved_states, f, indent=4)
                ui_message = f"Saved: '{state_name.upper()}'"
                ui_message_timer = time.time()
            except:
                pass

        elif action == "load_state":
            state_name = payload.get("state_name", "custom").lower()
            if state_name in saved_states:
                st = saved_states[state_name]
                global_rain_color = tuple(st.get("rain_color", (255, 255, 255)))
                global_rain_color_2 = tuple(st["rain_color_2"]) if st.get("rain_color_2") else None
                global_gravity = st.get("gravity", 0.04)
                global_wind = st.get("wind", 0.0)
                global_wind_swirl = st.get("wind_swirl", 0.0)
                global_rain_rate = st.get("rain_rate", 2)
                global_max_particles = st.get("max_particles", 1000)
                global_particle_size = st.get("particle_size", 1)
                global_emitter_source = st.get("emitter_source", "sky")
                obstacles = st.get("obstacles", []).copy()
                
                rain_active = True
                ui_message = f"Loaded: '{state_name.upper()}'"
                ui_message_timer = time.time()
                
                for p in particles:
                    if global_rain_color_2:
                        p['color'] = random.choice([global_rain_color, global_rain_color_2])
                    else:
                        p['color'] = global_rain_color
            
        elif action == "add_obstacle":
            obs_color = tuple(payload["color_rgb"]) if "color_rgb" in payload else (255, 255, 255)
            obs_size = payload.get("obstacle_size", 3) 
            obstacles.append({
                "x": cursor_x if cursor_x > 0 else CANVAS_W // 2,
                "y": cursor_y if cursor_y > 0 else CANVAS_H // 2,
                "size": obs_size,
                "color": obs_color
            })
            
        elif action == "modify_obstacle":
            if hovered_obstacle is not None:
                if "color_rgb" in payload:
                    hovered_obstacle['color'] = tuple(payload["color_rgb"])
                if "obstacle_size" in payload:
                    hovered_obstacle['size'] = payload["obstacle_size"]
            
        elif action == "clear_obstacles":
            obstacles.clear()
            
        elif action == "burst":
            for _ in range(30):
                particles.append({
                    "x": cursor_x if cursor_x > 0 else CANVAS_W // 2, 
                    "y": cursor_y if cursor_y > 0 else CANVAS_H // 2,
                    "vx": random.uniform(-1, 1), 
                    "vy": random.uniform(-1, 1), 
                    "color": global_rain_color
                })
                
        elif action == "clear":
            rain_active = False
            particles.clear()
            
        elif action == "rain":
            rain_active = True
            
        elif action == "quit":
            running = False
# Bundle the global variables so the physics engine can read them
    physics_state = {
        'gravity': global_gravity,
        'wind': global_wind,
        'wind_swirl': global_wind_swirl,
        'rain_rate': global_rain_rate,
        'max_particles': global_max_particles,
        'particle_size': global_particle_size,
        'emitter_source': global_emitter_source,
        'rain_color': global_rain_color,
        'rain_color_2': global_rain_color_2
    }

    # 1. Spawn new particles (if raining)
    if rain_active:
        spawn_particles(particles, cursor_x, cursor_y, CANVAS_W, CANVAS_H, physics_state)
        
    # 2. Update velocities, physics, and collisions
    update_physics(particles, obstacles, current_time, cursor_x, cursor_y, CANVAS_W, CANVAS_H, physics_state)
    for obs in obstacles:
        half_s = obs['size'] / 2
        rect = (int(obs['x'] - half_s), int(obs['y'] - half_s), obs['size'], obs['size'])
        pygame.draw.rect(screen, obs['color'], rect)
        
        if obs == hovered_obstacle:
            pygame.draw.rect(screen, (255, 255, 255), rect, 1)

    for p in particles:
        pygame.draw.rect(screen, p['color'], (int(p['x']), int(p['y']), global_particle_size, global_particle_size))
        
    pygame.draw.circle(screen, (255, 255, 255), (cursor_x, cursor_y), 6, 1)

    if time.time() - ui_message_timer < 4.0: 
        msg_text = font.render(ui_message, True, (0, 255, 255)) 
        screen.blit(msg_text, (10, 40))

    pygame.display.flip()
    clock.tick(60)

tracker.stop()
pygame.quit()
sys.exit()