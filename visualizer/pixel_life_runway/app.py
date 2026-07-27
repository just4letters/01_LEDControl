import os
import sys
import json
import time
import queue
import random
import threading
import numpy as np
import pygame
import speech_recognition as sr
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- DIRECTORY SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels (pixel_life_runway -> visualizer -> 01_LEDControl)
root_dir = os.path.dirname(os.path.dirname(current_dir)) 
sys.path.append(root_dir)

# Load .env file explicitly from two directories up (01_LEDControl/.env)
dotenv_path = os.path.join(root_dir, ".env")
load_dotenv(dotenv_path)

# --- CUSTOM ENGINE IMPORTS ---
from core.vision import HandTracker
from core import config
from physics import get_selected_node, spawn_particles, update_physics, draw_scene
from runway_worker import runway_background_worker, runway_task_queue
from runwayml import RunwayML

# Background Video State variables
bg_frames = []
bg_frame_idx = 0

# --- 1. AI BRAIN SETUP (GEMINI) ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

physics_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "action": types.Schema(
            type=types.Type.STRING, 
            enum=[
                # Universal System Commands
                "undo", "save_state", "load_state", "list_states", "close_menu", "clear_all", "quit", "ignore",
                # Pixel Life Specific
                "create_node", "modify_node", "remove_node", "set_finger", "set_global"
            ]
        ),
        "state_name": types.Schema(type=types.Type.STRING),
        "shape": types.Schema(type=types.Type.STRING, enum=["square", "circle", "rectangle", "line", "vertical_line", "triangle"]),
        "size": types.Schema(type=types.Type.INTEGER),
        "color": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
        "animation": types.Schema(type=types.Type.STRING, enum=["static", "pulse", "blink", "glisten"]),
        "physics_effect": types.Schema(type=types.Type.STRING, enum=["none", "attract", "bounce"]),
        "physics_strength": types.Schema(type=types.Type.NUMBER),
        "emit_speed": types.Schema(type=types.Type.NUMBER),
        "emit_rate": types.Schema(type=types.Type.INTEGER),
        "emit_size": types.Schema(type=types.Type.INTEGER),
        "emit_colors": types.Schema(
            type=types.Type.ARRAY, 
            items=types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER))
        ),
        "finger_mode": types.Schema(type=types.Type.STRING, enum=["none", "emit", "attract", "repel"]),
        "move_x": types.Schema(type=types.Type.INTEGER),
        "move_y": types.Schema(type=types.Type.INTEGER),
        "gravity_x": types.Schema(type=types.Type.NUMBER),
        "gravity_y": types.Schema(type=types.Type.NUMBER)
    },
    required=["action"]
)

sentence_queue = queue.Queue()
action_queue = queue.Queue()

# --- 2. LOCAL WHISPER LISTENER ---
def listen_loop():
    print("[DEBUG] listen_loop thread started.")
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 600 
    
    while True:
        try:
            with sr.Microphone(sample_rate=16000) as source:
                print("\n[VOICE] Local Whisper active!")
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
                            if "background" in text_lower or "backdrop" in text_lower:
                                clean_prompt = text_lower.replace("create a background of", "").replace("background", "").replace("backdrop", "").strip()
                                runway_task_queue.put(clean_prompt)
                                action_queue.put({"action": "ui_notify", "message": f"Generating Runway Backdrop: '{clean_prompt}'..."})
                                continue
                            
                            while not sentence_queue.empty():
                                try: sentence_queue.get_nowait()
                                except queue.Empty: break
                                    
                            sentence_queue.put(text)
                    except sr.WaitTimeoutError: continue
                    except sr.UnknownValueError: pass 
                    except Exception as inner_e: print(f"[Whisper Loop Error]: {inner_e}")
        except Exception as outer_e:
            print(f"⚠️ Microphone Error: {outer_e}. Rebooting in 2s...")
            time.sleep(2)

# --- 3. AI WORKER LOOP ---
def ai_worker_loop():
    print("[DEBUG] ai_worker_loop thread started.")
    fallback_models = ['gemini-3.1-flash-lite', 'gemini-flash-lite-latest', 'gemini-3.5-flash', 'gemini-2.5-flash']
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
        node_context = "None"
        if selected_node is not None:
            clean_node = {k: v for k, v in selected_node.items() if k not in ['last_emit_time', 'animation_timer']}
            node_context = json.dumps(clean_node)
            
        states_list = ", ".join(saved_states.keys()) if saved_states else "None"
            
        try:
            dynamic_prompt = base_prompt.format(
                gravity_x=global_state['gravity_x'],
                gravity_y=global_state['gravity_y'],
                node_count=len(nodes),
                selected_node_data=node_context,
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

# --- 4. LOAD CONFIG & SAVED STATES ---
config_path = os.path.join(root_dir, "led_config.json")
try:
    with open(config_path, "r") as f:
        led_cfg = json.load(f)
    homography_matrix = np.array(led_cfg["homography_matrix"], dtype=np.float32)
    CANVAS_W = int(led_cfg.get("width", 256))
    CANVAS_H = int(led_cfg.get("height", 256)) + 25
except Exception as e:
    print(f"Error loading led_config.json: {e}")
    sys.exit()

saves_path = os.path.join(current_dir, "saves_pixel_life.json")
saved_states = {}
if os.path.exists(saves_path):
    try:
        with open(saves_path, "r") as f:
            saved_states = json.load(f)
        print(f"[DEBUG] Loaded {len(saved_states)} saved configurations.")
    except Exception: pass

pygame.init()
OS_W, OS_H = 1920, 1080
try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except Exception:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)

# --- 5. APP STATE ---
nodes = []
particles = []
selected_node = None

finger_state = {
    'x': -100, 'y': -100, 
    'mode': 'none', 
    'physics_strength': 2.0,
    'emit_rate': 10,
    'emit_speed': 3.0,
    'emit_colors': [(255, 255, 255)],
    'last_emit_time': 0
}

global_state = {
    'gravity_x': 0.0,
    'gravity_y': 0.0
}

ui_message = ""
ui_message_timer = 0
show_saved_states = False  # Flag for saved worlds UI overlay

# --- 6. START THREADS ---
threading.Thread(target=listen_loop, daemon=True).start()
threading.Thread(target=ai_worker_loop, daemon=True).start()
tracker = HandTracker(homography_matrix)
tracker.start()

# Start Runway Thread
threading.Thread(
    target=runway_background_worker, 
    args=(action_queue, CANVAS_W, CANVAS_H), 
    daemon=True
).start()

# --- 7. MAIN GAME LOOP ---
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)
font_title = pygame.font.SysFont(None, 36)
running = True

while running:
    current_time = time.time()
    cursor_x, cursor_y = tracker.get_cursor()
    finger_state['x'] = cursor_x
    finger_state['y'] = cursor_y

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False
            
    selected_node = get_selected_node(cursor_x, cursor_y, nodes, selected_node)

    while not action_queue.empty():
        payload = action_queue.get()
        action = payload.get("action")
        
        if action == "set_background_frames":
            bg_frames = payload.get("frames", [])
            bg_frame_idx = 0
            ui_message = "New Runway Backdrop Loaded!"
            ui_message_timer = time.time()

        elif action == "ui_notify":
            ui_message = payload.get("message", "")
            ui_message_timer = time.time()

        elif action == "quit": running = False
        elif action == "ignore": continue
        elif action == "clear_all":
            nodes.clear()
            particles.clear()

        elif action == "list_states":
            show_saved_states = True

        elif action == "close_menu":
            show_saved_states = False

        elif action == "save_state":
            state_name = payload.get("state_name", "custom").lower()
            clean_finger = {k: v for k, v in finger_state.items() if k not in ['x', 'y', 'last_emit_time']}
            saved_states[state_name] = {
                "global_state": global_state.copy(),
                "finger_state": clean_finger,
                "nodes": json.loads(json.dumps(nodes))
            }
            try:
                with open(saves_path, "w") as f:
                    json.dump(saved_states, f, indent=4)
                ui_message = f"Saved: '{state_name.upper()}'"
                ui_message_timer = time.time()
            except Exception as e:
                print(f"Error saving state: {e}")

        elif action == "load_state":
            state_name = payload.get("state_name", "custom").lower()
            if state_name in saved_states:
                st = saved_states[state_name]
                global_state.update(st.get("global_state", {}))
                if "finger_state" in st:
                    finger_state.update(st["finger_state"])
                nodes = json.loads(json.dumps(st.get("nodes", [])))
                particles.clear()
                ui_message = f"Loaded: '{state_name.upper()}'"
                ui_message_timer = time.time()
                show_saved_states = False  # Auto-close menu when state loads
            else:
                ui_message = f"Not Found: '{state_name.upper()}'"
                ui_message_timer = time.time()
            
        elif action == "remove_node" and selected_node is not None:
            if selected_node in nodes:
                nodes.remove(selected_node)
            selected_node = None
            
        elif action == "set_global":
            if "gravity_x" in payload: global_state['gravity_x'] = payload['gravity_x']
            if "gravity_y" in payload: global_state['gravity_y'] = payload['gravity_y']
            
        elif action == "set_finger":
            if "finger_mode" in payload: finger_state['mode'] = payload['finger_mode']
            if "physics_strength" in payload: finger_state['physics_strength'] = payload['physics_strength']
            if "emit_rate" in payload: finger_state['emit_rate'] = payload['emit_rate']
            if "emit_colors" in payload: finger_state['emit_colors'] = [tuple(c) for c in payload['emit_colors']]

        elif action == "create_node":
            targ_x = cursor_x if cursor_x > 0 else CANVAS_W // 2
            targ_y = cursor_y if cursor_y > 0 else CANVAS_H // 2
            col = tuple(payload.get("color", [255, 255, 255]))
            emit_cols = [tuple(c) for c in payload.get("emit_colors", [])]
            emit_rate = payload.get("emit_rate", 0)
            if len(emit_cols) > 0 and emit_rate == 0: emit_rate = 15
                
            nodes.append({
                "x": targ_x, "y": targ_y,
                "size": payload.get("size", 1), 
                "shape": payload.get("shape", "square"), 
                "color": col,
                "animation": payload.get("animation", "static"),
                "physics_effect": payload.get("physics_effect", "none"),
                "physics_strength": payload.get("physics_strength", 1.0),
                "emit_rate": emit_rate,
                "emit_speed": payload.get("emit_speed", 2.0),
                "emit_size": payload.get("emit_size", 1),
                "emit_colors": emit_cols,
                "last_emit_time": 0, "animation_timer": 0
            })
            
        elif action == "modify_node" and selected_node is not None:
            if "size" in payload: selected_node['size'] = payload['size']
            if "shape" in payload: selected_node['shape'] = payload['shape']
            if "color" in payload: selected_node['color'] = tuple(payload['color'])
            if "animation" in payload: selected_node['animation'] = payload['animation']
            if "physics_effect" in payload: selected_node['physics_effect'] = payload['physics_effect']
            if "physics_strength" in payload: selected_node['physics_strength'] = payload['physics_strength']
            if "emit_rate" in payload: selected_node['emit_rate'] = payload['emit_rate']
            if "emit_speed" in payload: selected_node['emit_speed'] = payload['emit_speed']
            if "emit_size" in payload: selected_node['emit_size'] = payload['emit_size']
            if "move_x" in payload: selected_node['x'] += payload["move_x"]
            if "move_y" in payload: selected_node['y'] += payload["move_y"]
            
            if "emit_colors" in payload: 
                selected_node['emit_colors'] = [tuple(c) for c in payload['emit_colors']]
                if payload.get("emit_rate") is None and selected_node.get("emit_rate", 0) == 0:
                    selected_node['emit_rate'] = 15

    # Run Physics Calculations
    spawn_particles(particles, nodes, finger_state, current_time)
    update_physics(particles, nodes, finger_state, global_state, CANVAS_W, CANVAS_H)

    # --- 8. RENDERING ENGINE ---
    screen.fill((0, 0, 0)) 

    # Render active Runway video frame if available (as backdrop)
    if bg_frames and len(bg_frames) > 0:
        frame_rgb = bg_frames[bg_frame_idx]
        bg_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(bg_surface, (0, 0))
        bg_frame_idx = (bg_frame_idx + 1) % len(bg_frames)

    # Render Nodes, Physics Particles, and Cursor on top of backdrop
    draw_scene(screen, particles, nodes, selected_node, current_time)
    pygame.draw.circle(screen, (100, 100, 100), (cursor_x, cursor_y), 4, 1)

    # Render UI Notification Banner
    if time.time() - ui_message_timer < 4.0: 
        msg_text = font.render(ui_message, True, (0, 255, 255)) 
        screen.blit(msg_text, (10, 40))

    # --- RENDER SAVED WORLDS MENU OVERLAY ---
    if show_saved_states:
        overlay = pygame.Surface((400, 320), pygame.SRCALPHA)
        overlay.fill((15, 15, 25, 220)) # Dark transparent box
        pygame.draw.rect(overlay, (0, 200, 255), (0, 0, 400, 320), 2, border_radius=8)

        # Title Header
        title_surf = font_title.render("SAVED WORLDS", True, (0, 255, 255))
        overlay.blit(title_surf, (20, 15))

        # List saved names
        if saved_states:
            y_offset = 60
            for name in list(saved_states.keys())[:7]: # Limit to 7 items for clean fit
                item_surf = font.render(f"• {name.upper()}", True, (255, 255, 255))
                overlay.blit(item_surf, (30, y_offset))
                y_offset += 30
        else:
            empty_surf = font.render("(No saved states found)", True, (150, 150, 150))
            overlay.blit(empty_surf, (30, 60))

        # Footer Prompt
        footer_surf = font.render("Say 'Load [Name]' or 'Close'", True, (180, 180, 180))
        overlay.blit(footer_surf, (20, 280))

        # Place menu at top-left to ensure it hits the physical LED wall
        screen.blit(overlay, (20, 80))

    pygame.display.flip()
    clock.tick(60)

tracker.stop()
pygame.quit()
os._exit(0) # Force OS-level termination to kill hanging audio/camera threads