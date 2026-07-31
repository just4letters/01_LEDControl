# --- AMBIENT VISUALIZER (Unified Architecture) ---
# A slow, flowing light field (not particles) driven by weather, time of
# day, and room motion - plus voice commands to nudge palette/energy/
# brightness or switch the camera to finger-tracking instead of room-watch.
#
# Voice commands (local Whisper -> Gemini, see instructions.txt):
#   "sync weather to <city>"     "make it calmer" / "more energetic"
#   "dim it" / "brighten it up"  "go moody blue" / "sunset vibes"
#   "track my finger" / "watch the room"   "reset" / "back to normal"
#   "quit" / "shut down"

import os
import sys

# 1. FIX THE SYSTEM PATH FIRST SO PYTHON CAN FIND 'core'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

# 2. NOW WE CAN SAFELY IMPORT CUSTOM MODULES
from core.vision import HandTracker
from core import config
from patterns import render_ambient_frame
from data_sources.weather import WeatherService
from data_sources.time_of_day import TimeOfDayService
from data_sources.motion import MotionSensor
from data_sources.rss import RSSService
from data_sources.events import EventsService

# 3. IMPORT EVERYTHING ELSE
import json
import queue
import threading
import time

import numpy as np
import pygame
import speech_recognition as sr
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- 1. AI BRAIN SETUP (GEMINI) ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ambient_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "action": types.Schema(
            type=types.Type.STRING,
            enum=["set_location", "set_palette", "set_energy", "set_brightness",
                  "set_mode", "clear_override", "quit", "ignore"],
        ),
        "location_query": types.Schema(type=types.Type.STRING),
        "color_rgb": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
        "base_color_rgb": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
        "energy": types.Schema(type=types.Type.NUMBER),
        "brightness": types.Schema(type=types.Type.NUMBER),
        "mode": types.Schema(type=types.Type.STRING, enum=["finger", "motion"]),
        "reply_message": types.Schema(type=types.Type.STRING),
    },
    required=["action"],
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
                print("\n[VOICE] Ambient listener active! (e.g. 'sync weather to Austin', 'make it calmer')")
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
            print(f"⚠️ Hardware Microphone Error: {outer_e}. Rebooting in 2s...")
            time.sleep(2)


# --- 3. AI WORKER LOOP ---
def ai_worker_loop():
    print("[DEBUG] ai_worker_loop thread has started successfully.")
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

        weather_snapshot = weather_service.get_state()

        try:
            dynamic_prompt = base_prompt.format(
                energy=round(current_energy_for_prompt, 2),
                brightness=round(current_brightness_for_prompt, 2),
                mode=camera_mode,
                location_name=weather_snapshot.get("location_name") or "not set",
            )
        except KeyError as e:
            print(f"Prompt formatting error: missing {e}")
            continue

        success = False
        for model_name in fallback_models:
            if success:
                break
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"{dynamic_prompt}\nUser: {text}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ambient_schema,
                    ),
                )
                payload = json.loads(response.text)
                print(f"🤖 AI Decision ({model_name}): {payload}")
                action_queue.put(payload)
                success = True
            except Exception:
                continue

        if not success:
            print("❌ All AI models failed. Check Network.")


# --- 4. LOAD CONFIGS ---
try:
    with open("led_config.json", "r") as f:
        led_cfg = json.load(f)
    homography_matrix = np.array(led_cfg["homography_matrix"], dtype=np.float32)
    CANVAS_W = int(led_cfg.get("width", 256))
    CANVAS_H = int(led_cfg.get("height", 256)) + 25
except Exception:
    print("Error loading led_config.json.")
    sys.exit()

# --- 5. DATA SOURCES ---
weather_service = WeatherService()
weather_service.start()

time_of_day_service = TimeOfDayService()

motion_sensor = MotionSensor()

rss_service = RSSService()
rss_service.start()

events_service = EventsService()
events_service.start()

hand_tracker = None
camera_mode = "motion"  # default: ambient responds to room activity, not a tracked finger
motion_sensor.start()

# --- 6. MANUAL OVERRIDE STATE (set via voice, cleared with "reset") ---
manual_palette = None      # (base_color, accent_color) or None
manual_energy = None       # float 0-1, or None
manual_brightness = None   # float 0-1, or None

current_energy_for_prompt = 0.3
current_brightness_for_prompt = 0.5

ui_message = ""
ui_message_timer = 0.0

# --- 7. INITIALIZE HARDWARE (Pygame) ---
pygame.init()
OS_W, OS_H = 1920, 1080
try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except Exception:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
font = pygame.font.SysFont(None, 28)
clock = pygame.time.Clock()


def set_camera_mode(new_mode):
    """Only one of MotionSensor / HandTracker may hold the camera at a time."""
    global camera_mode, hand_tracker
    if new_mode == camera_mode:
        return
    if camera_mode == "motion":
        motion_sensor.stop()
    elif camera_mode == "finger" and hand_tracker is not None:
        hand_tracker.stop()
        hand_tracker = None

    if new_mode == "motion":
        motion_sensor.start()
    elif new_mode == "finger":
        hand_tracker = HandTracker(homography_matrix)
        hand_tracker.start()

    camera_mode = new_mode


# --- 8. START THREADS ---
threading.Thread(target=listen_loop, daemon=True).start()
threading.Thread(target=ai_worker_loop, daemon=True).start()

# --- 9. MAIN LOOP ---
running = True
start_time = time.time()

while running:
    t = time.time() - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # --- Handle voice-driven actions ---
    while not action_queue.empty():
        payload = action_queue.get()
        action = payload.get("action")

        if payload.get("reply_message"):
            ui_message = payload["reply_message"]
            ui_message_timer = time.time()

        if action == "ignore":
            continue
        elif action == "quit":
            running = False
        elif action == "set_location":
            query = payload.get("location_query")
            if query:
                success, resolved = weather_service.set_location(query)
                ui_message = f"Weather synced to {resolved}" if success else f"Couldn't find '{query}'"
                ui_message_timer = time.time()
        elif action == "set_palette":
            accent = tuple(payload["color_rgb"]) if payload.get("color_rgb") else (150, 150, 220)
            base = (
                tuple(payload["base_color_rgb"])
                if payload.get("base_color_rgb")
                else tuple(int(c * 0.2) for c in accent)
            )
            manual_palette = (base, accent)
        elif action == "set_energy":
            if payload.get("energy") is not None:
                manual_energy = max(0.0, min(1.0, float(payload["energy"])))
        elif action == "set_brightness":
            if payload.get("brightness") is not None:
                manual_brightness = max(0.0, min(1.0, float(payload["brightness"])))
        elif action == "set_mode":
            requested = payload.get("mode")
            if requested in ("finger", "motion"):
                set_camera_mode(requested)
        elif action == "clear_override":
            manual_palette = None
            manual_energy = None
            manual_brightness = None

    # --- Merge data sources into a single ambient_state ---
    weather_state = weather_service.get_state()
    tod_state = time_of_day_service.get_state()
    motion_state = (
        motion_sensor.get_state() if camera_mode == "motion" else {"available": False, "motion_level": 0.15}
    )

    if manual_palette is not None:
        base_color, accent_color = manual_palette
    else:
        w_base = np.array(weather_state["base_color"], dtype=np.float32)
        w_accent = np.array(weather_state["accent_color"], dtype=np.float32)
        tint = np.array(tod_state["tint_color"], dtype=np.float32)
        base_color = tuple((w_base * 0.7 + tint * 0.3).astype(int).tolist())
        accent_color = tuple((w_accent * 0.7 + tint * 0.3).astype(int).tolist())

    if manual_brightness is not None:
        brightness = manual_brightness
    else:
        brightness = max(0.05, min(1.0, weather_state["brightness"] * tod_state["brightness_factor"] * 1.4))

    if manual_energy is not None:
        energy = manual_energy
    elif camera_mode == "motion":
        energy = max(0.08, min(1.0, motion_state.get("motion_level", 0.15) * 1.6 + 0.08))
    else:
        energy = 0.25  # gentle baseline while in finger-track mode

    current_energy_for_prompt = energy
    current_brightness_for_prompt = brightness

    ambient_state = {
        "base_color": base_color,
        "accent_color": accent_color,
        "brightness": brightness,
        "energy": energy,
    }

    cursor = None
    if camera_mode == "finger" and hand_tracker is not None:
        cursor = hand_tracker.get_cursor()

    # --- Render ---
    render_ambient_frame(screen, CANVAS_W, CANVAS_H, t, ambient_state, cursor=cursor)

    if ui_message and time.time() - ui_message_timer < 4.0:
        msg_surface = font.render(ui_message, True, (255, 255, 255))
        screen.blit(msg_surface, (10, 40))

    pygame.display.flip()
    clock.tick(60)

if hand_tracker is not None:
    hand_tracker.stop()
motion_sensor.stop()
pygame.quit()
os._exit(0)  # Force OS-level termination to kill hanging audio/camera threads
