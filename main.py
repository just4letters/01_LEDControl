"""
main.py

Unified system launcher and visualizer manager.
Routes voice commands, handles pre-flight configuration checks,
and launches sub-processes clean of Pygame holds.
"""

import os
# 1. Mute redundant C++ info/warning logs from MediaPipe & TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=All, 1=Filter INFO, 2=Filter INFO/WARNING, 3=Filter ERRORS
os.environ["GLOG_minloglevel"] = "2"        # Silences MediaPipe internal C++ warnings

# 2. Prevent macOS Objective-C duplicate library warnings (SDL2/Pygame)
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
import subprocess
import sys
import threading
import pygame
import json
import queue

# --- PATH & MODULE IMPORTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from core.config import LED_CONFIG_PATH, ROOT_DIR
from core.voice import VoiceListener

pending_launch = None

# --- LOAD LED CONFIGURATION ---
try:
    with open(LED_CONFIG_PATH, "r") as f:
        led_cfg = json.load(f)
    CANVAS_W = int(led_cfg.get("width", 256))
    CANVAS_H = int(led_cfg.get("height", 256))
    OFFSET_X = int(led_cfg.get("offset_x", 0))
    OFFSET_Y = int(led_cfg.get("offset_y", 0))
except Exception:
    CANVAS_W, CANVAS_H, OFFSET_X, OFFSET_Y = 256, 256, 0, 0

# --- THE CALIBRATION ROUTING DICTIONARY ---
CALIBRATION_ROUTER = {
    "led_config.json": "tools/screen_calibration.py",
    "nose_config.json": "tools/nose_calibration.py",
    "finger_config.json": "tools/finger_calibrate_voice.py"
}

# Explicit tools for when the user asks to recalibrate manually
TOOLS = [
    {"name": "Recalibrate Screen", "path": "tools/screen_calibration.py", "keywords": ["screen", "led", "display"]},
    {"name": "Recalibrate Nose", "path": "tools/nose_calibration.py", "keywords": ["nose", "face", "head"]},
    {"name": "Recalibrate Finger", "path": "tools/finger_calibrate_voice.py", "keywords": ["finger", "voice", "hand"]}
]


def get_available_apps():
    """
    Builds a unified registry of visualizers and explicitly defines what configs they REQUIRE.
    """
    apps = [
        {
            "name": "Black Hole Sandbox", 
            "path": "visualizer/pixel_life_face/app.py", 
            "keywords": ["black hole", "sandbox", "physics"],
            "requires": ["led_config.json", "nose_config.json"]
        }
    ]
    
    # Dynamically scan for visualizers using standard app.py naming
    vis_dir = os.path.join(SCRIPT_DIR, "visualizer")
    if os.path.exists(vis_dir):
        folders = [d for d in os.listdir(vis_dir) 
                if os.path.isdir(os.path.join(vis_dir, d)) and not d.startswith('_') and not d.startswith('.')]
        folders.sort()
        
        for folder in folders:
            if folder == "pixel_life_face": 
                continue 
                
            potential_path = os.path.join(vis_dir, folder, "app.py")
            if os.path.exists(potential_path):
                clean_name = folder.replace("_", " ").title()
                apps.append({
                    "name": clean_name, 
                    "path": f"visualizer/{folder}/app.py", 
                    "keywords": [folder.replace("_", " "), clean_name.lower()],
                    "requires": ["led_config.json"]
                })
                
    return apps


def launch_app(app_dict):
    """
    Handles pre-flight checks, kills Pygame, and boots the new process.
    """
    # 1. PRE-FLIGHT CHECK
    missing_configs = []
    for req in app_dict.get("requires", []):
        if not os.path.exists(os.path.join(SCRIPT_DIR, req)):
            missing_configs.append(req)
            
    if missing_configs:
        target_config = missing_configs[0]
        tool_path = CALIBRATION_ROUTER.get(target_config)
        
        if tool_path:
            print(f"⚠️ Missing '{target_config}'! Rerouting to calibration tool...")
            script_path = os.path.join(SCRIPT_DIR, tool_path)
        else:
            print(f"❌ Error: Missing '{target_config}' and no tool is registered to create it.")
            return
    else:
        print(f"\n🚀 Booting engine for: {app_dict['name']}...")
        script_path = os.path.join(SCRIPT_DIR, app_dict['path'])
    
    # 2. FREE THE DISPLAY
    pygame.quit() 
    
    # 3. LAUNCH PROCESS
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SCRIPT_DIR}:{existing_pythonpath}" if existing_pythonpath else SCRIPT_DIR
    
    try:
        subprocess.run([sys.executable, script_path], env=env)
    except KeyboardInterrupt:
        pass
        
    os._exit(0)


def create_launcher_parser(apps):
    """
    Returns a custom_parser callback tailored to launcher navigation.
    """
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}

    def launcher_commands(text_lower):
        global pending_launch

        # 1. Command: Recalibrate
        if "calibrate" in text_lower or "recalibrate" in text_lower:
            for tool in TOOLS:
                for kw in tool["keywords"]:
                    if kw in text_lower:
                        pending_launch = tool
                        return {"action": "recalibrate"}

        # 2. Command: Launch by Name / Keyword
        for app in apps:
            for kw in app["keywords"]:
                if kw in text_lower:
                    pending_launch = app
                    return {"action": "launch"}

        # 3. Command: Launch by Number
        words = text_lower.replace(".", "").replace(",", "").split()
        for word in words:
            num = word_to_num.get(word)
            if num is None and word.isdigit():
                num = int(word)
            if num and 1 <= num <= len(apps):
                pending_launch = apps[num - 1]
                return {"action": "launch"}

        return None

    return launcher_commands


def main():
    global pending_launch
    apps = get_available_apps()
    
    sentence_queue = queue.Queue()
    action_queue = queue.Queue()

    # Hook up central VoiceListener using the launcher custom parser
    voice_listener = VoiceListener(
        sentence_queue, 
        action_queue, 
        custom_parser=create_launcher_parser(apps)
    )
    voice_listener.start(prompt_message="🎤 Voice listener active... Say App Name or Number")

    pygame.init()
    try:
        screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN, display=1)
    except Exception:
        screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
        
    pygame.mouse.set_visible(False)
    canvas = pygame.Surface((CANVAS_W, CANVAS_H))
    
    title_size = max(12, int(CANVAS_W * 0.08))
    item_size = max(10, int(CANVAS_W * 0.06))
    
    font_title = pygame.font.SysFont(None, title_size)
    font_item = pygame.font.SysFont(None, item_size)
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Check action queue for quit commands from voice_listener
        while not action_queue.empty():
            payload = action_queue.get()
            if payload.get("action") == "quit":
                running = False

        if pending_launch:
            voice_listener.stop()
            launch_app(pending_launch)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(apps):
                        launch_app(apps[idx])
                elif event.key == pygame.K_s:
                    launch_app(TOOLS[0]) # Screen
                elif event.key == pygame.K_n:
                    launch_app(TOOLS[1]) # Nose
                elif event.key == pygame.K_f:
                    launch_app(TOOLS[2]) # Finger

        canvas.fill((10, 10, 15))
        
        margin_x = int(CANVAS_W * 0.05)
        start_y = int(CANVAS_H * 0.05)
        line_height = int(item_size * 1.5)
        
        canvas.blit(font_title.render("SYSTEM LAUNCHER", True, (0, 255, 255)), (margin_x, start_y))
        
        for i, app in enumerate(apps):
            y_pos = start_y + (title_size * 2) + (i * line_height)
            if y_pos < CANVAS_H - (line_height * 3): 
                canvas.blit(font_item.render(f"[{i + 1}] {app['name']}", True, (200, 200, 200)), (margin_x, y_pos))
            
        # UI Footer
        mic_y = CANVAS_H - int(item_size * 3)
        canvas.blit(font_item.render("🎤 Say App Name or Number", True, (100, 150, 100)), (margin_x, mic_y))
        canvas.blit(font_item.render("🎤 Say 'Recalibrate [Nose/Screen]'", True, (100, 100, 150)), (margin_x, mic_y + line_height))
        
        screen.fill((0, 0, 0))
        screen.blit(canvas, (OFFSET_X, OFFSET_Y))
        pygame.display.flip()
        clock.tick(30)
        
    voice_listener.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()