import os
import subprocess
import sys
import threading
import pygame
import json
import speech_recognition as sr

pending_launch = None

# --- LOAD LED CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LED_CONFIG_PATH = os.path.join(SCRIPT_DIR, "led_config.json")

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
# This maps the missing config file to the tool that creates it
CALIBRATION_ROUTER = {
    "led_config.json": "tools/screen_calibration.py",
    "nose_config.json": "tools/nose_calibration.py",
    "finger_config.json": "tools/finger_calibrate_voice.py" # Update this filename to whatever your script saves!
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
            "path": "visualizer/pixel_life_face/nose_blink_app.py", 
            "keywords": ["black hole", "sandbox", "physics"],
            "requires": ["led_config.json", "nose_config.json"] # <--- SET REQUIREMENTS HERE
        }
    ]
    
    # Dynamically scan for older visualizers using the standard app.py naming
    vis_dir = os.path.join(SCRIPT_DIR, "visualizer")
    if os.path.exists(vis_dir):
        folders = [d for d in os.listdir(vis_dir) 
                if os.path.isdir(os.path.join(vis_dir, d)) and not d.startswith('_') and not d.startswith('.')]
        folders.sort()
        
        for folder in folders:
            if folder == "pixel_life_face": continue # Skip our manual entry
                
            potential_path = os.path.join(vis_dir, folder, "app.py")
            if os.path.exists(potential_path):
                clean_name = folder.replace("_", " ").title()
                apps.append({
                    "name": clean_name, 
                    "path": f"visualizer/{folder}/app.py", 
                    "keywords": [folder.replace("_", " "), clean_name.lower()],
                    "requires": ["led_config.json"] # <--- DEFAULT REQUIREMENT FOR OLD APPS
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
        # All clear!
        print(f"\n🚀 Booting engine for: {app_dict['name']}...")
        script_path = os.path.join(SCRIPT_DIR, app_dict['path'])
    
    # 2. FREE THE DISPLAY
    pygame.quit() 
    
    # 3. LAUNCH PROCESS
    env = os.environ.copy()
    env["PYTHONPATH"] = SCRIPT_DIR
    
    try:
        subprocess.run([sys.executable, script_path], env=env)
    except KeyboardInterrupt:
        pass
        
    os._exit(0)


def voice_listener(apps):
    global pending_launch
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 600
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    
    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("🎤 Voice listener active...")
            while True:
                try:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    text = recognizer.recognize_whisper(audio, model="base.en").strip().lower()
                    
                    if text:
                        print(f"Heard: '{text}'")
                        
                        # Command: Exit
                        if "quit" in text or "exit" in text or "stop" in text:
                            os._exit(0)
                            
                        # Command: Recalibrate
                        if "calibrate" in text or "recalibrate" in text:
                            for tool in TOOLS:
                                for kw in tool["keywords"]:
                                    if kw in text:
                                        pending_launch = tool
                                        break
                            if pending_launch: continue
                        
                        # Command: Launch by Name
                        for app in apps:
                            for kw in app["keywords"]:
                                if kw in text:
                                    pending_launch = app
                                    break
                        if pending_launch: continue
                        
                        # Command: Launch by Number
                        words = text.replace(".", "").replace(",", "").split()
                        for word in words:
                            num = word_to_num.get(word)
                            if num is None and word.isdigit():
                                num = int(word)
                            if num and 1 <= num <= len(apps):
                                pending_launch = apps[num-1]
                                break
                                
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ Microphone error: {e}")


def main():
    global pending_launch
    apps = get_available_apps()
    
    threading.Thread(target=voice_listener, args=(apps,), daemon=True).start()
    
    pygame.init()
    try:
        screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN, display=1)
    except:
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
        if pending_launch:
            launch_app(pending_launch)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                # Trigger apps via number keys
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(apps):
                        launch_app(apps[idx])
                # Trigger recalibration via letter keys
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
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()