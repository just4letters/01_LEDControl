import os
import subprocess
import sys
import threading
import pygame
import speech_recognition as sr

pending_launch = None

def get_available_apps():
    vis_dir = "visualizer"
    if not os.path.exists(vis_dir):
        print(f"❌ Error: Could not find '{vis_dir}'")
        sys.exit()
    apps = [d for d in os.listdir(vis_dir) 
            if os.path.isdir(os.path.join(vis_dir, d)) and not d.startswith('_') and not d.startswith('.')]
    apps.sort()
    return apps

def launch_app(app_name):
    print(f"\n🚀 Booting engine for visualizer: {app_name.upper()}...")
    script_path = os.path.join("visualizer", app_name, "app.py")
    
    # CRITICAL: We must free the GPU/Display before opening the next app!
    pygame.quit() 
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    try:
        subprocess.run([sys.executable, script_path], env=env)
    except KeyboardInterrupt:
        pass
    os._exit(0)

def voice_listener(apps):
    global pending_launch
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 600
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    
    try:
        with sr.Microphone(sample_rate=16000) as source:
            while True:
                try:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    text = recognizer.recognize_whisper(audio, model="base.en").strip().lower()
                    
                    if text:
                        for app in apps:
                            if app.replace("_", " ") in text:
                                pending_launch = app
                        
                        words = text.replace(".", "").replace(",", "").split()
                        for word in words:
                            num = word_to_num.get(word)
                            if num is None and word.isdigit():
                                num = int(word)
                            if num and 1 <= num <= len(apps):
                                pending_launch = apps[num-1]
                        
                        if "quit" in text or "exit" in text:
                            os._exit(0)
                except Exception:
                    pass
    except Exception:
        pass

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
    font_title = pygame.font.SysFont(None, 64)
    font_item = pygame.font.SysFont(None, 48)
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
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(apps):
                        launch_app(apps[idx])

        screen.fill((0, 0, 0))
        
        # Draw Launcher UI
        screen.blit(font_title.render("SYSTEM LAUNCHER", True, (0, 255, 255)), (100, 100))
        for i, app in enumerate(apps):
            clean_name = app.replace("_", " ").title()
            screen.blit(font_item.render(f"[{i + 1}] - {clean_name}", True, (255, 255, 255)), (150, 200 + (i * 60)))
            
        screen.blit(font_item.render("🎤 Voice Active: Say the name or number...", True, (100, 100, 100)), (150, 800))
        
        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()