"""
screen_calibration.py

Description:
    A raster-mapping tool for LED matrices.
    1. Generates a numbered grid across the entire OS display.
    2. Listens via Whisper (or keyboard) for the Top-Left and Bottom-Right numbers.
    3. Highlights the boxes as you speak or type them.
    4. Enters an interactive tuning mode with accelerating arrow keys.
    5. Saves the exact X, Y, Width, and Height to 'led_config.json' in the root directory.
"""

import os
import sys
import json
import pygame
import threading
import queue
import re
import time
import speech_recognition as sr

# --- BULLETPROOF PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
LED_CONFIG_PATH = os.path.join(ROOT_DIR, "led_config.json")

voice_queue = queue.Queue()

# --- VOICE LISTENER THREAD ---
def voice_listener():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 600
    
    # Simple fallback dictionary just in case Whisper outputs a raw word instead of a digit
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, 
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    
    try:
        with sr.Microphone(sample_rate=16000) as source:
            while True:
                try:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    text = recognizer.recognize_whisper(audio, model="base.en").strip().lower()
                    
                    if text:
                        print(f"🎤 [Heard]: {text}")
                        # Look for digits first (Whisper usually outputs '35' instead of 'thirty five')
                        digits = re.findall(r'\d+', text)
                        if digits:
                            voice_queue.put(int(digits[0]))
                            continue
                            
                        # Fallback to word dictionary
                        words = text.replace(".", "").replace(",", "").split()
                        for word in words:
                            if word in word_to_num:
                                voice_queue.put(word_to_num[word])
                                break
                                
                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    pass
    except Exception as e:
        print(f"⚠️ Microphone error: {e}")

def run_screen_calibration():
    pygame.init()
    OS_W, OS_H = 1920, 1080

    try:
        screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
    except Exception:
        screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)

    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont(None, 36)
    font_large = pygame.font.SysFont(None, 64)
    CELL_SIZE = 80
    cols = (OS_W // CELL_SIZE) + 1
    rows = (OS_H // CELL_SIZE) + 1

    # Start Voice Thread
    threading.Thread(target=voice_listener, daemon=True).start()

    # --- STATE VARIABLES ---
    state = "GET_TL"
    tl_val = None
    br_val = None
    input_str = ""
    
    offset_x, offset_y, width, height = 0, 0, 0, 0
    
    # Key acceleration tracking
    held_frames = {pygame.K_UP: 0, pygame.K_DOWN: 0, pygame.K_LEFT: 0, pygame.K_RIGHT: 0}

    clock = pygame.time.Clock()
    running = True
    
    while running:
        # --- PROCESS VOICE QUEUE ---
        voice_num = None
        while not voice_queue.empty():
            voice_num = voice_queue.get()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                    
                if state in ["GET_TL", "GET_BR"]:
                    if event.key == pygame.K_RETURN and input_str:
                        if state == "GET_TL":
                            tl_val = int(input_str)
                            input_str = ""
                            state = "GET_BR"
                        elif state == "GET_BR":
                            br_val = int(input_str)
                            state = "TRANSITION"
                    elif event.key == pygame.K_BACKSPACE:
                        input_str = input_str[:-1]
                    elif event.unicode.isdigit():
                        input_str += event.unicode
                        
                elif state == "FINE_TUNE":
                    if event.key == pygame.K_RETURN:
                        running = False

        # Apply voice commands if received
        if voice_num is not None:
            if state == "GET_TL":
                tl_val = voice_num
                input_str = ""
                state = "GET_BR"
            elif state == "GET_BR":
                br_val = voice_num
                state = "TRANSITION"

        # --- RENDER: GRID PHASE ---
        if state in ["GET_TL", "GET_BR", "TRANSITION"]:
            screen.fill((20, 20, 20))
            
            # Draw Grid
            for r in range(rows):
                for c in range(cols):
                    num = r * cols + c
                    x = c * CELL_SIZE
                    y = r * CELL_SIZE
                    
                    # Checkerboard
                    color = (40, 40, 40) if (r + c) % 2 == 0 else (10, 10, 10)
                    pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, (80, 80, 80), (x, y, CELL_SIZE, CELL_SIZE), 1)
                    
                    # Numbers
                    text = font.render(str(num), True, (255, 255, 255))
                    screen.blit(text, (x + 10, y + 25))

                    # Live Typing Highlight
                    if input_str and str(num) == input_str:
                        pygame.draw.rect(screen, (255, 255, 0), (x, y, CELL_SIZE, CELL_SIZE), 5)
                        
                    # Locked TL Highlight (Green)
                    if tl_val is not None and tl_val == num:
                        pygame.draw.rect(screen, (0, 255, 0), (x, y, CELL_SIZE, CELL_SIZE), 5)
                        
                    # Locked BR Highlight (Blue)
                    if br_val is not None and br_val == num:
                        pygame.draw.rect(screen, (0, 100, 255), (x, y, CELL_SIZE, CELL_SIZE), 5)

            # Draw Instructions Overlay
            overlay = pygame.Surface((850, 150))
            overlay.fill((30, 30, 40))
            overlay_rect = overlay.get_rect(center=(OS_W // 2, OS_H // 2))
            screen.blit(overlay, overlay_rect)
            pygame.draw.rect(screen, (100, 100, 100), overlay_rect, 2)

            if state == "GET_TL":
                prompt = f"Say or Type the TOP-LEFT number: {input_str}_"
                color = (0, 255, 0)
            elif state == "GET_BR":
                prompt = f"Say or Type the BOTTOM-RIGHT number: {input_str}_"
                color = (0, 100, 255)
            else:
                prompt = "CALCULATING..."
                color = (255, 255, 255)
                
            text_surf = font_large.render(prompt, True, color)
            text_rect = text_surf.get_rect(center=(OS_W // 2, OS_H // 2))
            screen.blit(text_surf, text_rect)
            
            # Transition Logic
            if state == "TRANSITION":
                pygame.display.flip()
                pygame.time.wait(1000)
                
                # Calculate initial rough bounding box
                tl_x = (tl_val % cols) * CELL_SIZE
                tl_y = (tl_val // cols) * CELL_SIZE
                br_x = (br_val % cols) * CELL_SIZE + CELL_SIZE
                br_y = (br_val // cols) * CELL_SIZE + CELL_SIZE

                offset_x, offset_y = tl_x, tl_y
                width, height = br_x - tl_x, br_y - tl_y
                
                state = "FINE_TUNE_INSTRUCTIONS"

        # --- RENDER: INSTRUCTIONS PHASE ---
        elif state == "FINE_TUNE_INSTRUCTIONS":
            screen.fill((0, 0, 0))
            lines = [
                "FINE TUNING MODE",
                "",
                "[ ARROW KEYS ] = Move Box (Hold to accelerate)",
                "[ SHIFT + ARROWS ] = Resize Box",
                "[ ENTER ] = Save and Quit"
            ]
            for i, line in enumerate(lines):
                text_surf = font_large.render(line, True, (0, 255, 255) if i==0 else (255,255,255))
                screen.blit(text_surf, (OS_W//2 - 300, OS_H//2 - 100 + (i*60)))
            pygame.display.flip()
            pygame.time.wait(3500)
            state = "FINE_TUNE"

        # --- RENDER: FINE TUNE PHASE ---
        elif state == "FINE_TUNE":
            keys = pygame.key.get_pressed()
            shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            # Key Acceleration Logic
            for k in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                if keys[k]:
                    held_frames[k] += 1
                    step = 0
                    
                    # Fire immediately on press
                    if held_frames[k] == 1:
                        step = 1
                    # Wait a moment, then start slow repeat
                    elif held_frames[k] > 20 and held_frames[k] < 60:
                        if held_frames[k] % 3 == 0: step = 1
                    # Medium acceleration
                    elif held_frames[k] >= 60 and held_frames[k] < 120:
                        step = 2
                    # Max acceleration
                    elif held_frames[k] >= 120:
                        step = 5
                        
                    if step > 0:
                        if shift_pressed:
                            if k == pygame.K_UP: height -= step
                            elif k == pygame.K_DOWN: height += step
                            elif k == pygame.K_LEFT: width -= step
                            elif k == pygame.K_RIGHT: width += step
                        else:
                            if k == pygame.K_UP: offset_y -= step
                            elif k == pygame.K_DOWN: offset_y += step
                            elif k == pygame.K_LEFT: offset_x -= step
                            elif k == pygame.K_RIGHT: offset_x += step
                else:
                    held_frames[k] = 0

            # Render black screen with framing borders
            screen.fill((0, 0, 0))
            pygame.draw.rect(screen, (0, 255, 0), (offset_x, offset_y, width, height), 1)
            
            # Crosshair
            cx, cy = offset_x + width//2, offset_y + height//2
            pygame.draw.line(screen, (0, 150, 0), (cx - 10, cy), (cx + 10, cy), 1)
            pygame.draw.line(screen, (0, 150, 0), (cx, cy - 10), (cx, cy + 10), 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    # --- SAVE AND EXIT ---
    config_data = {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "width": width,
        "height": height
    }
    
    with open(LED_CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"\n🎉 SAVED! Your LED Canvas is {width}x{height} at coordinates ({offset_x}, {offset_y}).")

if __name__ == "__main__":
    run_screen_calibration()