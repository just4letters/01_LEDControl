"""
screen_calibration.py

Description:
    A raster-mapping tool for LED matrices.
    Combines instantaneous local regex/phonetic parsing with Gemini Voice AI fallback
    to guarantee any grid box (e.g. 33, 102, 307) or movement command works seamlessly.
"""

import os
import sys
import json
import pygame
import queue
import re
from google.genai import types

# --- BULLETPROOF PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core import config
from core.voice import VoiceListener

LED_CONFIG_PATH = config.LED_CONFIG_PATH

# Global state tracker
calibration_state = "GET_TL"


def smart_extract_number(text_lower):
    """
    Instantly extracts whole integers from Whisper transcriptions:
    - Cleans decimals ("1.02" -> 102)
    - Resolves phonetic slips ("one or two" -> 102, "three or seven" -> 307)
    - Converts spoken word numbers ("four" -> 4, "twelve" -> 12)
    """
    # 1. Clean decimal artifacts ("1.02" -> "102")
    cleaned = re.sub(r'(\d+)\.(\d+)', r'\1\2', text_lower)

    # 2. Fix phonetic patterns for 3-digit grid coordinates
    cleaned = re.sub(r'\bone\s+or\s+(\w+)\b', r'10\1', cleaned)
    cleaned = re.sub(r'\btwo\s+or\s+(\w+)\b', r'20\1', cleaned)
    cleaned = re.sub(r'\bthree\s+or\s+(\w+)\b', r'30\1', cleaned)

    # 3. Comprehensive word to number map
    word_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
        "forty": "40", "fifty": "50"
    }

    # First check for raw digits
    digits = re.findall(r'\d+', cleaned)
    if digits:
        return int("".join(digits))

    # Convert spoken word numbers
    words = cleaned.replace("-", " ").replace(".", "").replace(",", "").split()
    for word in words:
        if word in word_map:
            return int(word_map[word])

    return None


def calib_fast_parser(text_lower):
    """Fast-path local parser that runs immediately before Gemini."""
    global calibration_state

    # --- PHASE 1: GRID NUMBER SELECTION ---
    if calibration_state in ["GET_TL", "GET_BR"]:
        num = smart_extract_number(text_lower)
        if num is not None:
            return {"action": "select_number", "number": num}

    # --- PHASE 2: FINE TUNING ---
    elif calibration_state == "FINE_TUNE":
        if any(kw in text_lower for kw in ["looks good", "save", "confirm", "done", "sweet", "perfect", "that's good"]):
            return {"action": "confirm"}

        edge = None
        if "top" in text_lower: edge = "top"
        elif "bottom" in text_lower: edge = "bottom"
        elif "left" in text_lower: edge = "left"
        elif "right" in text_lower: edge = "right"

        direction = None
        if "up" in text_lower or "over" in text_lower: direction = "up" if "top" in text_lower or "bottom" in text_lower else "right"
        if "down" in text_lower: direction = "down"
        elif "left" in text_lower: direction = "left"
        elif "right" in text_lower: direction = "right"

        # Use smart_extract_number to capture spoken numbers like "four" or "12"
        amount = smart_extract_number(text_lower)
        if amount is None:
            amount = 1

        if edge and direction:
            return {
                "action": "move_edge",
                "edge": edge,
                "direction": direction,
                "amount": amount
            }

    return None


# Schema for Gemini fallback
calib_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "action": types.Schema(type=types.Type.STRING, enum=["select_number", "move_edge", "confirm", "quit", "ignore"]),
        "number": types.Schema(type=types.Type.INTEGER),
        "edge": types.Schema(type=types.Type.STRING, enum=["top", "bottom", "left", "right"]),
        "direction": types.Schema(type=types.Type.STRING, enum=["up", "down", "left", "right"]),
        "amount": types.Schema(type=types.Type.INTEGER)
    },
    required=["action"]
)

def get_ai_voice_prompt():
    global calibration_state
    if calibration_state in ["GET_TL", "GET_BR"]:
        return """
        Context: Calibration grid box selection. User is speaking integer box numbers (0-500).
        Extract the integer box number. (e.g., "1.02", "one or two", "102" -> number: 102).
        Return JSON: {"action": "select_number", "number": <integer>}
        """
    else:
        return """
        Context: Bounding outline adjustment.
        Completion: {"action": "confirm"}
        Movement: {"action": "move_edge", "edge": "top"|"bottom"|"left"|"right", "direction": "up"|"down"|"left"|"right", "amount": <integer>}
        """


def run_screen_calibration():
    global calibration_state

    pygame.init()
    OS_W, OS_H = 1920, 1080

    try:
        screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
    except Exception:
        screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)

    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont(None, 28)
    font_large = pygame.font.SysFont(None, 54)

    CELL_SIZE = 60
    cols = (OS_W // CELL_SIZE) + 1
    rows = (OS_H // CELL_SIZE) + 1

    sentence_queue = queue.Queue()
    action_queue = queue.Queue()

    # Fast local parser + Gemini AI fallback
    voice_listener = VoiceListener(
        sentence_queue, 
        action_queue, 
        custom_parser=calib_fast_parser,
        ai_context_prompt=get_ai_voice_prompt,
        response_schema=calib_schema
    )
    voice_listener.start(prompt_message="🎤 Say grid numbers or edge movements...")

    calibration_state = "GET_TL"
    tl_val = None
    br_val = None
    input_str = ""
    
    offset_x, offset_y, width, height = 0, 0, 0, 0
    held_frames = {pygame.K_UP: 0, pygame.K_DOWN: 0, pygame.K_LEFT: 0, pygame.K_RIGHT: 0}

    clock = pygame.time.Clock()
    running = True
    
    while running:
        # --- PROCESS ACTIONS FROM QUEUE ---
        while not action_queue.empty():
            payload = action_queue.get()
            act = payload.get("action")

            if act == "select_number":
                num = payload.get("number")
                if num is not None:
                    if calibration_state == "GET_TL":
                        tl_val = num
                        input_str = ""
                        calibration_state = "GET_BR"
                        print(f"✅ Locked Top-Left Box: {tl_val}")
                    elif calibration_state == "GET_BR":
                        br_val = num
                        calibration_state = "TRANSITION"
                        print(f"✅ Locked Bottom-Right Box: {br_val}")

            elif act == "move_edge" and calibration_state == "FINE_TUNE":
                amt = payload.get("amount", 1)
                edge = payload.get("edge")
                direction = payload.get("direction")

                print(f"📐 Moving {edge} edge {direction} by {amt}px")

                if edge == "top":
                    if direction == "up": offset_y -= amt; height += amt
                    elif direction == "down": offset_y += amt; height -= amt
                elif edge == "bottom":
                    if direction == "up": height -= amt
                    elif direction == "down": height += amt
                elif edge == "left":
                    if direction == "left": offset_x -= amt; width += amt
                    elif direction == "right": offset_x += amt; width -= amt
                elif edge == "right":
                    if direction == "left": width -= amt
                    elif direction == "right": width += amt

            elif act == "confirm" and calibration_state == "FINE_TUNE":
                running = False

            elif act == "quit":
                voice_listener.stop()
                pygame.quit()
                sys.exit()

        # --- PROCESS KEYBOARD EVENTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    voice_listener.stop()
                    pygame.quit()
                    sys.exit()
                    
                if calibration_state in ["GET_TL", "GET_BR"]:
                    if event.key == pygame.K_RETURN and input_str:
                        if calibration_state == "GET_TL":
                            tl_val = int(input_str)
                            input_str = ""
                            calibration_state = "GET_BR"
                        elif calibration_state == "GET_BR":
                            br_val = int(input_str)
                            calibration_state = "TRANSITION"
                    elif event.key == pygame.K_BACKSPACE:
                        input_str = input_str[:-1]
                    elif event.unicode.isdigit():
                        input_str += event.unicode
                        
                elif calibration_state == "FINE_TUNE":
                    if event.key == pygame.K_RETURN:
                        running = False

        # --- RENDER: GRID PHASE ---
        if calibration_state in ["GET_TL", "GET_BR", "TRANSITION"]:
            screen.fill((20, 20, 20))
            
            for r in range(rows):
                for c in range(cols):
                    num = r * cols + c
                    x = c * CELL_SIZE
                    y = r * CELL_SIZE
                    
                    color = (40, 40, 40) if (r + c) % 2 == 0 else (10, 10, 10)
                    pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, (70, 70, 70), (x, y, CELL_SIZE, CELL_SIZE), 1)
                    
                    text = font.render(str(num), True, (220, 220, 220))
                    screen.blit(text, (x + 4, y + 4))

                    if input_str and str(num) == input_str:
                        pygame.draw.rect(screen, (255, 255, 0), (x, y, CELL_SIZE, CELL_SIZE), 4)
                    if tl_val is not None and tl_val == num:
                        pygame.draw.rect(screen, (0, 255, 0), (x, y, CELL_SIZE, CELL_SIZE), 4)
                    if br_val is not None and br_val == num:
                        pygame.draw.rect(screen, (0, 100, 255), (x, y, CELL_SIZE, CELL_SIZE), 4)

            overlay = pygame.Surface((800, 130))
            overlay.fill((30, 30, 40))
            overlay_rect = overlay.get_rect(center=(OS_W // 2, OS_H // 2))
            screen.blit(overlay, overlay_rect)
            pygame.draw.rect(screen, (100, 100, 100), overlay_rect, 2)

            if calibration_state == "GET_TL":
                prompt = f"Say/Type TOP-LEFT number: {input_str}_"
                color = (0, 255, 0)
            elif calibration_state == "GET_BR":
                prompt = f"Say/Type BOTTOM-RIGHT number: {input_str}_"
                color = (0, 100, 255)
            else:
                prompt = "CALCULATING..."
                color = (255, 255, 255)
                
            text_surf = font_large.render(prompt, True, color)
            text_rect = text_surf.get_rect(center=(OS_W // 2, OS_H // 2))
            screen.blit(text_surf, text_rect)
            
            if calibration_state == "TRANSITION":
                pygame.display.flip()
                pygame.time.wait(800)
                
                tl_x = (tl_val % cols) * CELL_SIZE
                tl_y = (tl_val // cols) * CELL_SIZE
                br_x = (br_val % cols) * CELL_SIZE + CELL_SIZE
                br_y = (br_val // cols) * CELL_SIZE + CELL_SIZE

                offset_x, offset_y = tl_x, tl_y
                width, height = br_x - tl_x, br_y - tl_y
                
                calibration_state = "FINE_TUNE"

        # --- RENDER: FINE TUNE PHASE ---
        elif calibration_state == "FINE_TUNE":
            keys = pygame.key.get_pressed()
            shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            for k in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                if keys[k]:
                    held_frames[k] += 1
                    step = 0
                    if held_frames[k] == 1: step = 1
                    elif 20 < held_frames[k] < 60: 
                        if held_frames[k] % 3 == 0: step = 1
                    elif 60 <= held_frames[k] < 120: step = 2
                    elif held_frames[k] >= 120: step = 5
                        
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

            screen.fill((0, 0, 0))
            pygame.draw.rect(screen, (0, 255, 0), (offset_x, offset_y, width, height), 1)
            
            cx, cy = offset_x + width//2, offset_y + height//2
            pygame.draw.line(screen, (0, 150, 0), (cx - 10, cy), (cx + 10, cy), 1)
            pygame.draw.line(screen, (0, 150, 0), (cx, cy - 10), (cx, cy + 10), 1)

        pygame.display.flip()
        clock.tick(60)

    voice_listener.stop()
    pygame.quit()

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