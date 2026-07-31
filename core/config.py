"""
core/config.py

Centralized traffic control for all file paths, hardware settings, and global constants.
Import paths from here rather than hardcoding them in individual visualizers.
"""
import os

# --- PATH ROUTING ---
# config.py lives in 'core/', so we go up one level to find the root.
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CORE_DIR, '..'))

# Configurations
LED_CONFIG_PATH = os.path.join(ROOT_DIR, "led_config.json")
NOSE_CONFIG_PATH = os.path.join(ROOT_DIR, "nose_config.json")
FINGER_CONFIG_PATH = os.path.join(ROOT_DIR, "finger_config.json")

# AI Models
MODELS_DIR = os.path.join(ROOT_DIR, "models")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

# Tools / Instructions
CALIB_INSTRUCTIONS_PATH = os.path.join(ROOT_DIR, "tools", "calib_instructions.txt")

# --- HARDWARE SETTINGS (Keep your existing ones here) ---
CAMERA_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 60

# --- DEVICE OVERRIDES ---
# Change these if you want different cameras for different tasks
# (e.g., 0 = Laptop webcam, 1 = External USB webcam)
FACE_CAMERA_INDEX = 1
FINGER_CAMERA_INDEX = 0

# --- AUDIO & VOICE SETTINGS ---
# Adjust this based on room noise. (Lower = more sensitive, Higher = less sensitive)
MIC_ENERGY_THRESHOLD = 100