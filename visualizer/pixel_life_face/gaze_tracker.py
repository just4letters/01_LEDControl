import os
import cv2
import json
import math
import time
import threading
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GazeTracker:
    def __init__(self, config_path="gaze_config.json"):
        print("👁️ Initializing Background Gaze Tracker...")
        
        # 1. Load Calibration Config
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            self.matrix = np.array(config["homography_matrix"], dtype=np.float32)
            self.canvas_w = config.get("canvas_width", 256)
            self.canvas_h = config.get("canvas_height", 256)
            print(f"✅ Gaze config loaded! Canvas: {self.canvas_w}x{self.canvas_h}")
        except Exception as e:
            print(f"⚠️ Error loading {config_path}: {e}")
            print("Did you run gaze_calibration.py first?")
            self.matrix = None
            self.canvas_w, self.canvas_h = 256, 256

        # 2. Setup MediaPipe
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("⏳ Downloading model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False, # Not needed for pure blendshape tracking
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # 3. State & Smoothing Variables
        self.current_x = self.canvas_w // 2
        self.current_y = self.canvas_h // 2
        self.lock = threading.Lock()
        self.running = True
        
        # 4. Start Background Thread
        self.cap = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    def _tracking_loop(self):
        """Runs continuously in the background to process frames."""
        
        # Smoothing parameters
        smooth_x, smooth_y = self.canvas_w / 2, self.canvas_h / 2
        deadzone_radius = 2.0  # Pixels of micro-jitter to completely ignore
        
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)

            # LAYER 1: Blink Protection
            # If no face/eyes are detected, do nothing. The cursor stays exactly where it was.
            if results.face_blendshapes and self.matrix is not None:
                bs = {cat.category_name: cat.score for cat in results.face_blendshapes[0]}
                
                # Extract Eye Rotation
                r_yaw = bs.get('eyeLookOutRight', 0) - bs.get('eyeLookInRight', 0)
                l_yaw = bs.get('eyeLookInLeft', 0) - bs.get('eyeLookOutLeft', 0)
                eye_yaw = (r_yaw + l_yaw) / 2.0
                
                r_pitch = bs.get('eyeLookDownRight', 0) - bs.get('eyeLookUpRight', 0)
                l_pitch = bs.get('eyeLookDownLeft', 0) - bs.get('eyeLookUpLeft', 0)
                eye_pitch = (r_pitch + l_pitch) / 2.0

                # Transform via Homography Matrix
                src_pt = np.array([[[eye_yaw, eye_pitch]]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(src_pt, self.matrix)
                raw_x, raw_y = dst_pt[0][0][0], dst_pt[0][0][1]

                # Distance from current smooth position to new raw position
                dist = math.hypot(raw_x - smooth_x, raw_y - smooth_y)

                # LAYER 2: The Saccade Deadzone
                if dist > deadzone_radius:
                    
                    # LAYER 3: Dynamic Exponential Smoothing
                    # If darting across screen (high dist), alpha approaches 1.0 (instant).
                    # If focusing on a spot (low dist), alpha drops to ~0.05 (heavy smoothing).
                    dynamic_alpha = min(1.0, 0.05 + (dist / 100.0))
                    
                    smooth_x = (dynamic_alpha * raw_x) + ((1.0 - dynamic_alpha) * smooth_x)
                    smooth_y = (dynamic_alpha * raw_y) + ((1.0 - dynamic_alpha) * smooth_y)

                # Clamp to canvas boundaries safely
                final_x = max(0, min(self.canvas_w, int(smooth_x)))
                final_y = max(0, min(self.canvas_h, int(smooth_y)))

                # Thread-safe update of the exposed variables
                with self.lock:
                    self.current_x = final_x
                    self.current_y = final_y

    def get_gaze_position(self):
        """Called by app.py to get the instantly available cursor coordinates."""
        with self.lock:
            return self.current_x, self.current_y

    def stop(self):
        """Safely shuts down the camera thread."""
        self.running = False
        self.thread.join()
        self.cap.release()
        print("🛑 Gaze Tracker gracefully shut down.")

# Simple test block: run this file directly to see it spit out coordinates
if __name__ == "__main__":
    tracker = GazeTracker()
    try:
        while True:
            x, y = tracker.get_gaze_position()
            print(f"Cursor locked at: X: {x}, Y: {y}", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        tracker.stop()