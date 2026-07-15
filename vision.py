import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import threading
import sys
import os

# Import our central configuration
import config

class HandTracker:
    def __init__(self, homography_matrix):
        self.homography_matrix = np.array(homography_matrix, dtype=np.float32)
        self.lock = threading.Lock()
        self.cursor_x = -100
        self.cursor_y = -100
        self.running = False

        # --- INITIALIZE MEDIAPIPE ---
        MP_MODEL_PATH = 'hand_landmarker.task'
        if not os.path.exists(MP_MODEL_PATH):
            print("❌ Error: MediaPipe AI model missing (hand_landmarker.task).")
            sys.exit()

        base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        # --- INITIALIZE CAMERA ---
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera.")
            sys.exit()
        
        # Apply the central overrides
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)

        # Read back the actual hardware values to be safe
        self.cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[DEBUG] Camera initialized in tracking thread at: {self.cam_w}x{self.cam_h}")

    def start(self):
        """Starts the background tracking thread."""
        self.running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def stop(self):
        """Safely shuts down the camera."""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

    def get_cursor(self):
        """Returns the smoothed X and Y coordinates safely."""
        with self.lock:
            return self.cursor_x, self.cursor_y

    def _camera_loop(self):
        print("[DEBUG] Vision thread started successfully.")
        smooth_x, smooth_y = -100, -100
        alpha = 0.35 
        lost_frames = 0 
        
        while self.running:
            success, frame = self.cap.read()
            if success:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)

                if detection_result.hand_landmarks:
                    lost_frames = 0 
                    index_finger = detection_result.hand_landmarks[0][8]
                    
                    # Multiply normalized finger position (0.0 to 1.0) by configured camera size
                    cam_px = index_finger.x * config.CAM_WIDTH
                    cam_py = index_finger.y * config.CAM_HEIGHT
                    
                    pt = np.array([[[cam_px, cam_py]]], dtype=np.float32)
                    warped_pt = cv2.perspectiveTransform(pt, self.homography_matrix)
                    
                    target_x = warped_pt[0][0][0]
                    target_y = warped_pt[0][0][1]
                    
                    if smooth_x == -100:
                        smooth_x, smooth_y = target_x, target_y 
                    else:
                        smooth_x = smooth_x + alpha * (target_x - smooth_x)
                        smooth_y = smooth_y + alpha * (target_y - smooth_y)
                    
                    with self.lock:
                        self.cursor_x = int(smooth_x)
                        self.cursor_y = int(smooth_y)
                else:
                    lost_frames += 1
                    if lost_frames > 15: 
                        smooth_x, smooth_y = -100, -100
                        with self.lock:
                            self.cursor_x, self.cursor_y = -100, -100