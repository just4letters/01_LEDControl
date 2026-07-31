import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import threading
import math
import sys
import os

# Import our central configuration
from core import config

class HandTracker:
    def __init__(self, homography_matrix):
        self.homography_matrix = np.array(homography_matrix, dtype=np.float32)
        self.lock = threading.Lock()
        self.cursor_x = -100
        self.cursor_y = -100
        self.running = False

        # --- INITIALIZE MEDIAPIPE ---
        if not os.path.exists(config.HAND_MODEL_PATH):
            print(f"❌ Error: MediaPipe model missing at {config.HAND_MODEL_PATH}")
            sys.exit()

        base_options = python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        # --- INITIALIZE CAMERA ---
        self.cap = cv2.VideoCapture(config.FINGER_CAMERA_INDEX)
        
        if not self.cap.isOpened():
            print(f"❌ Error: Could not open hand camera (Index {config.FINGER_CAMERA_INDEX}).")
            sys.exit()
        
        # Apply the central overrides
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)

        self.cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[DEBUG] Hand camera initialized at: {self.cam_w}x{self.cam_h}")

    def start(self):
        self.running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

    def get_cursor(self):
        with self.lock:
            return self.cursor_x, self.cursor_y

    def _camera_loop(self):
        print("[DEBUG] Hand tracking thread started.")
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


class FaceTracker:
    def __init__(self, homography_matrix):
        self.homography_matrix = np.array(homography_matrix, dtype=np.float32)
        self.lock = threading.Lock()
        
        # State variables
        self.cursor_x = -100
        self.cursor_y = -100
        self.blink_score = 0.0
        self.jaw_open = 0.0
        self.running = False

        # --- INITIALIZE MEDIAPIPE (WITH BLENDSHAPES) ---
        if not os.path.exists(config.FACE_MODEL_PATH):
            print(f"❌ Error: MediaPipe model missing at {config.FACE_MODEL_PATH}")
            sys.exit()

        base_options = python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options, 
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

        # --- INITIALIZE CAMERA ---
        self.cap = cv2.VideoCapture(config.FACE_CAMERA_INDEX)
        
        if not self.cap.isOpened():
            print(f"❌ Error: Could not open face camera (Index {config.FACE_CAMERA_INDEX}).")
            sys.exit()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
        print("[DEBUG] Face camera initialized successfully.")

    def start(self):
        self.running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

    def get_state(self):
        """Returns: (X, Y, Blink Score, Jaw Openness)"""
        with self.lock:
            return self.cursor_x, self.cursor_y, self.blink_score, self.jaw_open

    def _camera_loop(self):
        print("[DEBUG] Face tracking thread started.")
        smooth_x, smooth_y = -100, -100
        alpha = 0.35 
        lost_frames = 0 
        
        while self.running:
            success, frame = self.cap.read()
            if success:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)

                if detection_result.face_landmarks:
                    lost_frames = 0 
                    
                    # 1. Extract Blendshapes
                    blink = 0.0
                    jaw = 0.0
                    if detection_result.face_blendshapes:
                        bs = {cat.category_name: cat.score for cat in detection_result.face_blendshapes[0]}
                        blink = (bs.get('eyeBlinkLeft', 0) + bs.get('eyeBlinkRight', 0)) / 2.0
                        jaw = bs.get('jawOpen', 0.0)

                    # 2. Extract 3D Head Pose for smoother pointing
                    if detection_result.facial_transformation_matrixes:
                        R = detection_result.facial_transformation_matrixes[0][:3, :3] 
                        cam_px = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2)))
                        cam_py = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                    else:
                        # Fallback to nose tip if matrix fails
                        nose_tip = detection_result.face_landmarks[0][1]
                        cam_px = nose_tip.x * config.CAM_WIDTH
                        cam_py = nose_tip.y * config.CAM_HEIGHT
                    
                    # 3. Transform and Smooth
                    pt = np.array([[[cam_px, cam_py]]], dtype=np.float32)
                    warped_pt = cv2.perspectiveTransform(pt, self.homography_matrix)
                    
                    target_x, target_y = warped_pt[0][0][0], warped_pt[0][0][1]
                    
                    if smooth_x == -100:
                        smooth_x, smooth_y = target_x, target_y 
                    else:
                        smooth_x = smooth_x + alpha * (target_x - smooth_x)
                        smooth_y = smooth_y + alpha * (target_y - smooth_y)
                    
                    with self.lock:
                        self.cursor_x = int(smooth_x)
                        self.cursor_y = int(smooth_y)
                        self.blink_score = blink
                        self.jaw_open = jaw
                else:
                    lost_frames += 1
                    if lost_frames > 15: 
                        with self.lock:
                            self.cursor_x, self.cursor_y = -100, -100
                            self.blink_score, self.jaw_open = 0.0, 0.0