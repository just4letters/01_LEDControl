import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import threading
import math
import sys
import os
import time  # <-- Added time for video timestamps

# Import our central configuration
from core import config

# =========================================================================
# 📷 GLOBAL CAMERA REGISTRY
# Tracks active camera indices to prevent multiple trackers from hijacking 
# the same fallback camera (index 0) in dual-tracker applications.
# =========================================================================
_ACTIVE_CAMERAS = set()

def _acquire_camera(preferred_index, tracker_name):
    print(f"[{tracker_name}] Attempting to open camera {preferred_index}...")
    
    if preferred_index not in _ACTIVE_CAMERAS:
        cap = cv2.VideoCapture(preferred_index)
        if cap.isOpened():
            _ACTIVE_CAMERAS.add(preferred_index)
            print(f"[{tracker_name}] ✅ Successfully opened camera {preferred_index}.")
            return cap, preferred_index
        cap.release()
    elif preferred_index == 0:
        raise RuntimeError(
            f"\n❌ [{tracker_name}] FATAL ERROR: Camera 0 is already in use!\n"
            f"You need multiple cameras connected to run both trackers simultaneously."
        )

    print(f"[{tracker_name}] ⚠️ Camera {preferred_index} unavailable. Falling back to index 0...")
    
    if 0 in _ACTIVE_CAMERAS:
        raise RuntimeError(
            f"\n❌ [{tracker_name}] FATAL ERROR: Cannot initialize!\n"
            f"Tried to fall back to camera index 0, but it is already in use by another tracker.\n"
            f"You likely need two cameras connected to run both trackers simultaneously."
        )
        
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        _ACTIVE_CAMERAS.add(0)
        print(f"[{tracker_name}] ✅ Successfully opened fallback camera 0.")
        return cap, 0
    cap.release()

    raise RuntimeError(f"[{tracker_name}] ❌ Fatal Error: Could not open camera {preferred_index} or fallback 0.")

def _release_camera(index):
    if index is not None and index in _ACTIVE_CAMERAS:
        _ACTIVE_CAMERAS.remove(index)

# =========================================================================


class HandTracker:
    def __init__(self, homography_matrix):
        self.homography_matrix = np.array(homography_matrix, dtype=np.float32)
        self.lock = threading.Lock()
        self.cursor_x = -100
        self.cursor_y = -100
        self.running = False
        self.active_cam_idx = None 

        if not os.path.exists(config.HAND_MODEL_PATH):
            print(f"❌ Error: MediaPipe model missing at {config.HAND_MODEL_PATH}")
            sys.exit()

        base_options = python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=1,
            running_mode=mp_vision.RunningMode.VIDEO  # <-- ADDED VIDEO MODE
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        self.cap, self.active_cam_idx = _acquire_camera(config.FINGER_CAMERA_INDEX, "HandTracker")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)

    def start(self):
        self.running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        _release_camera(self.active_cam_idx)

    def get_cursor(self):
        with self.lock:
            return self.cursor_x, self.cursor_y

    def _camera_loop(self):
        print("[DEBUG] Hand tracking thread started.")
        smooth_x, smooth_y = -100, -100
        alpha = 0.35 
        lost_frames = 0 
        start_time = time.time()  # <-- ADDED TIMESTAMP TRACKING
        
        while self.running:
            success, frame = self.cap.read()
            if success:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                frame_timestamp_ms = int((time.time() - start_time) * 1000)
                # <-- CHANGED TO detect_for_video
                detection_result = self.detector.detect_for_video(mp_image, frame_timestamp_ms)

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
        
        self.cursor_x = -100
        self.cursor_y = -100
        self.blink_score = 0.0
        self.jaw_open = 0.0
        self.running = False
        self.active_cam_idx = None 

        if not os.path.exists(config.FACE_MODEL_PATH):
            print(f"❌ Error: MediaPipe model missing at {config.FACE_MODEL_PATH}")
            sys.exit()

        base_options = python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options, 
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp_vision.RunningMode.VIDEO  # <-- ADDED VIDEO MODE
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

        self.cap, self.active_cam_idx = _acquire_camera(config.FACE_CAMERA_INDEX, "FaceTracker")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
        print("[DEBUG] Face camera initialized successfully.")

    def start(self):
        self.running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        _release_camera(self.active_cam_idx)

    def get_state(self):
        with self.lock:
            return self.cursor_x, self.cursor_y, self.blink_score, self.jaw_open

    def _camera_loop(self):
        print("[DEBUG] Face tracking thread started.")
        smooth_x, smooth_y = -100, -100
        alpha = 0.35 
        lost_frames = 0 
        start_time = time.time()  # <-- ADDED TIMESTAMP TRACKING
        
        while self.running:
            success, frame = self.cap.read()
            if success:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                frame_timestamp_ms = int((time.time() - start_time) * 1000)
                # <-- CHANGED TO detect_for_video
                detection_result = self.detector.detect_for_video(mp_image, frame_timestamp_ms)

                if detection_result.face_landmarks:
                    lost_frames = 0 
                    
                    blink = 0.0
                    jaw = 0.0
                    if detection_result.face_blendshapes:
                        bs = {cat.category_name: cat.score for cat in detection_result.face_blendshapes[0]}
                        blink = (bs.get('eyeBlinkLeft', 0) + bs.get('eyeBlinkRight', 0)) / 2.0
                        jaw = bs.get('jawOpen', 0.0)

                    if detection_result.facial_transformation_matrixes:
                        R = detection_result.facial_transformation_matrixes[0][:3, :3] 
                        cam_px = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2)))
                        cam_py = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                    else:
                        nose_tip = detection_result.face_landmarks[0][1]
                        cam_px = nose_tip.x * config.CAM_WIDTH
                        cam_py = nose_tip.y * config.CAM_HEIGHT
                    
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