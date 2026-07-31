import os
import cv2
import json
import numpy as np
import pygame
import urllib.request
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("🎬 Initializing True 3D AI Gaze Calibrator...")

# 1. Load LED Configuration
led_config_path = "led_config.json"
try:
    with open(led_config_path, "r") as f:
        led_cfg = json.load(f)
    CANVAS_W = int(led_cfg.get("width", 256))
    CANVAS_H = int(led_cfg.get("height", 256))
    print(f"✅ Loaded LED config: Canvas Size = {CANVAS_W}x{CANVAS_H}")
except Exception as e:
    print(f"⚠️ Could not load {led_config_path} ({e}). Defaulting to 256x256.")
    CANVAS_W, CANVAS_H = 256, 256

# 2. Download MediaPipe Model if missing
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("⏳ Downloading face_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

# 3. Setup MediaPipe Face Landmarker (Now with 3D Features Enabled!)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=True,                  # Track exact facial muscles & eye rotation
    output_facial_transformation_matrixes=True,    # Track exact 3D head rotation (Yaw/Pitch/Roll)
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.FaceLandmarker.create_from_options(options)

# 4. Setup Camera & Pygame
cap = cv2.VideoCapture(0)
pygame.init()

OS_W, OS_H = 1920, 1080
try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except Exception:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)

pygame.mouse.set_visible(False)
font = pygame.font.SysFont(None, 22)

# 5. Define Calibration Targets
padding = max(15, int(CANVAS_W * 0.1))
targets_canvas = [
    (padding, padding),                        
    (CANVAS_W - padding, padding),             
    (CANVAS_W - padding, CANVAS_H - padding),  
    (padding, CANVAS_H - padding)              
]

def run_calibration_and_test():
    eye_points = []
    current_target = 0
    calibrating = True
    testing = False
    matrix = None

    smooth_x, smooth_y = CANVAS_W // 2, CANVAS_H // 2
    alpha = 0.35 
    
    last_input_time = 0.0
    debounce_delay = 0.5  

    print("\n🎯 GAZE CALIBRATION STARTED")

    while calibrating or testing:
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        gaze_vector_x, gaze_vector_y = 0.0, 0.0
        eyes_detected = False
        preview_frame = frame.copy()

        if results.face_blendshapes and results.facial_transformation_matrixes:
            # --- 1. EXTRACT 3D HEAD POSE (For UI Feedback) ---
            transform_matrix = results.facial_transformation_matrixes[0]
            R = transform_matrix[:3, :3] # Extract 3x3 Rotation Matrix
            head_yaw = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2)))
            head_pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))

            # --- 2. EXTRACT INTERNAL EYE ROTATION (The Magic Math) ---
            # Blendshapes give us the exact rotation of the eyeball inside the socket, completely independent of the head.
            bs = {cat.category_name: cat.score for cat in results.face_blendshapes[0]}
            
            # Eye Yaw (Negative = Looking Left, Positive = Looking Right)
            r_yaw = bs.get('eyeLookOutRight', 0) - bs.get('eyeLookInRight', 0)
            l_yaw = bs.get('eyeLookInLeft', 0) - bs.get('eyeLookOutLeft', 0)
            eye_yaw = (r_yaw + l_yaw) / 2.0
            
            # Eye Pitch (Negative = Looking Up, Positive = Looking Down)
            r_pitch = bs.get('eyeLookDownRight', 0) - bs.get('eyeLookUpRight', 0)
            l_pitch = bs.get('eyeLookDownLeft', 0) - bs.get('eyeLookUpLeft', 0)
            eye_pitch = (r_pitch + l_pitch) / 2.0
            
            # We use the isolated eye rotation as our gaze vector!
            gaze_vector_x = eye_yaw
            gaze_vector_y = eye_pitch
            eyes_detected = True

            # --- Visual Feedback for Desktop ---
            status_text = "TESTING GAZE" if testing else f"CALIBRATING ({current_target + 1}/4)"
            cv2.putText(preview_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview_frame, f"3D Head Yaw: {head_yaw:.1f} | Pitch: {head_pitch:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            cv2.putText(preview_frame, f"3D Eye Yaw : {eye_yaw:.3f} | Pitch: {eye_pitch:.3f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            cv2.putText(preview_frame, "NO EYES DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Desktop Camera Preview", cv2.flip(preview_frame, 1))

        # --- EVENT HANDLING ---
        space_pressed, r_pressed, esc_pressed = False, False, False
        current_time = time.time()
        can_input = (current_time - last_input_time) > debounce_delay

        cv_key = cv2.waitKey(1) & 0xFF
        if cv_key == 27: esc_pressed = True
        elif cv_key == 32: space_pressed = True
        elif cv_key == ord('r') or cv_key == ord('R'): r_pressed = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT: esc_pressed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: esc_pressed = True
                elif event.key == pygame.K_SPACE: space_pressed = True
                elif event.key == pygame.K_r: r_pressed = True

        if esc_pressed: return False, None
            
        if can_input:
            if calibrating and space_pressed:
                if eyes_detected:
                    last_input_time = current_time
                    tx, ty = targets_canvas[current_target]
                    print(f"✅ Target {current_target + 1}/4 locked! Eye Rotation ({gaze_vector_x:.4f}, {gaze_vector_y:.4f})")
                    eye_points.append([gaze_vector_x, gaze_vector_y])
                    current_target += 1
                    
                    if current_target >= 4:
                        pts_src = np.array(eye_points, dtype=np.float32)
                        pts_dst = np.array(targets_canvas, dtype=np.float32)
                        matrix, _ = cv2.findHomography(pts_src, pts_dst)
                        
                        calibrating = False
                        testing = True
                        print("\n✨ ENTERING TEST MODE!")
                else:
                    print("⚠️ Spacebar pressed, but no eyes detected!")

            elif testing:
                if space_pressed:
                    last_input_time = current_time
                    return True, matrix
                elif r_pressed:
                    last_input_time = current_time
                    print("🔄 Recalibrating...")
                    return run_calibration_and_test()

        # --- RENDERING ---
        screen.fill((0, 0, 0))

        if calibrating:
            tx, ty = targets_canvas[current_target]
            pygame.draw.circle(screen, (255, 0, 0), (tx, ty), 8)
            pygame.draw.circle(screen, (255, 255, 255), (tx, ty), 8, 1)
            msg = f"Target {current_target + 1}/4 - Press SPACE" if eyes_detected else "NO EYES DETECTED"
            screen.blit(font.render(msg, True, (0, 255, 0) if eyes_detected else (255, 0, 0)), (10, CANVAS_H - 20))

        elif testing and matrix is not None and eyes_detected:
            # Transform the pure eye-rotation vector into Canvas coordinates
            src_pt = np.array([[[gaze_vector_x, gaze_vector_y]]], dtype=np.float32)
            dst_pt = cv2.perspectiveTransform(src_pt, matrix)
            raw_x, raw_y = dst_pt[0][0][0], dst_pt[0][0][1]

            smooth_x = int(alpha * raw_x + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * raw_y + (1 - alpha) * smooth_y)

            cx = max(0, min(CANVAS_W, smooth_x))
            cy = max(0, min(CANVAS_H, smooth_y))

            pygame.draw.circle(screen, (0, 255, 255), (cx, cy), 12)       
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6)       
            pygame.draw.circle(screen, (0, 200, 255), (cx, cy), 18, 1)    

            screen.blit(font.render("Look Around! [SPACE]=Save  [R]=Retry", True, (0, 255, 255)), (10, CANVAS_H - 20))

        pygame.display.flip()

    return False, None

success, matrix = run_calibration_and_test()

cap.release()
cv2.destroyAllWindows()
pygame.quit()

if success and matrix is not None:
    config_data = {
        "homography_matrix": matrix.tolist(),
        "canvas_width": CANVAS_W,
        "canvas_height": CANVAS_H,
        "tracking_mode": "true_3d_blendshapes"
    }
    with open("gaze_config.json", "w") as f:
        json.dump(config_data, f, indent=4)
    print("\n🎉 GAZE CALIBRATION APPROVED AND SAVED!")
else:
    print("\n❌ Calibration cancelled.")