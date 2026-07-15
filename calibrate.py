import config
import cv2
import numpy as np
import pygame
import sys
import json

# --- 1. INITIALIZE RESOURCES ---
pygame.init()
pygame.key.set_repeat(250, 50) 

OS_W, OS_H = 1920, 1080
try:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
except Exception as e:
    screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# --- 2. GENERATE THE SMART GRID ---
GRID_MARKER_SIZE = 100 
GRID_QUIET_ZONE = 10
GRID_INNER = GRID_MARKER_SIZE - (GRID_QUIET_ZONE * 2)

bg_surface = pygame.Surface((OS_W, OS_H))
# DIM THE BACKGROUND TO PREVENT BLOOM
bg_surface.fill((100, 100, 100))

marker_map = {} 
current_id = 10 

for y in range(0, OS_H, GRID_MARKER_SIZE):
    for x in range(0, OS_W, GRID_MARKER_SIZE):
        if current_id < 1000:
            img = cv2.aruco.generateImageMarker(aruco_dict, current_id, GRID_INNER)
            # DIM THE WHITE PIXELS IN THE MARKER
            img[img == 255] = 100
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            bg_surface.blit(surf, (x + GRID_QUIET_ZONE, y + GRID_QUIET_ZONE))
            marker_map[current_id] = {"x": x, "y": y}
            current_id += 1
            
# --- 3. HELPER: GENERATE CORNER MARKERS ---
TUNE_QUIET_ZONE = 10
def get_corner_markers(size):
    inner = size - (TUNE_QUIET_ZONE * 2)
    markers = []
    for i in range(4):
        img = cv2.aruco.generateImageMarker(aruco_dict, i, inner)
        # DIM THE WHITE PIXELS IN THE MARKER
        img[img == 255] = 100
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        bg = pygame.Surface((size, size))
        # DIM THE QUIET ZONE
        bg.fill((100, 100, 100))
        bg.blit(surf, (TUNE_QUIET_ZONE, TUNE_QUIET_ZONE))
        markers.append(bg)
    return markers

# Look for this in calibrate.py:
# --- INITIALIZE CAMERA FROM CONFIG ---
cap = cv2.VideoCapture(config.CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)

# --- 4. STATE VARIABLES ---
state = "GRID"
roi_x, roi_y, roi_w, roi_h = 0, 0, 256, 256 

print("\n--- PHASE 1: AUTO-DETECT ROUGH BOUNDS ---")
print("Point the webcam. Press [SPACE] to lock rough bounds.")
print("Press [Q] or [ESC] to quit at any time.")

# --- 5. THE MAIN LOOP ---
running = True
while running:
    # Handle Pygame Events (Your keyboard inputs)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False
                
            if state == "GRID" and event.key == pygame.K_SPACE:
                state = "TUNE"
                # Removed the window destroy command so you can keep seeing the camera!
                print("\n\n--- PHASE 2: PIXEL-PERFECT TUNE ---")
                print("Hold [SHIFT] to move 10 pixels at a time. Release for 1 pixel.")
                print("[W] / [S]         -> Move TOP edge")
                print("[UP] / [DOWN]     -> Move BOTTOM edge")
                print("[A] / [D]         -> Move LEFT edge")
                print("[LEFT] / [RIGHT]  -> Move RIGHT edge")
                print("\nPress [ENTER] to save matrix and exit.")

            elif state == "TUNE":
                mods = pygame.key.get_mods()
                step = 10 if mods & pygame.KMOD_SHIFT else 1
                
                if event.key == pygame.K_w: roi_y -= step; roi_h += step
                elif event.key == pygame.K_s: roi_y += step; roi_h -= step
                elif event.key == pygame.K_UP: roi_h -= step
                elif event.key == pygame.K_DOWN: roi_h += step
                elif event.key == pygame.K_a: roi_x -= step; roi_w += step
                elif event.key == pygame.K_d: roi_x += step; roi_w -= step
                elif event.key == pygame.K_LEFT: roi_w -= step
                elif event.key == pygame.K_RIGHT: roi_w += step
                
                elif event.key == pygame.K_RETURN:
                    success, frame = cap.read()
                    if success:
                        blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
                        corners, ids, rejected = detector.detectMarkers(blurred_frame)
                        
                        if ids is not None and all(i in ids for i in [0, 1, 2, 3]):
                            print("\nCalculating Perspective Matrix...")
                            cam_pts = []
                            for i in range(4):
                                idx = np.where(ids == i)[0][0]
                                center_x = np.mean(corners[idx][0][:, 0])
                                center_y = np.mean(corners[idx][0][:, 1])
                                cam_pts.append([center_x, center_y])
                            
                            tune_marker_size = max(50, min(80, roi_w // 3, roi_h // 3))
                            half = tune_marker_size / 2
                            
                            pygame_pts = [
                                [roi_x + half, roi_y + half],
                                [roi_x + roi_w - half, roi_y + half],
                                [roi_x + roi_w - half, roi_y + roi_h - half],
                                [roi_x + half, roi_y + roi_h - half]
                            ]
                            
                            matrix, _ = cv2.findHomography(np.float32(cam_pts), np.float32(pygame_pts))
                            
                            config = {
                                "roi_x": roi_x, "roi_y": roi_y, 
                                "roi_w": roi_w, "roi_h": roi_h,
                                "homography_matrix": matrix.tolist()
                            }
                            with open("led_config.json", "w") as f:
                                json.dump(config, f, indent=4)
                                
                            print(f"\nSUCCESS! Bounds ({roi_w}x{roi_h}) and Homography Matrix saved.")
                            print("Closing calibration tool...")
                            
                            cap.release()
                            cv2.destroyAllWindows()
                            pygame.quit()
                            sys.exit(0)
                        else:
                            # It fails gracefully here, letting the loop continue so the camera stays live
                            print("\nError: Camera cannot see all 4 corner markers clearly. Adjust the camera and try again.")

                roi_w, roi_h = max(100, roi_w), max(100, roi_h)

    # --- CONSTANT WEBCAM FEED ---
    success, frame = cap.read()

    if success:
        # 1. Update the LED Panel Graphics
        if state == "GRID":
            screen.fill((0, 0, 0)) 
            screen.blit(bg_surface, (0, 0))
            
            blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
            corners, ids, rejected = detector.detectMarkers(blurred_frame)
            
            valid_ids = []
            if ids is not None:
                for marker_id in ids.flatten():
                    if marker_id in marker_map:
                        valid_ids.append(marker_id)
            
            if len(valid_ids) > 0:
                min_x = min([marker_map[i]["x"] for i in valid_ids])
                min_y = min([marker_map[i]["y"] for i in valid_ids])
                max_x = max([marker_map[i]["x"] + GRID_MARKER_SIZE for i in valid_ids])
                max_y = max([marker_map[i]["y"] + GRID_MARKER_SIZE for i in valid_ids])
                
                roi_x, roi_y = min_x, min_y
                roi_w, roi_h = max_x - min_x, max_y - min_y
            
            pygame.draw.rect(screen, (0, 255, 0), (roi_x, roi_y, roi_w, roi_h), 4)

        elif state == "TUNE":
            screen.fill((0, 0, 0))
            
            tune_marker_size = max(50, min(80, roi_w // 3, roi_h // 3))
            tune_markers = get_corner_markers(tune_marker_size)
            
            pygame.draw.rect(screen, (0, 255, 0), (roi_x, roi_y, roi_w, roi_h), 2)
            
            screen.blit(tune_markers[0], (roi_x, roi_y))
            screen.blit(tune_markers[1], (roi_x + roi_w - tune_marker_size, roi_y))
            screen.blit(tune_markers[2], (roi_x + roi_w - tune_marker_size, roi_y + roi_h - tune_marker_size))
            screen.blit(tune_markers[3], (roi_x, roi_y + roi_h - tune_marker_size))
            
            sys.stdout.write(f"\rFine-Tuning: X:{roi_x} Y:{roi_y} | Resolution: {roi_w}x{roi_h}  ")
            sys.stdout.flush()

        pygame.display.flip()

        # 2. Update the Live Computer Webcam Window
        display_frame = cv2.flip(frame, 1) # Mirrored for human viewing
        
        # ONLY do the heavy math to draw the preview boxes if we are in Phase 1!
        if state == "GRID":
            display_corners, display_ids, _ = detector.detectMarkers(cv2.GaussianBlur(display_frame, (7, 7), 0))
            if display_ids is not None:
                cv2.aruco.drawDetectedMarkers(display_frame, display_corners, display_ids)
            
        cv2.imshow("Webcam View", display_frame)
        
       # 3. Universal Key Listener (Works flawlessly even if Webcam window is clicked)
        key = cv2.waitKeyEx(1)
        if key != -1:
            if key == 27 or key == ord('q'): 
                running = False
            elif key == 32 and state == "GRID": 
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            elif key == 13 and state == "TUNE": 
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
            
            # If we are in TUNE mode, adjust the bounding box directly!
            elif state == "TUNE":
                # Capital letters (holding Shift) trigger the 10-pixel jump
                step = 10 if key in [ord('W'), ord('S'), ord('A'), ord('D')] else 1
                
                # WASD controls TOP and LEFT edges
                if key in [ord('w'), ord('W')]: roi_y -= step; roi_h += step 
                elif key in [ord('s'), ord('S')]: roi_y += step; roi_h -= step 
                elif key in [ord('a'), ord('A')]: roi_x -= step; roi_w += step 
                elif key in [ord('d'), ord('D')]: roi_x += step; roi_w -= step 
                
                # Mac Arrow Codes control BOTTOM and RIGHT edges
                # Up=63232, Down=63233, Left=63234, Right=63235
                elif key == 63232: roi_h -= step 
                elif key == 63233: roi_h += step 
                elif key == 63234: roi_w -= step 
                elif key == 63235: roi_w += step 

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()