import os
import time
import queue
import cv2
import requests
import tempfile
from dotenv import load_dotenv

load_dotenv()

runway_task_queue = queue.Queue()

def runway_background_worker(action_queue, canvas_w, canvas_h):
    api_key = os.getenv("RUNWAYML_API_SECRET") or os.getenv("RUNWAY_API_KEY")
    
    if not api_key:
        print("⚠️ [Runway Worker] RUNWAYML_API_SECRET missing in .env! Background thread disabled.")
        return

    print("🎬 [Runway Worker] Connected to Runway Router API.")

    while True:
        prompt = runway_task_queue.get()
        if not prompt:
            continue

        print(f"🎬 [Runway Worker] Starting video generation for prompt: '{prompt}'...")
        
        try:
            print("⏳ [Runway Worker] Submitting via raw HTTP Router. Waiting for completion...")
            
            # 1. Send request to Runway Router API
            url = 'https://api.dev.runwayml.com/v1/generate/video'
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Runway-Version": "2024-11-06"
            }
            payload = {
                "configId": "pixel-background",
                "input": {
                    "promptText": f"Seamless looping ambient texture, {prompt}, dark high-contrast background"
                }
            }
            
            create_response = requests.post(url, headers=headers, json=payload)
            if create_response.status_code != 200:
                print(f"❌ [Runway Worker] Creation API Error: {create_response.text}")
                continue
                
            task_id = create_response.json().get("id")
            print(f"⏳ [Runway Worker] Task ID {task_id} submitted! Polling status...")

            # 2. Poll task status until finished
            video_url = None
            while True:
                task_url = f"https://api.dev.runwayml.com/v1/tasks/{task_id}"
                status_res = requests.get(task_url, headers=headers).json()
                status = status_res.get("status")
                
                if status == "SUCCEEDED":
                    output_list = status_res.get("output", [])
                    if output_list:
                        video_url = output_list[0]
                        print(f"✅ [Runway Worker] Video ready! URL: {video_url}")
                    break
                elif status in ["FAILED", "CANCELLED"]:
                    print(f"❌ [Runway Worker] Task Failed: {status_res}")
                    break
                    
                time.sleep(3.0)
                
            if not video_url:
                continue

            # 3. Download MP4 File
            response = requests.get(video_url, stream=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp_file.write(chunk)
                tmp_path = tmp_file.name

            # 4. Decode, Center Crop, and Resize Video Frames for the LED Canvas
            cap = cv2.VideoCapture(tmp_path)
            led_frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                target_ratio = canvas_w / canvas_h
                frame_ratio = w / h

                # Center crop to match canvas aspect ratio perfectly
                if frame_ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    start_x = (w - new_w) // 2
                    cropped_frame = frame[:, start_x:start_x + new_w]
                else:
                    new_h = int(w / target_ratio)
                    start_y = (h - new_h) // 2
                    cropped_frame = frame[start_y:start_y + new_h, :]
                
                resized_frame = cv2.resize(cropped_frame, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                led_frames.append(rgb_frame)

            cap.release()
            os.remove(tmp_path)

            if led_frames:
                print(f"🎨 [Runway Worker] Successfully loaded {len(led_frames)} background frames!")
                action_queue.put({
                    "action": "set_background_frames",
                    "frames": led_frames
                })

        except Exception as e:
            print(f"❌ [Runway Worker] Error: {e}")