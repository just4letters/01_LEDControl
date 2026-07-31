"""
Room motion data source for the ambient visualizer.

Watches the webcam in a lightweight frame-differencing mode (no MediaPipe)
to estimate how much activity is happening in the room, which app.py uses
to nudge the pattern's energy/speed. Uses the same central camera config
(core/config.py) as core.vision.HandTracker for consistency.

IMPORTANT: this and core.vision.HandTracker both want exclusive access to
the camera. app.py is responsible for only having one of them running at
a time and calling .stop() before switching modes.

--- Data source interface contract ---
    .start()        -> opens the camera and begins a background thread
    .stop()         -> releases the camera
    .get_state()    -> thread-safe dict, always safe to call

get_state() returns:
    {
        "available": bool,
        "motion_level": float,   # smoothed 0.0 (still) - 1.0 (lots of activity)
    }
"""

import threading
import time

import cv2
import numpy as np

from core import config


class MotionSensor:
    def __init__(self, smoothing=0.15):
        self.smoothing = smoothing
        self.lock = threading.Lock()
        self.running = False
        self.cap = None
        self._motion_level = 0.0
        self._available = False

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(config.FINGER_CAMERA_INDEX)
        if not self.cap.isOpened():
            print("[motion] Could not open camera.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        time.sleep(0.05)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        with self.lock:
            self._available = False

    def get_state(self):
        with self.lock:
            return {"available": self._available, "motion_level": self._motion_level}

    def _loop(self):
        prev_gray = None
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.05)
                continue

            small = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
                changed_ratio = float(np.count_nonzero(thresh)) / thresh.size
                # Everyday small motion rarely changes more than ~15% of pixels;
                # scale so that counts as "full" energy.
                raw_level = min(1.0, changed_ratio / 0.15)

                with self.lock:
                    self._motion_level = (
                        self._motion_level * (1 - self.smoothing) + raw_level * self.smoothing
                    )
                    self._available = True

            prev_gray = gray
            time.sleep(0.05)  # ~20fps sampling is plenty for a motion estimate
