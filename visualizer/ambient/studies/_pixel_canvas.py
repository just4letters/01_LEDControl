"""
Shared harness for ambient "studies" - small standalone pygame sketches that
prototype one visual idea in isolation before it's folded into the real-time
patterns.py used by the installation.

No camera, no voice, no AI here on purpose: a study is meant to be run,
watched, and tweaked in a tight loop (`python study_XX_name.py`), not wired
into the full app.

DISPLAY SETUP - this mirrors exactly what every other app.py in this repo
does (main.py, ambient/app.py, pixel_life/app.py, screen_calibration.py):
pygame opens the second monitor (display=1) fullscreen at its native
1920x1080 signal. The actual animation is a small Surface sized EXACTLY to
the LED grid (`width` x `height` from led_config.json) blitted onto that
fullscreen surface at (`offset_x`, `offset_y`) - the same rectangle
screen_calibration.py measured out and every other visualizer already
blits into. Nothing here is ever scaled or stretched: the animation's own
resolution IS the LED grid's resolution, one buffer pixel is one physical
LED, placed exactly where led_config.json says it goes.

Controls (all studies): Esc or close the window to quit.
"""

import os
import json

import numpy as np
import pygame

STUDIES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(STUDIES_DIR, "..", "..", ".."))
LED_CONFIG_PATH = os.path.join(ROOT_DIR, "led_config.json")


class PixelCanvas:
    def __init__(self, title="study"):
        with open(LED_CONFIG_PATH, "r") as f:
            cfg = json.load(f)

        self.width = int(cfg.get("width", 256))
        self.height = int(cfg.get("height", 256))
        self.offset_x = int(cfg.get("offset_x", 0))
        self.offset_y = int(cfg.get("offset_y", 0))

        # Exactly one entry per physical LED - no padding, no scaling.
        self.buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        pygame.init()
        pygame.display.set_caption(title)

        OS_W, OS_H = 1920, 1080
        try:
            self.screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
        except Exception:
            self.screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)

        # The animation's own surface - exactly the LED grid's resolution.
        self._frame_surf = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

    # -- drawing into the logical buffer -------------------------------

    def clear(self, color=(0, 0, 0)):
        self.buffer[:] = color

    def set_px(self, x, y, color):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < self.width and 0 <= yi < self.height:
            self.buffer[yi, xi] = color

    def add_px(self, x, y, color, mix=1.0):
        """Blend `color` into the existing pixel (mix=1.0 replaces it outright)."""
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < self.width and 0 <= yi < self.height:
            cur = self.buffer[yi, xi].astype(np.float32)
            new = np.array(color, dtype=np.float32)
            self.buffer[yi, xi] = np.clip(cur + (new - cur) * mix, 0, 255).astype(np.uint8)

    def hline(self, y, color, x0=0, x1=None):
        x1 = self.width if x1 is None else x1
        yi = int(round(y))
        if 0 <= yi < self.height:
            self.buffer[yi, x0:x1] = color

    # -- present ----------------------------------------------------------

    def present(self, fps=60):
        pygame.surfarray.blit_array(self._frame_surf, self.buffer.swapaxes(0, 1))
        self.screen.fill((0, 0, 0))
        self.screen.blit(self._frame_surf, (self.offset_x, self.offset_y))
        pygame.display.flip()
        self.clock.tick(fps)

    def poll(self):
        """Returns False when the study should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def quit(self):
        pygame.quit()
