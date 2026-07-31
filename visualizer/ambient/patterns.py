"""
Ambient pattern renderer.

Draws a slow, flowing "plasma"-style field across the LED canvas each
frame, colored and paced by whatever ambient_state app.py hands it (a
blend of weather, time-of-day, room motion, and any voice overrides).

This deliberately does NOT reuse the particle-physics engines the other
visualizers use (waterfall/physics.py, pixel_life/physics.py) - ambient
patterns are a continuous field rather than discrete particles, so a
vectorized numpy approach keeps it smooth even at 60fps.
"""

import numpy as np
import pygame

_grid_cache = {}


def _get_grid(canvas_w, canvas_h):
    """Cache the normalized coordinate grid; canvas size never changes at runtime."""
    key = (canvas_w, canvas_h)
    if key not in _grid_cache:
        xs = np.linspace(0, 1, canvas_w, dtype=np.float32)
        ys = np.linspace(0, 1, canvas_h, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)  # shape (canvas_h, canvas_w)
        _grid_cache[key] = (gx, gy)
    return _grid_cache[key]


def render_ambient_frame(screen, canvas_w, canvas_h, t, ambient_state, cursor=None):
    """
    screen: full pygame display Surface (pattern is drawn at origin, canvas-sized)
    t: elapsed seconds (monotonic), drives the animation
    ambient_state: {
        "base_color":   (r, g, b),
        "accent_color": (r, g, b),
        "brightness":   0-1,
        "energy":       0-1,   # speed/turbulence of the flow field
    }
    cursor: optional (x, y) canvas-space point to draw a soft highlight ripple
            (used when finger-tracking is enabled instead of room-motion mode)
    """
    gx, gy = _get_grid(canvas_w, canvas_h)

    energy = max(0.05, min(1.0, ambient_state.get("energy", 0.3)))
    speed = 0.15 + energy * 0.9
    scale = 3.0 + energy * 2.0

    phase = t * speed
    field = (
        np.sin((gx * scale) + phase)
        + np.sin((gy * scale) - phase * 0.8)
        + np.sin(((gx + gy) * scale * 0.6) + phase * 1.3)
        + np.sin((np.sqrt((gx - 0.5) ** 2 + (gy - 0.5) ** 2) * scale * 2.5) - phase * 1.6)
    )
    field = (field + 4) / 8.0  # roughly normalize to 0..1
    field = np.clip(field, 0.0, 1.0)

    base = np.array(ambient_state.get("base_color", (10, 15, 40)), dtype=np.float32)
    accent = np.array(ambient_state.get("accent_color", (60, 90, 150)), dtype=np.float32)
    brightness = max(0.05, min(1.0, ambient_state.get("brightness", 0.5)))

    field3 = field[:, :, None]
    rgb = base[None, None, :] * (1 - field3) + accent[None, None, :] * field3
    rgb *= brightness

    if cursor is not None:
        cx, cy = cursor
        if cx is not None and cy is not None and cx > 0 and cy > 0:
            px, py = cx / canvas_w, cy / canvas_h
            dist = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            ripple = np.exp(-(dist * 6.0) ** 2)[:, :, None]
            rgb += accent[None, None, :] * ripple * 0.6

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # pygame.surfarray wants (width, height, 3)
    rgb_t = np.transpose(rgb, (1, 0, 2))
    frame_surface = pygame.surfarray.make_surface(rgb_t)
    screen.blit(frame_surface, (0, 0))
