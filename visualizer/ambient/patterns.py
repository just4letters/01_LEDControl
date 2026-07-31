"""
Ambient pattern renderer.

Generative approaches live in PATTERNS - a name -> render function registry.
Trying a different "look" later (a cellular pattern, a particle-drift
field, whatever) is just adding a new function below and giving it a key;
app.py never changes, it always calls render_ambient_frame(...) and gets
whichever pattern is selected (defaults to "flow_field").

The default pattern, "flow_field", is a domain-warped fractal noise field
(no external noise library - a small hand-rolled value-noise implementation
below, since the project's venv doesn't have one) rendered through OKLab
color blending instead of raw RGB lerp, which avoids the muddy grey
dead-zone you get interpolating complementary hues directly in RGB. A
curl-like vector derived from the field's own spatial gradient drives a
secondary "shimmer" layer so the texture reads as flowing rather than
pulsing in place.
"""

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Value noise (dependency-free stand-in for Perlin/simplex noise)
# ---------------------------------------------------------------------------

def _hash3(x, y, z):
    """Cheap deterministic pseudo-random field in [0, 1), shader-style hash."""
    p = np.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453123
    return p - np.floor(p)


def _fade(t):
    """Quintic smootherstep - Perlin's improved fade curve (C2 continuous,
    avoids the visible grid-line seams a linear or smoothstep fade leaves)."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _value_noise_3d(x, y, z):
    """Trilinearly-interpolated value noise, roughly in [0, 1]."""
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = _fade(x - x0), _fade(y - y0), _fade(z - z0)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    c000 = _hash3(x0, y0, z0); c100 = _hash3(x1, y0, z0)
    c010 = _hash3(x0, y1, z0); c110 = _hash3(x1, y1, z0)
    c001 = _hash3(x0, y0, z1); c101 = _hash3(x1, y0, z1)
    c011 = _hash3(x0, y1, z1); c111 = _hash3(x1, y1, z1)

    x00 = c000 + fx * (c100 - c000)
    x10 = c010 + fx * (c110 - c010)
    x01 = c001 + fx * (c101 - c001)
    x11 = c011 + fx * (c111 - c011)

    y0_ = x00 + fy * (x10 - x00)
    y1_ = x01 + fy * (x11 - x01)

    return y0_ + fz * (y1_ - y0_)


def _fbm3(x, y, z, octaves=3, lacunarity=2.0, gain=0.5):
    """Fractal Brownian motion: layered octaves of value noise, ~[0, 1]."""
    amp, freq = 0.5, 1.0
    total = np.zeros_like(x)
    norm = 0.0
    for _ in range(octaves):
        total += amp * _value_noise_3d(x * freq, y * freq, z * freq)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


# ---------------------------------------------------------------------------
# OKLab color blending (Bjorn Ottosson's OKLab) - blending in a perceptual
# space instead of raw sRGB keeps gradients between complementary colors
# (e.g. deep blue -> warm orange) looking vivid instead of passing through
# a grey/brown dead zone in the middle.
# ---------------------------------------------------------------------------

def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)
    return np.clip(out * 255.0, 0, 255)


def _rgb_to_oklab(rgb):
    r, g, b = _srgb_to_linear(rgb[0]), _srgb_to_linear(rgb[1]), _srgb_to_linear(rgb[2])
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.array([L, A, B], dtype=np.float64)


def _oklab_to_rgb(lab):
    L, A, B = lab[0], lab[1], lab[2]
    l_ = L + 0.3963377774 * A + 0.2158037573 * B
    m_ = L - 0.1055613458 * A - 0.0638541728 * B
    s_ = L - 0.0894841775 * A - 1.2914855480 * B
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return _linear_to_srgb(r), _linear_to_srgb(g), _linear_to_srgb(b)


# ---------------------------------------------------------------------------
# Shared canvas-coordinate grid
# ---------------------------------------------------------------------------

_grid_cache = {}
NOISE_SCALE = 42.0     # pixels per "noise unit" - bigger = larger, slower-looking features
TIME_LOOP = 100000.0   # wrap elapsed time so long-running installs don't lose float precision

# Everything below runs in float32, not the numpy default float64: at the
# ~10 noise-octave evaluations/frame this pattern does, float64 was costing
# roughly 2.5-3x the frame time for no visible quality gain. The one place
# this needs care is that float32 arrays combined with a *numpy* float64
# scalar (as opposed to a plain python float) silently upcast the whole
# expression back to float64 - see the explicit np.float32(...) casts below
# wherever a scalar crosses from python-land into the noise functions.


def _get_grid(canvas_w, canvas_h):
    key = (canvas_w, canvas_h)
    if key not in _grid_cache:
        xs = np.arange(canvas_w, dtype=np.float32) / np.float32(NOISE_SCALE)
        ys = np.arange(canvas_h, dtype=np.float32) / np.float32(NOISE_SCALE)
        gx, gy = np.meshgrid(xs, ys)  # shape (canvas_h, canvas_w)
        _grid_cache[key] = (gx, gy)
    return _grid_cache[key]


# ---------------------------------------------------------------------------
# Pattern: flow_field (default)
# ---------------------------------------------------------------------------

def _render_flow_field(canvas_w, canvas_h, t, ambient_state, cursor=None):
    gx, gy = _get_grid(canvas_w, canvas_h)

    energy = max(0.05, min(1.0, ambient_state.get("energy", 0.3)))
    speed = 0.03 + energy * 0.12
    z = np.float32((t * speed) % TIME_LOOP)

    # Domain warp: distort the sample coordinates with a slower, larger
    # noise field before reading the main field. This single trick is most
    # of what separates "sine-wave plasma" from organic drifting light.
    # (Single octave here on purpose - warp is meant to be smooth/large-
    # scale, extra octaves just cost frame time without changing the look.)
    warp_strength = 1.1
    wx = _fbm3(gx + 5.2, gy + 1.3, z * 0.6, octaves=1)
    wy = _fbm3(gx + 8.3, gy + 2.8, z * 0.6, octaves=1)
    warped_x = gx + (wx - 0.5) * warp_strength
    warped_y = gy + (wy - 0.5) * warp_strength

    field = _fbm3(warped_x, warped_y, z, octaves=3)

    # Curl-like vector from the field's own spatial gradient (rotate 90deg
    # = divergence-free), used to advect a finer "shimmer" layer so it
    # looks like it's flowing along the same currents as the main field
    # rather than just being independent sparkle on top.
    grad_y, grad_x = np.gradient(field)
    curl_x, curl_y = -grad_y, grad_x
    shimmer = _fbm3(
        gx * 2.6 + curl_x * 3.0,
        gy * 2.6 + curl_y * 3.0,
        z * 1.7 + 50.0,
        octaves=2,
    )

    mix = np.clip(field * 0.8 + shimmer * 0.2, 0.0, 1.0)

    base_lab = _rgb_to_oklab(np.array(ambient_state.get("base_color", (10, 15, 40)), dtype=np.float32))
    accent_lab = _rgb_to_oklab(np.array(ambient_state.get("accent_color", (60, 90, 150)), dtype=np.float32))

    mix3 = mix[None, :, :]
    lab = base_lab[:, None, None] * (1 - mix3) + accent_lab[:, None, None] * mix3
    r, g, b = _oklab_to_rgb(lab)

    brightness = max(0.05, min(1.0, ambient_state.get("brightness", 0.5)))
    r, g, b = r * brightness, g * brightness, b * brightness

    if cursor is not None:
        cx, cy = cursor
        if cx is not None and cy is not None and cx > 0 and cy > 0:
            px, py = cx / NOISE_SCALE, cy / NOISE_SCALE
            dist = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            ripple = np.exp(-(dist * 2.4) ** 2) * 0.6
            accent = np.array(ambient_state.get("accent_color", (60, 90, 150)), dtype=np.float32)
            r = r + accent[0] * ripple
            g = g + accent[1] * ripple
            b = b + accent[2] * ripple

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


PATTERNS = {
    "flow_field": _render_flow_field,
}


def render_ambient_frame(screen, canvas_w, canvas_h, t, ambient_state, cursor=None, pattern="flow_field"):
    """
    screen: full pygame display Surface (pattern is drawn at origin, canvas-sized)
    t: elapsed seconds (monotonic), drives the animation
    ambient_state: {
        "base_color":   (r, g, b),
        "accent_color": (r, g, b),
        "brightness":   0-1,
        "energy":       0-1,   # speed/turbulence of the flow field
    }
    cursor: optional (x, y) canvas-space point for a soft highlight ripple
            (used when finger-tracking is enabled instead of room-motion mode)
    pattern: key into PATTERNS - this is the hook for trying other ambient
             approaches later without changing app.py's call site.
    """
    render_fn = PATTERNS.get(pattern, _render_flow_field)
    rgb = render_fn(canvas_w, canvas_h, t, ambient_state, cursor=cursor)

    # pygame.surfarray wants (width, height, 3)
    rgb_t = np.transpose(rgb, (1, 0, 2))
    frame_surface = pygame.surfarray.make_surface(rgb_t)
    screen.blit(frame_surface, (0, 0))
