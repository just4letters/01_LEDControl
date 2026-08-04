"""
Study 02 - Selective Membrane
=============================

A variation on Study 01 (Barrier Drift). Same rigid line, same buoyant,
turbulent gas pressing against it from below - but now the line is a
membrane that only lets a pixel through some of the time it tries.
Pressure still has to build up before a crossing attempt happens at all,
same as Study 01, but each attempt is its own coin flip - there's no
"chosen one" pixel marked permeable at birth. That's deliberate: a fixed
subset of eternally-permeable pixels can, by bad luck, all use up their
one chance and leave nothing behind able to leak - a dead end with no way
back. A fresh roll on every attempt can't get stuck that way.

The ones that do pass through cross a threshold that gives them an
identity - they come out dim red, dim green, or dim blue (chosen at
random the instant they leak), where before they were undifferentiated
gray-blue gas. Every crossing is an event the whole line reacts to: the
instant a pixel gets through, the entire barrier flashes to that pixel's
color, then fades back to neutral gray.

Above the line, pixels drift the same as before, and feel a weak pull
toward whatever's already stuck in their own color - red drawn to red,
green to green, blue to blue. The side walls bounce, like the gas does
below the barrier - and so does anything not its own color: a
different-colored cluster is a solid obstacle, not something to pass
through. The only ways to actually come to rest are the very top edge or
touching a stuck pixel of its own color. Combined with the attraction,
that means a pixel can get pulled straight toward its own kind and then
deflected off someone else's color guarding the way - which can produce
some genuinely unpredictable paths.

PERFORMANCE NOTE: the gas below the barrier is a large, constant-size
population (GAS_TARGET), updated and redrawn every single frame - at any
reasonable GAS_TARGET this dwarfs every other cost in the simulation, so
its physics and rendering are done with numpy (batch math over the whole
population at once) instead of a plain Python per-particle loop. Flutter
and the leak check stay plain Python since profiling showed they're not
the bottleneck (flutter/stuck-per-color counts are small). This is meant
to be behaviorally identical to a naive per-particle version, just much
cheaper to run - if something here ever looks different than expected,
that's a bug in the vectorized math, not an intentional change.

Run:  python study_02_selective_membrane.py
Opens fullscreen on the LED monitor (display 1) at native resolution,
positioned per led_config.json. Esc or close the window to quit.
"""

import os
import sys
import math
import random

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from _pixel_canvas import PixelCanvas

# --- tunables ----------------------------------------------------------
GAS_TARGET = 2200        # gas pixels kept alive below the barrier - a hard
                         # cap, not just a floor (see rejoin logic below) -
                         # so fresh particles keep getting spawned forever
BUOYANCY = 0.012        # constant upward drift for gas particles
JITTER = 0.10           # per-frame random velocity kick (turbulence)
TURB_FREQ = 0.6         # wind-sway frequency
TURB_AMP = 0.05
DAMPING = 0.985         # velocity damping so speeds settle rather than run away

PRESSURE_GAIN = 1.0     # pressure added per barrier contact
PRESSURE_DECAY = 0.965  # per-frame decay of each column's pressure
LEAK_K = 0.0000008      # leak probability scaling per unit pressure^2 (4x study 01's,
                         # since only 25% of successful attempts actually get through here)
LEAK_RESET = 0.35       # fraction of pressure kept after a leak (partial relief)

PERMEABLE_FRACTION = 0.5   # chance any given crossing ATTEMPT actually succeeds
FAIL_RELIEF = 0.85          # pressure kept after a failed attempt (softer than LEAK_RESET)

NEAR_BARRIER_BAND = 6    # px - counts as "pressed against the line" for give-up timing
PRESS_GIVEUP_MIN = 120   # frames (~2s) minimum dwell at the line before giving up
PRESS_GIVEUP_MAX = 300   # frames (~5s) maximum - randomized per particle so they don't sink in sync

FLUTTER_WOBBLE = 0.14
FLUTTER_RISE = 0.02
FLUTTER_DAMP = 0.992

STICK_RADIUS = 1   # a flutter pixel sticks/collides once a neighbor within this many px is stuck
BOUNCE_DAMP = -0.5  # velocity multiplier on bouncing off a different-colored pixel (same as walls)

ATTRACTION_STRENGTH = 0.015   # steady per-frame pull toward the nearest same-color stuck pixel
ATTRACTION_MAX_RANGE = 120    # px - ignore same-color clusters farther than this (no pull at all)
ATTRACTION_REFRESH_FRAMES = 20  # re-pick the nearest same-color target this often, not every
                                 # frame - the pull is weak/slow, so which exact pixel is
                                 # "nearest" barely changes within a third of a second, and this
                                 # is the one part of the sim whose cost scales with total stuck
                                 # pixels rather than the (small, bounded) flutter count

BARRIER_GLOW_DECAY = 0.95   # per-frame decay of the whole-barrier color flash after a crossing

GAS_COLOR = (35, 45, 60)          # neutral gray-blue - undifferentiated, hasn't passed through yet
GAS_COLOR_HI = (60, 75, 95)       # brighter flicker variant
BARRIER_COLOR = (150, 150, 150)
BARRIER_FLASH = (255, 255, 255)

LEAK_COLORS = [
    (50, 0, 0),   # dim red
    (0, 20, 0),   # dim green
    (0, 0, 60),   # dim blue
]


def new_gas_particle(w, h, barrier_y):
    return {
        "x": random.uniform(0, w - 1),
        "y": random.uniform(barrier_y + 4, h - 1),
        "vx": random.uniform(-0.05, 0.05),
        "vy": random.uniform(-0.05, 0.0),
        "phase": random.uniform(0, math.tau),
        "press_frames": 0,
        "give_up_after": random.randint(PRESS_GIVEUP_MIN, PRESS_GIVEUP_MAX),
    }


def check_contact(x, y, color, stuck_positions_by_color):
    """What's touching (x, y): 'own' if a same-color stuck pixel is within
    STICK_RADIUS, 'other' if a different-colored one is (and no same-color
    one), else None. Own-color takes priority - if a pixel is squeezed
    between its own kind and a different color at once, it sticks rather
    than bounces."""
    xi, yi = int(round(x)), int(round(y))
    own_positions = stuck_positions_by_color[color]
    other_positions = [
        positions for c, positions in stuck_positions_by_color.items() if c != color
    ]
    touched_other = False
    for dx in range(-STICK_RADIUS, STICK_RADIUS + 1):
        for dy in range(-STICK_RADIUS, STICK_RADIUS + 1):
            cell = (xi + dx, yi + dy)
            if cell in own_positions:
                return "own"
            if not touched_other and any(cell in positions for positions in other_positions):
                touched_other = True
    return "other" if touched_other else None


def nearest_same_color(x, y, color, stuck_positions_by_color):
    """Closest already-stuck pixel of `color`, or None if there isn't one
    (or the nearest one is farther than ATTRACTION_MAX_RANGE)."""
    positions = stuck_positions_by_color[color]
    if not positions:
        return None
    best = None
    best_d2 = ATTRACTION_MAX_RANGE * ATTRACTION_MAX_RANGE
    for (px, py) in positions:
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (px, py)
    return best


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def update_gas_vectorized(gas, t, W, H, barrier_y, pressure):
    """One frame of gas physics for the whole population at once (numpy).
    Mutates `gas` (writes updated x/y/vx/vy/press_frames/give_up_after back
    into each particle dict) and `pressure` (numpy array) in place.

    This does exactly what the old per-particle Python loop did, in the
    same order (turbulence/jitter -> damping -> move -> wall bounce ->
    floor bounce -> barrier collision -> give-up timer), just batched.
    """
    n = len(gas)
    if n == 0:
        return

    xs = np.fromiter((p["x"] for p in gas), dtype=np.float64, count=n)
    ys = np.fromiter((p["y"] for p in gas), dtype=np.float64, count=n)
    vxs = np.fromiter((p["vx"] for p in gas), dtype=np.float64, count=n)
    vys = np.fromiter((p["vy"] for p in gas), dtype=np.float64, count=n)
    phases = np.fromiter((p["phase"] for p in gas), dtype=np.float64, count=n)
    press_f = np.fromiter((p["press_frames"] for p in gas), dtype=np.float64, count=n)
    give_ups = np.fromiter((p["give_up_after"] for p in gas), dtype=np.float64, count=n)

    # turbulence + jitter + buoyancy
    vxs += np.sin(t * TURB_FREQ + phases) * TURB_AMP * 0.02
    vxs += np.random.uniform(-JITTER, JITTER, n) * 0.02
    vys += np.random.uniform(-JITTER, JITTER, n) * 0.015
    vys -= BUOYANCY

    vxs *= DAMPING
    vys *= DAMPING

    xs += vxs
    ys += vys

    # side walls: bounce
    left = xs < 0
    xs[left] = 0.0
    vxs[left] *= -0.5
    right = xs > (W - 1)
    xs[right] = float(W - 1)
    vxs[right] *= -0.5

    # floor: bounce
    floor = ys > (H - 1)
    ys[floor] = float(H - 1)
    vys[floor] *= -0.4

    # pressing against the barrier from below
    hit = (ys <= barrier_y + 1) & (vys < 0)
    ys[hit] = float(barrier_y + 1)
    vys[hit] *= -0.25
    if np.any(hit):
        hit_cols = np.clip(xs[hit].astype(np.int64), 0, W - 1)
        np.add.at(pressure, hit_cols, PRESSURE_GAIN)
        np.clip(pressure, None, 40.0, out=pressure)

    # been hovering at the line too long without finding a way through -
    # give up and sink back into the mix
    near = ys <= (barrier_y + NEAR_BARRIER_BAND)
    press_f = np.where(near, press_f + 1, 0.0)
    giveup = near & (press_f > give_ups)
    n_giveup = int(np.count_nonzero(giveup))
    if n_giveup:
        ys[giveup] = np.random.uniform(barrier_y + 20, H - 1, n_giveup)
        vys[giveup] = np.random.uniform(-0.05, 0.05, n_giveup)
        vxs[giveup] = np.random.uniform(-0.1, 0.1, n_giveup)
        press_f[giveup] = 0.0
        give_ups[giveup] = np.random.randint(PRESS_GIVEUP_MIN, PRESS_GIVEUP_MAX + 1, n_giveup)

    for i, p in enumerate(gas):
        p["x"] = xs[i]
        p["y"] = ys[i]
        p["vx"] = vxs[i]
        p["vy"] = vys[i]
        p["press_frames"] = press_f[i]
        p["give_up_after"] = give_ups[i]


def draw_gas_vectorized(canvas, gas, t):
    """Batch version of: for p in gas: compute flicker color, set_px.

    Deliberately re-reads x/y/phase from `gas` rather than reusing the
    arrays update_gas_vectorized built earlier this frame: by render time,
    the leak check has already removed any particle that just crossed the
    barrier (and give-up/rejoin may have added others) - re-extracting
    keeps this in sync with whatever's actually still gas right now.
    """
    n = len(gas)
    if n == 0:
        return
    xs = np.fromiter((p["x"] for p in gas), dtype=np.float64, count=n)
    ys = np.fromiter((p["y"] for p in gas), dtype=np.float64, count=n)
    phases = np.fromiter((p["phase"] for p in gas), dtype=np.float64, count=n)

    flick = (0.5 + 0.5 * np.sin(t * 2.0 + phases * 3.0)) * 0.4
    colors = np.empty((n, 3), dtype=np.float64)
    for i in range(3):
        colors[:, i] = GAS_COLOR[i] + (GAS_COLOR_HI[i] - GAS_COLOR[i]) * flick
    colors = colors.astype(np.uint8)

    xi = np.round(xs).astype(np.int64)
    yi = np.round(ys).astype(np.int64)
    valid = (xi >= 0) & (xi < canvas.width) & (yi >= 0) & (yi < canvas.height)
    canvas.buffer[yi[valid], xi[valid]] = colors[valid]


def draw_stuck_vectorized(canvas, stuck, t, cache):
    """Batch version of: for s in stuck: compute twinkle color, set_px.

    `stuck` only ever grows, and each entry's x/y/color/twinkle-phase are
    set once and never change - only the sin() driving the twinkle moves.
    So the x/y/color/phase arrays are cached and only rebuilt on frames
    where a new pixel actually got added (a handful of times a second),
    instead of re-reading the whole (ever-growing) list from Python every
    single frame. The animation itself is unaffected either way.
    """
    n = len(stuck)
    if n == 0:
        return
    if cache["n"] != n:
        cache["xs"] = np.fromiter((s["x"] for s in stuck), dtype=np.int64, count=n)
        cache["ys"] = np.fromiter((s["y"] for s in stuck), dtype=np.int64, count=n)
        cache["twinkles"] = np.fromiter((s["twinkle"] for s in stuck), dtype=np.float64, count=n)
        cache["base_colors"] = np.array([s["color"] for s in stuck], dtype=np.float64)
        cache["n"] = n

    xs, ys = cache["xs"], cache["ys"]
    tw = 0.75 + 0.25 * np.sin(t * 1.5 + cache["twinkles"])
    colors = (cache["base_colors"] * tw[:, None]).astype(np.uint8)

    valid = (xs >= 0) & (xs < canvas.width) & (ys >= 0) & (ys < canvas.height)
    canvas.buffer[ys[valid], xs[valid]] = colors[valid]


def main():
    canvas = PixelCanvas(title="Study 02 - Selective Membrane")
    W, H = canvas.width, canvas.height
    barrier_y = H // 2

    gas = [new_gas_particle(W, H, barrier_y) for _ in range(GAS_TARGET)]
    flutter = []
    stuck = []      # [{x, y, color, twinkle}]
    stuck_positions_by_color = {c: set() for c in LEAK_COLORS}  # mirrors `stuck`, split by color
    pressure = np.zeros(W, dtype=np.float64)
    stuck_render_cache = {"n": -1}

    barrier_glow_color = BARRIER_COLOR
    barrier_glow = 0.0   # 0 = normal gray, 1 = fully the color of whatever just crossed

    t = 0.0
    running = True
    while running:
        running = canvas.poll()
        t += 1 / 60

        # keep the gas population topped up to EXACTLY the target - this is
        # what keeps fresh (properly ~50% permeable) particles entering the
        # system forever. If particles could pile up past the target (e.g.
        # every flutter pixel that drifts back down unconditionally
        # rejoining gas), the top-up would stop firing and the population
        # would be sealed forever with whatever mix it happened to have at
        # that moment. Capped population -> leaks always free a slot ->
        # top-up always re-rolls a fresh one -> the leak rate stays alive
        # indefinitely instead of dying out.
        while len(gas) < GAS_TARGET:
            gas.append(new_gas_particle(W, H, barrier_y))

        pressure *= PRESSURE_DECAY
        barrier_glow *= BARRIER_GLOW_DECAY

        # ---- gas physics: buoyant, turbulent, bounded (vectorized) ----
        update_gas_vectorized(gas, t, W, H, barrier_y, pressure)

        # ---- leak check: pressure earns an attempt, then a coin flip ----
        for x in range(W):
            if pressure[x] <= 15.0:
                continue
            prob = 1 - math.exp(-LEAK_K * pressure[x] * pressure[x])
            if random.random() < prob:
                candidates = [p for p in gas if abs(p["x"] - x) < 2 and p["y"] <= barrier_y + 3]
                if candidates:
                    if random.random() < PERMEABLE_FRACTION:
                        chosen = random.choice(candidates)
                        gas = [g for g in gas if g is not chosen]
                        chosen["y"] = barrier_y - 2
                        chosen["vy"] = -random.uniform(0.15, 0.35)
                        chosen["vx"] *= 2.0
                        chosen["color"] = random.choice(LEAK_COLORS)
                        chosen["attract_target"] = None
                        chosen["attract_refresh_in"] = 0  # pick a target on its very first flutter frame
                        flutter.append(chosen)
                        pressure[x] *= LEAK_RESET
                        # the whole line reacts to what just got through
                        barrier_glow_color = chosen["color"]
                        barrier_glow = 1.0
                    else:
                        # pressed hard enough to try, but this attempt was
                        # one of the failures that doesn't make it through
                        pressure[x] *= FAIL_RELIEF

        # ---- flutter physics: wobble, decaying rise, then stick on contact ----
        still_flying = []
        for p in flutter:
            p["vx"] += math.sin(t * 1.3 + p["phase"]) * FLUTTER_WOBBLE * 0.03
            p["vx"] += random.uniform(-0.05, 0.05)
            p["vy"] -= FLUTTER_RISE * 0.02
            p["vy"] += random.uniform(-0.03, 0.03)

            # weak, steady pull toward the nearest already-stuck pixel of
            # its own color - the rest of the motion (wobble, rise, jitter)
            # is unchanged, this just biases where it drifts. Which pixel
            # counts as "nearest" is only re-picked every
            # ATTRACTION_REFRESH_FRAMES frames (see tunables); the pull
            # itself is still applied fresh every frame using the
            # particle's current position, it just aims at a target that's
            # a little stale, which is imperceptible for a force this weak.
            p["attract_refresh_in"] -= 1
            if p["attract_refresh_in"] <= 0:
                p["attract_target"] = nearest_same_color(p["x"], p["y"], p["color"], stuck_positions_by_color)
                p["attract_refresh_in"] = ATTRACTION_REFRESH_FRAMES
            target = p["attract_target"]
            if target is not None:
                dx, dy = target[0] - p["x"], target[1] - p["y"]
                dist = math.hypot(dx, dy) or 1.0
                p["vx"] += (dx / dist) * ATTRACTION_STRENGTH
                p["vy"] += (dy / dist) * ATTRACTION_STRENGTH

            p["vx"] *= FLUTTER_DAMP
            p["vy"] *= FLUTTER_DAMP

            prev_x, prev_y = p["x"], p["y"]
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # side walls: bounce, same as the gas below - never a place to stick
            if p["x"] < 0:
                p["x"] = 0
                p["vx"] *= -0.5
            elif p["x"] > W - 1:
                p["x"] = W - 1
                p["vx"] *= -0.5

            stick_here = False
            if p["y"] <= 0:
                p["y"] = 0
                stick_here = True
            elif p["y"] >= barrier_y:
                # drifted back down onto the line - rejoin the gas below it
                # (only if there's room - see the population note above),
                # losing its color identity in the process
                p["y"] = barrier_y + 1
                p["vy"] = 0
                p["press_frames"] = 0
                p.pop("color", None)
                if len(gas) < GAS_TARGET:
                    gas.append(p)
                # else: dissipates - the slot's already been backfilled
                continue
            else:
                contact = check_contact(p["x"], p["y"], p["color"], stuck_positions_by_color)
                if contact == "own":
                    stick_here = True
                elif contact == "other":
                    # a different color is a solid obstacle, not a surface -
                    # undo the move and bounce off it instead of tunneling
                    # through or sticking
                    p["x"], p["y"] = prev_x, prev_y
                    p["vx"] *= BOUNCE_DAMP
                    p["vy"] *= BOUNCE_DAMP

            if stick_here:
                xi, yi = int(round(p["x"])), int(round(p["y"]))
                stuck.append({"x": xi, "y": yi, "color": p["color"], "twinkle": random.uniform(0, math.tau)})
                stuck_positions_by_color[p["color"]].add((xi, yi))
            else:
                still_flying.append(p)
        flutter = still_flying

        # ---- draw ----
        canvas.clear((0, 0, 0))

        for x in range(W):
            if pressure[x] > 0.5:
                flash = min(1.0, pressure[x] / 20.0)
                color = lerp_color(BARRIER_COLOR, BARRIER_FLASH, flash)
            else:
                color = BARRIER_COLOR
            if barrier_glow > 0.01:
                color = lerp_color(color, barrier_glow_color, barrier_glow)
            canvas.set_px(x, barrier_y, color)

        draw_gas_vectorized(canvas, gas, t)

        for p in flutter:
            canvas.set_px(p["x"], p["y"], p["color"])

        draw_stuck_vectorized(canvas, stuck, t, stuck_render_cache)

        canvas.present(fps=60)

    canvas.quit()


if __name__ == "__main__":
    main()
