"""
Study 01 - Barrier Drift
========================

Geometric constraint, fluid behavior. A single 1px horizontal line sits
halfway up the screen - the most rigid, deliberate shape there is. Below it,
loose pixels drift upward and sway on a slow current, occasionally bumping
into the line from underneath. Each bump adds a little "pressure" at that
spot on the line; enough pressure and a pixel finds its way through. Once
through, it stops behaving like gas and starts behaving like something
falling in reverse - it flutters, wobbling side to side, slowly rising -
until it touches the top edge or a side edge, where it sticks for good.

Over a long run the line keeps getting tested, a slow trickle keeps leaking
through, and the edges above it slowly accumulate a fringe of stuck pixels.

Run:  python study_01_barrier_drift.py
Opens fullscreen on the LED monitor (display 1) and draws at exactly the
panel's native resolution, positioned per led_config.json - same as every
other visualizer in this repo. Esc or close the window to quit.
"""

import os
import sys
import math
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from _pixel_canvas import PixelCanvas

# --- tunables ----------------------------------------------------------
GAS_TARGET = 220        # gas pixels kept alive below the barrier
BUOYANCY = 0.012        # constant upward drift for gas particles
JITTER = 0.10           # per-frame random velocity kick (turbulence)
TURB_FREQ = 0.6         # wind-sway frequency
TURB_AMP = 0.05
DAMPING = 0.985         # velocity damping so speeds settle rather than run away

PRESSURE_GAIN = 1.0     # pressure added per barrier contact
PRESSURE_DECAY = 0.965  # per-frame decay of each column's pressure
LEAK_K = 0.0000002      # leak probability scaling per unit pressure^2
LEAK_RESET = 0.35       # fraction of pressure kept after a leak (partial relief)

NEAR_BARRIER_BAND = 6    # px - counts as "pressed against the line" for give-up timing
PRESS_GIVEUP_MIN = 120   # frames (~2s) minimum dwell at the line before giving up
PRESS_GIVEUP_MAX = 300   # frames (~5s) maximum - randomized per particle so they don't sink in sync

FLUTTER_WOBBLE = 0.14
FLUTTER_RISE = 0.02
FLUTTER_DAMP = 0.992

GAS_COLOR = (40, 90, 130)
GAS_COLOR_HI = (90, 170, 210)     # brighter flicker variant
BARRIER_COLOR = (150, 150, 150)
BARRIER_FLASH = (255, 255, 255)
FLUTTER_COLOR = (210, 235, 255)
STUCK_COLOR = (255, 250, 235)


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


def main():
    canvas = PixelCanvas(title="Study 01 - Barrier Drift")
    W, H = canvas.width, canvas.height
    barrier_y = H // 2

    gas = [new_gas_particle(W, H, barrier_y) for _ in range(GAS_TARGET)]
    flutter = []
    stuck = []  # {x, y, twinkle}
    pressure = [0.0] * W

    t = 0.0
    running = True
    while running:
        running = canvas.poll()
        t += 1 / 60

        # keep the gas population topped up regardless of how many pixels
        # are currently off cycling through flutter/stuck
        while len(gas) < GAS_TARGET:
            gas.append(new_gas_particle(W, H, barrier_y))

        for x in range(W):
            pressure[x] *= PRESSURE_DECAY

        # ---- gas physics: buoyant, turbulent, bounded ----
        for p in gas:
            p["vx"] += math.sin(t * TURB_FREQ + p["phase"]) * TURB_AMP * 0.02
            p["vx"] += random.uniform(-JITTER, JITTER) * 0.02
            p["vy"] += random.uniform(-JITTER, JITTER) * 0.015
            p["vy"] -= BUOYANCY

            p["vx"] *= DAMPING
            p["vy"] *= DAMPING

            p["x"] += p["vx"]
            p["y"] += p["vy"]

            if p["x"] < 0:
                p["x"] = 0
                p["vx"] *= -0.5
            elif p["x"] > W - 1:
                p["x"] = W - 1
                p["vx"] *= -0.5

            if p["y"] > H - 1:
                p["y"] = H - 1
                p["vy"] *= -0.4

            # pressing against the barrier from below
            if p["y"] <= barrier_y + 1 and p["vy"] < 0:
                p["y"] = barrier_y + 1
                p["vy"] *= -0.25
                col = int(p["x"])
                pressure[col] = min(40.0, pressure[col] + PRESSURE_GAIN)

            # been hovering at the line too long without leaking through -
            # give up and sink back into the mix so the lower half keeps
            # circulating instead of draining into a static jam
            if p["y"] <= barrier_y + NEAR_BARRIER_BAND:
                p["press_frames"] += 1
                if p["press_frames"] > p["give_up_after"]:
                    p["y"] = random.uniform(barrier_y + 20, H - 1)
                    p["vy"] = random.uniform(-0.05, 0.05)
                    p["vx"] = random.uniform(-0.1, 0.1)
                    p["press_frames"] = 0
                    p["give_up_after"] = random.randint(PRESS_GIVEUP_MIN, PRESS_GIVEUP_MAX)
            else:
                p["press_frames"] = 0

        # ---- leak check: a pressed pixel occasionally punches through ----
        for x in range(W):
            if pressure[x] <= 15.0:
                continue
            prob = 1 - math.exp(-LEAK_K * pressure[x] * pressure[x])
            if random.random() < prob:
                candidates = [p for p in gas if abs(p["x"] - x) < 2 and p["y"] <= barrier_y + 3]
                if candidates:
                    chosen = random.choice(candidates)
                    gas = [g for g in gas if g is not chosen]
                    chosen["y"] = barrier_y - 2
                    chosen["vy"] = -random.uniform(0.15, 0.35)
                    chosen["vx"] *= 2.0
                    flutter.append(chosen)
                    pressure[x] *= LEAK_RESET

        # ---- flutter physics: wobble, decaying rise, then stick to an edge ----
        still_flying = []
        for p in flutter:
            p["vx"] += math.sin(t * 1.3 + p["phase"]) * FLUTTER_WOBBLE * 0.03
            p["vx"] += random.uniform(-0.05, 0.05)
            p["vy"] -= FLUTTER_RISE * 0.02
            p["vy"] += random.uniform(-0.03, 0.03)

            p["vx"] *= FLUTTER_DAMP
            p["vy"] *= FLUTTER_DAMP

            p["x"] += p["vx"]
            p["y"] += p["vy"]

            landed_on = None
            if p["y"] <= 0:
                p["y"] = 0
                landed_on = "top"
            elif p["x"] <= 0:
                p["x"] = 0
                landed_on = "left"
            elif p["x"] >= W - 1:
                p["x"] = W - 1
                landed_on = "right"
            elif p["y"] >= barrier_y:
                # drifted back down onto the line - rejoin the gas below it,
                # but only if there's room under the target. Keeping the
                # population capped (not just floored) means the top-up
                # spawner below keeps firing forever instead of stalling
                # out once enough rejoins push the count above target.
                p["y"] = barrier_y + 1
                p["vy"] = 0
                p["press_frames"] = 0
                if len(gas) < GAS_TARGET:
                    gas.append(p)
                continue

            if landed_on:
                stuck.append({"x": p["x"], "y": p["y"], "twinkle": random.uniform(0, math.tau)})
            else:
                still_flying.append(p)
        flutter = still_flying

        # ---- draw ----
        canvas.clear((0, 0, 0))

        for x in range(W):
            if pressure[x] > 0.5:
                flash = min(1.0, pressure[x] / 20.0)
                color = tuple(
                    int(BARRIER_COLOR[i] + (BARRIER_FLASH[i] - BARRIER_COLOR[i]) * flash) for i in range(3)
                )
            else:
                color = BARRIER_COLOR
            canvas.set_px(x, barrier_y, color)

        for p in gas:
            flick = 0.5 + 0.5 * math.sin(t * 2.0 + p["phase"] * 3.0)
            color = tuple(int(GAS_COLOR[i] + (GAS_COLOR_HI[i] - GAS_COLOR[i]) * flick * 0.4) for i in range(3))
            canvas.set_px(p["x"], p["y"], color)

        for p in flutter:
            canvas.set_px(p["x"], p["y"], FLUTTER_COLOR)

        for s in stuck:
            tw = 0.75 + 0.25 * math.sin(t * 1.5 + s["twinkle"])
            color = tuple(int(c * tw) for c in STUCK_COLOR)
            canvas.set_px(s["x"], s["y"], color)

        canvas.present(fps=60)

    canvas.quit()


if __name__ == "__main__":
    main()
