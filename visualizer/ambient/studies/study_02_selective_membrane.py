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
identity - they come out dim red, dim green, or dim blue, where before
they were undifferentiated gray-blue gas.

A crossing is telegraphed, not instantaneous. The line rests at a dim,
steady white; when a spot on it wins the right to let something through,
three pixels there start glowing brighter and brighter in the color that
is about to emerge, and the pixel only bursts out at full saturation.
You can see it coming for over a second before it happens. Only a few
places can be building at once, deliberately - a flash that fires
constantly stops reading as an event, so scarcity is doing as much work
here as the animation is.

The two halves are meant to feel like different kinds of thing. Below
the line the gas has no horizontal freedom at all - each pixel is locked
to the column it was born in and can only rise and press - and it is fed
on a strict three-second pulse: a solid bar forms across the bottom of
the screen and spends itself into gas, losing exactly the pixel that
just left it, then rests on a bare floor for three quarters of a second
before the next bar snaps in whole. That rest is what makes the pulse
legible - it sends a gap climbing up through the rising stream, widening
as it goes. That's the bass line the rest of the piece sits on. Above
the barrier, nothing keeps time at all: drift, wobble, deflection,
accumulation.

Above the line, pixels feel a weak pull
toward whatever's already stuck in their own color - red drawn to red,
green to green, blue to blue. The side walls bounce, like the gas does
below the barrier - and so does anything not its own color: a
different-colored cluster is a solid obstacle, not something to pass
through. The only ways to actually come to rest are the very top edge or
touching a stuck pixel of its own color. Combined with the attraction,
that means a pixel can get pulled straight toward its own kind and then
deflected off someone else's color guarding the way - which can produce
some genuinely unpredictable paths.

PERFORMANCE NOTE: the gas below the barrier is a large population (~2000),
updated and redrawn every single frame - at that size it dwarfs every
other cost in the simulation, so
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
# --- the bass line ------------------------------------------------------
# All gas comes from a solid bar of pixels at the very bottom of the screen.
# The bar forms whole, then dissolves over exactly BAR_PERIOD_FRAMES as its
# pixels detach one by one and rise - the bar is literally spending itself to
# make the gas, losing exactly the pixel that just left. It hits empty on the
# last frame of the period and a fresh bar snaps in, which is the downbeat.
#
# Because the bar is the ONLY source, something has to be the sink or the
# population runs away. The sink is the give-up timer at the barrier: a
# particle that presses long enough without getting through is absorbed rather
# than recirculated. That's invisible in practice - PRESS_FADE_FRAMES has
# already faded it to near-black well before its earliest give-up - so it
# reads as gas quietly exhausting itself against the membrane. Population is
# now an emergent number (inflow x lifetime), not a set target.
# The cycle is DRAIN then REST. The bar spends all of itself over the drain,
# then the bottom of the screen is empty and nothing is emitted at all until
# the next bar snaps in whole. That silence is the point: it puts a visible
# gap in the rising stream, which is what proves to anyone watching that the
# gas is coming FROM the bar rather than just appearing. The gap also grows
# as it climbs - particles are still accelerating, so the ones released just
# before the rest keep pulling away from the ones released just after it.
BAR_DRAIN_FRAMES = 135    # 2.25s of the bar dissolving into gas
BAR_REST_FRAMES = 45      # 0.75s of nothing at all - the rest in the bar
BAR_ROWS = 3              # height of the bar; each row is W pixels of fuel
BAR_COLOR = (155, 170, 190)  # denser than the gas it releases - condensed, waiting

GAS_SAFETY_CAP = 6000    # defensive only. Population should settle around 2000 on
                         # its own; if it ever runs past this the bar keeps dissolving
                         # on schedule but stops spawning, so a tuning mistake degrades
                         # the look instead of grinding the framerate to a halt.
GAS_HORIZONTAL_DRIFT = False  # when False, gas below the barrier moves ONLY vertically -
                              # every particle is locked to the column it was born in and
                              # rises in a straight line. The half above the barrier stays
                              # loose and organic; this makes the half below read as
                              # mechanical by contrast - ranks of pixels feeding the line
                              # rather than a drifting cloud. Flip to True for the old
                              # swaying-cloud behavior.
BUOYANCY = 0.012        # constant upward drift for gas particles
JITTER = 0.10           # per-frame random velocity kick (turbulence)
TURB_FREQ = 0.6         # wind-sway frequency (horizontal drift only)
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
PRESS_GIVEUP_MIN = 250   # frames (~4.2s) minimum dwell at the line before being absorbed
PRESS_GIVEUP_MAX = 400   # frames (~6.7s) maximum - randomized per particle so they don't go in sync
                         # NOTE: both are deliberately longer than PRESS_FADE_FRAMES, so a
                         # particle has always faded to PRESS_FADE_FLOOR before it can be
                         # absorbed. That's what makes removal invisible rather than a pop.

PRESS_FADE_FRAMES = 240  # ~4s - a pressed particle dims down over this many frames of
                         # continuous contact, so a crowd hitting the line reads as a slow
                         # fading pulse rather than a static glowing row. It only lights back
                         # up by leaking through (becoming flutter); otherwise this fade is
                         # effectively its death animation - see the give-up note above.
PRESS_FADE_FLOOR = 0.08  # dimmest a still-pressed particle gets - never fully invisible

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

# --- charging crossings -------------------------------------------------
# A crossing is telegraphed rather than instantaneous. When a column earns an
# attempt (enough pressure) and wins its coin flip, it doesn't leak straight
# away - it starts CHARGING. A small window of barrier pixels centered on that
# column brightens toward the color the pixel is going to come out as, and only
# when it reaches full saturation does the pixel actually burst through. The
# barrier itself stays a dim, steady white the rest of the time.
#
# Scarcity is the whole point: a flash that fires constantly stops reading as
# an event. MAX_CONCURRENT_CHARGES hard-caps how many places can be building at
# once regardless of how much pressure exists, which is what keeps each one
# legible as its own small piece of drama.
CHARGE_FRAMES = 100          # ~1.7s from first glimmer to breakthrough
MAX_CONCURRENT_CHARGES = 3   # how many places on the line can be building at once
CHARGE_HALF_WIDTH = 1        # px either side of the crossing column (1 -> 3 px total)
CHARGE_NEIGHBOR_FALLOFF = 0.55  # how bright the flanking pixels get vs the center
CHARGE_MIN_SEPARATION = 6    # px - don't start a new charge this close to an existing one,
                             # so two charges never visually merge into one smear
CHARGE_EASE = 2.2            # >1 makes the glow start slow and rush at the end (anticipation);
                             # 1.0 would be a flat linear ramp

GAS_COLOR = (35, 45, 60)          # neutral gray-blue - undifferentiated, hasn't passed through yet
GAS_COLOR_HI = (60, 75, 95)       # brighter flicker variant
BARRIER_COLOR = (70, 70, 70)      # dim, steady white - the line's resting state

LEAK_COLORS = [
    (50, 0, 0),   # dim red
    (0, 20, 0),   # dim green
    (0, 0, 60),   # dim blue
]

# Saturated counterparts of LEAK_COLORS, by index - what the barrier charges up
# TO. The leak colors themselves are deliberately dim (they have to sit quietly
# above the line for a long time), which is far too dark to read as a building
# threat on the barrier, so the charge glow uses these instead.
CHARGE_GLOW_COLORS = [
    (255, 40, 40),   # red
    (40, 255, 40),   # green
    (60, 90, 255),   # blue
]


def new_gas_particle(x, y):
    """A pixel that has just detached from the bar at exactly (x, y)."""
    return {
        "x": float(x),
        "y": float(y),
        "vx": random.uniform(-0.05, 0.05) if GAS_HORIZONTAL_DRIFT else 0.0,
        "vy": random.uniform(-0.05, 0.0),
        "phase": random.uniform(0, math.tau),
        "press_frames": 0,
        "give_up_after": random.randint(PRESS_GIVEUP_MIN, PRESS_GIVEUP_MAX),
    }


def new_bar(W, H):
    """A full bar of fuel at the bottom of the screen, as a drain-order list.

    Built so that pop()-ing off the end drains the TOP row first and the
    bottom row last: the bar erodes downward from the face nearest the gas,
    which reads as being consumed rather than randomly decaying. Within a row
    the order is shuffled so it dissolves in speckle instead of sweeping left
    to right.
    """
    ordered = []
    for row in range(BAR_ROWS):   # row 0 is the bottom-most line of pixels
        y = H - 1 - row
        row_cells = [(x, y) for x in range(W)]
        random.shuffle(row_cells)
        ordered.extend(row_cells)
    return ordered   # last element = top row = first to leave


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
        return None

    xs = np.fromiter((p["x"] for p in gas), dtype=np.float64, count=n)
    ys = np.fromiter((p["y"] for p in gas), dtype=np.float64, count=n)
    vxs = np.fromiter((p["vx"] for p in gas), dtype=np.float64, count=n)
    vys = np.fromiter((p["vy"] for p in gas), dtype=np.float64, count=n)
    phases = np.fromiter((p["phase"] for p in gas), dtype=np.float64, count=n)
    press_f = np.fromiter((p["press_frames"] for p in gas), dtype=np.float64, count=n)
    give_ups = np.fromiter((p["give_up_after"] for p in gas), dtype=np.float64, count=n)

    # turbulence + jitter + buoyancy
    vys += np.random.uniform(-JITTER, JITTER, n) * 0.015
    vys -= BUOYANCY
    vys *= DAMPING
    ys += vys

    if GAS_HORIZONTAL_DRIFT:
        vxs += np.sin(t * TURB_FREQ + phases) * TURB_AMP * 0.02
        vxs += np.random.uniform(-JITTER, JITTER, n) * 0.02
        vxs *= DAMPING
        xs += vxs

        # side walls: bounce
        left = xs < 0
        xs[left] = 0.0
        vxs[left] *= -0.5
        right = xs > (W - 1)
        xs[right] = float(W - 1)
        vxs[right] *= -0.5
    else:
        # locked to its birth column: no sway, no jitter, no wall contact.
        # x is set once at spawn (already inside [0, W-1]) and never moves.
        vxs[:] = 0.0

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

    # been pressing at the line too long without finding a way through -
    # absorbed. This is the population's only sink (the bar at the bottom is
    # its only source), which is what keeps the count stable without a hard
    # cap. It isn't visible as a disappearance: PRESS_GIVEUP_MIN is longer
    # than PRESS_FADE_FRAMES, so anything eligible to be absorbed has already
    # faded to PRESS_FADE_FLOOR and is barely lit by the time it goes.
    near = ys <= (barrier_y + NEAR_BARRIER_BAND)
    press_f = np.where(near, press_f + 1, 0.0)
    absorbed = near & (press_f > give_ups)

    for i, p in enumerate(gas):
        p["x"] = xs[i]
        p["y"] = ys[i]
        p["vx"] = vxs[i]
        p["vy"] = vys[i]
        p["press_frames"] = press_f[i]
        p["give_up_after"] = give_ups[i]

    return absorbed


def draw_gas_vectorized(canvas, gas, t):
    """Batch version of: for p in gas: compute flicker color, set_px.

    Deliberately re-reads x/y/phase from `gas` rather than reusing the
    arrays update_gas_vectorized built earlier this frame: by render time,
    the leak check has already removed any particle that just crossed the
    barrier (and give-up/rejoin may have added others) - re-extracting
    keeps this in sync with whatever's actually still gas right now.

    Particles that are currently pressed against the barrier (press_frames
    > 0) fade out over PRESS_FADE_FRAMES of continuous contact - without
    this, a large gas population piling up against the line every frame
    reads as a second solid row sitting right under the real barrier row
    instead of an active press that comes and goes.
    """
    n = len(gas)
    if n == 0:
        return
    xs = np.fromiter((p["x"] for p in gas), dtype=np.float64, count=n)
    ys = np.fromiter((p["y"] for p in gas), dtype=np.float64, count=n)
    phases = np.fromiter((p["phase"] for p in gas), dtype=np.float64, count=n)
    press_f = np.fromiter((p["press_frames"] for p in gas), dtype=np.float64, count=n)

    flick = (0.5 + 0.5 * np.sin(t * 2.0 + phases * 3.0)) * 0.4
    colors = np.empty((n, 3), dtype=np.float64)
    for i in range(3):
        colors[:, i] = GAS_COLOR[i] + (GAS_COLOR_HI[i] - GAS_COLOR[i]) * flick

    fade = np.clip(press_f / PRESS_FADE_FRAMES, 0.0, 1.0)
    brightness = 1.0 - fade * (1.0 - PRESS_FADE_FLOOR)
    colors *= brightness[:, None]
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

    gas = []   # starts empty - the first bar has to dissolve before there's
               # any gas at all, so the opening builds instead of inrushing
    flutter = []
    stuck = []      # [{x, y, color, twinkle}]
    stuck_positions_by_color = {c: set() for c in LEAK_COLORS}  # mirrors `stuck`, split by color
    pressure = np.zeros(W, dtype=np.float64)
    stuck_render_cache = {"n": -1}

    # crossings currently building up on the line, each
    # {x, color_index, progress} with progress running 0 -> 1
    charges = []

    bar = new_bar(W, H)      # remaining fuel pixels, in drain order
    bar_total = len(bar)
    bar_frame = 0            # frames elapsed in the current bar's cycle

    t = 0.0
    running = True
    while running:
        running = canvas.poll()
        t += 1 / 60

        # ---- the bass line: the bar spends itself into gas ----
        # Emission is paced off elapsed time rather than a fixed per-frame
        # count, so the bar lands on exactly empty at the end of the period
        # no matter how the integer division falls. Every pixel that leaves
        # the bar becomes the gas particle that rises out of that same spot.
        bar_frame += 1
        if bar_frame <= BAR_DRAIN_FRAMES:
            should_remain = int(round(bar_total * (1.0 - bar_frame / BAR_DRAIN_FRAMES)))
            should_remain = max(0, should_remain)
            while len(bar) > should_remain:
                bx, by = bar.pop()
                if len(gas) < GAS_SAFETY_CAP:
                    gas.append(new_gas_particle(bx, by))
        # else: the rest. `bar` is empty, so nothing draws and nothing is born.

        if bar_frame >= BAR_DRAIN_FRAMES + BAR_REST_FRAMES:
            # downbeat - a whole new bar snaps in
            bar = new_bar(W, H)
            bar_total = len(bar)
            bar_frame = 0

        pressure *= PRESSURE_DECAY

        # ---- gas physics: buoyant, turbulent, bounded (vectorized) ----
        absorbed = update_gas_vectorized(gas, t, W, H, barrier_y, pressure)
        if absorbed is not None and absorbed.any():
            gas = [p for i, p in enumerate(gas) if not absorbed[i]]

        # ---- leak check: pressure earns an attempt, then a coin flip, then
        # the winner starts CHARGING rather than crossing immediately ----
        if len(charges) < MAX_CONCURRENT_CHARGES:
            for x in range(W):
                if pressure[x] <= 15.0:
                    continue
                # don't stack a new charge on top of one already building nearby
                if any(abs(c["x"] - x) < CHARGE_MIN_SEPARATION for c in charges):
                    continue
                prob = 1 - math.exp(-LEAK_K * pressure[x] * pressure[x])
                if random.random() < prob:
                    # only worth starting if there's actually gas here to send through
                    has_candidate = any(
                        abs(p["x"] - x) < 2 and p["y"] <= barrier_y + 3 for p in gas
                    )
                    if has_candidate:
                        if random.random() < PERMEABLE_FRACTION:
                            charges.append({
                                "x": x,
                                "color_index": random.randrange(len(LEAK_COLORS)),
                                "progress": 0.0,
                            })
                            if len(charges) >= MAX_CONCURRENT_CHARGES:
                                break
                        else:
                            # pressed hard enough to try, but this attempt was
                            # one of the failures that doesn't make it through
                            pressure[x] *= FAIL_RELIEF

        # ---- advance charges; a full one bursts a pixel through ----
        still_charging = []
        for ch in charges:
            ch["progress"] += 1.0 / CHARGE_FRAMES
            if ch["progress"] < 1.0:
                still_charging.append(ch)
                continue

            # fully saturated - this is the moment the pixel comes through
            x = ch["x"]
            candidates = [p for p in gas if abs(p["x"] - x) < 2 and p["y"] <= barrier_y + 3]
            if candidates:
                chosen = random.choice(candidates)
                gas = [g for g in gas if g is not chosen]
                chosen["y"] = barrier_y - 2
                chosen["vy"] = -random.uniform(0.15, 0.35)
                chosen["vx"] *= 2.0
                chosen["color"] = LEAK_COLORS[ch["color_index"]]
                chosen["attract_target"] = None
                chosen["attract_refresh_in"] = 0  # pick a target on its very first flutter frame
                flutter.append(chosen)
                pressure[x] *= LEAK_RESET
            # if the gas happened to clear out of this column during the
            # charge, the build-up simply fizzles - the glow ends and nothing
            # crosses. Rare, and honest: the pressure wasn't there after all.
        charges = still_charging

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
                # losing its color identity in the process. No cap check
                # needed any more - flutter is small and rare, and the
                # absorb-at-the-barrier sink handles population on its own.
                p["y"] = barrier_y + 1
                p["vy"] = 0
                p["press_frames"] = 0
                p.pop("color", None)
                gas.append(p)
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

        # the line's resting state: dim, steady, uniform. It no longer reacts
        # to pressure at all - pressure sits near its cap on almost every
        # column almost all the time, so a pressure-driven flash was pinned
        # bright permanently and read as a solid glowing bar rather than as
        # anything happening.
        for x in range(W):
            canvas.set_px(x, barrier_y, BARRIER_COLOR)

        # the only thing that lights the line is an imminent crossing: a small
        # window brightening toward the color that's about to come through
        for ch in charges:
            glow = CHARGE_GLOW_COLORS[ch["color_index"]]
            eased = ch["progress"] ** CHARGE_EASE
            for dx in range(-CHARGE_HALF_WIDTH, CHARGE_HALF_WIDTH + 1):
                xx = ch["x"] + dx
                if 0 <= xx < W:
                    falloff = 1.0 if dx == 0 else CHARGE_NEIGHBOR_FALLOFF
                    canvas.set_px(xx, barrier_y, lerp_color(BARRIER_COLOR, glow, eased * falloff))

        # the bar of remaining fuel at the bottom, eroding in real time
        for (bx, by) in bar:
            canvas.set_px(bx, by, BAR_COLOR)

        draw_gas_vectorized(canvas, gas, t)

        for p in flutter:
            canvas.set_px(p["x"], p["y"], p["color"])

        draw_stuck_vectorized(canvas, stuck, t, stuck_render_cache)

        canvas.present(fps=60)

    canvas.quit()


if __name__ == "__main__":
    main()
