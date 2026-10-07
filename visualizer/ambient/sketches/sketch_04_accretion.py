"""
Sketch 04 - Accretion
====================

A variation on Sketch 03 (Affinity). Everything below the line is identical:
the same bar of fuel dissolving into gas and then resting on a bare floor,
the same columns of pixels rising and pressing, the same membrane that
telegraphs a crossing before it happens. The top half still has no
stickiness at all - pixels drift, blown by a wind, drawn toward their own
color and pushed away from the others.

What changes is how a pixel ends.

In Sketch 03 drifting pixels simply timed out. It worked, but it was a
bookkeeping device rather than an event: pixels kept arriving through the
membrane and pixels kept quietly ceasing to exist, and nothing on screen
explained why. There was no reason for anything to disappear.

So here there's a reason. Every sixth bar - once every eighteen seconds or
so - a black hole opens somewhere at random in the upper field. It is a
single pixel, almost always dark, and what it does is bend the wind: within
its reach the prevailing flow is replaced by an inward spiral, so anything
nearby stops being carried by the weather and starts falling. Pixels wind
in, accelerating, and vanish at the center. It flares each time it takes
one - that flicker is the only thing that gives it away - and then after a
few seconds it closes, and the field is quiet again until the next one
opens somewhere else.

Population is now a genuine equilibrium rather than a set number: pixels
arrive at whatever rate the membrane allows, and leave at whatever rate the
black holes can catch them, which rises and falls with how crowded the
field is. Nobody sets the total. It finds itself.

Over all of it blows a wind: a wide, slow, swirling flow that everything
is carried by. The affinity happens in the wind's own frame of reference,
so a group holds together while the weather moves it - it drifts as a
group rather than being pulled apart. And the pull has a partner: the
same force turned ninety degrees, so a gathering doesn't just contract,
it turns. Both scale with how much of your own color is nearby, which
means a lone pixel is barely affected by anything, a pair drifts together
almost imperceptibly, and a dense knot holds itself hard and rotates
visibly as it travels.

That's the entire top-half rule, and it's enough. Loose pixels find each
other. Groups form, drift as groups, deform, collide with other-colored
groups and slide off them, merge with groups of their own color, and shed
stragglers. Nothing is fixed - a cluster is only a cluster for as long as
its members keep choosing it.

It also isn't absolute. Every so often a pixel simply stops caring for a
while: it ignores its own color entirely, wanders off on its own momentum,
and only later re-develops a preference. That's what keeps the field from
settling into three tidy blobs - there is always some traffic between
groups, and always a few pixels out in the dark on their own.

HOW THE AFFINITY WORKS: not pixel-to-pixel. Every frame the drifting pixels
are accumulated into a coarse, blurred density map per color - basically a
low-resolution picture of "where is there a lot of red." A pixel then reads
the slope of that map underneath itself and drifts uphill on its own color's
map and downhill on everyone else's, plus downhill on a much sharper
all-colors map, which is what stops a group from collapsing into a single
overlapping point. This costs the same whether there are fifty pixels or
five thousand, and it produces smoother, more fluid grouping than
nearest-neighbor attraction does - clusters have soft edges and get pulled
toward each other as wholes.

Run:  python sketch_04_accretion.py
Opens fullscreen on the LED monitor (display 1) at native resolution,
positioned per led_config.json. Esc or close the window to quit.
"""

import os
import sys
import math
import random

import numpy as np

# The shared canvas harness lives one level up, in ambient/, rather than
# alongside this file. That's deliberate: a piece gets promoted between tiers by
# moving it (sketches/ -> studies/ -> selects/), and resolving the harness
# relative to the PARENT directory means that move never breaks the import.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMBIENT_DIR = os.path.dirname(SCRIPT_DIR)
if AMBIENT_DIR not in sys.path:
    sys.path.append(AMBIENT_DIR)

from _pixel_canvas import PixelCanvas

# --- the bass line (identical to Sketch 02) ------------------------------
# The cycle is DRAIN then REST. The bar spends all of itself over the drain,
# then the bottom of the screen is empty and nothing is emitted at all until
# the next bar snaps in whole. That silence is the point: it puts a visible
# gap in the rising stream, which is what proves to anyone watching that the
# gas is coming FROM the bar rather than just appearing. The gap also grows
# as it climbs - particles are still accelerating, so the ones released just
# before the rest keep pulling away from the ones released just after it.
BAR_DRAIN_FRAMES = 110    # 2.25s of the bar dissolving into gas
BAR_REST_FRAMES = 55      # 0.75s of nothing at all - the rest in the bar
BAR_ROWS = 3              # height of the bar; each row is W pixels of fuel
BAR_COLOR = (155, 170, 190)

GAS_SAFETY_CAP = 6000
GAS_HORIZONTAL_DRIFT = False   # gas below the line is locked to its birth column

BUOYANCY = 0.012
JITTER = 0.10
TURB_FREQ = 0.6
TURB_AMP = 0.05
DAMPING = 0.985

PRESSURE_GAIN = 1.0
PRESSURE_DECAY = 0.965
LEAK_K = 0.0000008
LEAK_RESET = 0.35

PERMEABLE_FRACTION = 0.5
FAIL_RELIEF = 0.85

NEAR_BARRIER_BAND = 6
PRESS_GIVEUP_MIN = 250   # must stay > PRESS_FADE_FRAMES so absorption is invisible
PRESS_GIVEUP_MAX = 400

PRESS_FADE_FRAMES = 240
PRESS_FADE_FLOOR = 0.08

# --- charging crossings (identical mechanic, tuned busier) ---------------
# Sketch 02 wants each crossing to be a rare, legible event because each one
# becomes a permanent mark. Here nothing is permanent, so the top half needs
# a real population to have anything to flock WITH - hence more charges at
# once and a shorter build. It's still telegraphed, just less ceremonial.
CHARGE_FRAMES = 80
MAX_CONCURRENT_CHARGES = 5
CHARGE_HALF_WIDTH = 1
CHARGE_NEIGHBOR_FALLOFF = 0.55
CHARGE_MIN_SEPARATION = 6
CHARGE_EASE = 2.2

# --- the drifting field (inherited from Sketch 03) ------------------------
FIELD_CELL = 4            # px per density-map cell. Smaller = finer grouping, more cost.
COHESION_BLUR = 4         # blur passes on the own-color map. More = larger, softer groups.
SEPARATION_BLUR = 1       # blur passes on the all-colors map used for personal space

# Cohesion and spin both scale with how much of your own color is nearby, so
# a lone pixel is barely pulled at all and a dense knot holds itself together
# hard. CLUMP_GAIN converts "smoothed same-color pixels per cell" into that
# multiplier; CLUMP_MAX stops a very dense knot from becoming a black hole.
CLUMP_GAIN = 1.4
CLUMP_MAX = 2.0
# CLUMP_GAIN has to be re-tuned whenever the population changes a lot. It
# converts local density into the mass multiplier, and CLUMP_MAX caps that -
# so if the field gets denser without this coming down, nearly every pixel
# ends up pinned at the cap, every pixel gets identical maximum cohesion, and
# the whole "a loner barely feels it, a knot holds hard" distinction
# disappears into uniform mush. Rough guide: 3.0 at ~450 drifters, 1.4 at
# ~850, 0.7 at ~2000. Halve it whenever you double the population.

COHESION_STRENGTH = 0.013     # pull toward more of your own color, x clump factor
SPIN_STRENGTH = 0.011         # push at right angles to that pull - makes clumps turn
SPIN_SIGNS = (1.0, -1.0, 1.0)  # per color, so the three species don't all rotate alike
OTHER_REPEL_STRENGTH = 0.07   # push away from concentrations of the other colors
SEPARATION_STRENGTH = 0.05    # short-range push so a group stays a cloud, not a dot

# --- wind ---------------------------------------------------------------
# A large-scale flow that everything is carried by, built from a stream
# function so it's divergence-free: it swirls and shifts but has no sinks for
# pixels to pile up in, which a naive sum-of-sines wind field would have.
# Pixels are advected toward the local wind velocity rather than shoved by it,
# so a whole clump drifts together instead of being torn apart.
WIND_SPEED = 0.45         # px/frame - the speed the flow carries things at
WIND_SCALE_A = 210.0      # px - wavelength of the big slow swirls. Bigger than the panel is
                          # deliberate: at ~130 the flow reversed every 65px and pixels just
                          # rocked back and forth in place. Wide, lazy cells let a clump
                          # actually travel somewhere before the current turns it around.
WIND_SCALE_B = 99.0       # px - wavelength of the finer overlaid swirls
WIND_RATE_A = 0.13        # how fast the big swirls evolve
WIND_RATE_B = 0.21

DRIFT_RISE = 0.004        # slight constant lift, so pixels don't settle onto the membrane
BOUNDARY_BAND = 22        # px - how far from the ceiling/membrane the soft push starts
BOUNDARY_PUSH = 0.020     # strength of that push at the very edge, easing to 0 at the band

DRIFT_NOISE = 0.012       # per-frame random velocity, keeps the field from looking scripted
DRIFT_DAMP = 0.97         # damping of motion RELATIVE to the wind. Because everything below
                          # happens in the wind's own moving frame, this no longer fights the
                          # flow - it only settles a pixel's private motion back down into the
                          # current, so it doubles as how much lag/inertia a pixel has.
DRIFT_REL_MAX_SPEED = 0.5  # px/frame ceiling on motion relative to the wind (absolute speed
                           # can reach roughly this plus WIND_SPEED)
WALL_BOUNCE = -0.6        # velocity multiplier when hitting an edge or the membrane

# Defection: occasionally a pixel stops caring about its own color entirely
# and coasts on whatever momentum it has. This is the whole reason the field
# never resolves into three static blobs - it guarantees a trickle of loose
# pixels and traffic between groups.
WANDER_CHANCE = 0.0009     # per pixel per frame (~1 in 18s each)
WANDER_FRAMES_MIN = 120    # 2s adrift
WANDER_FRAMES_MAX = 420    # 7s adrift
WANDER_KICK = 0.35         # impulse applied the moment it breaks away

DRIFT_SHIMMER = 0.18      # gentle per-pixel brightness wobble

# --- the black hole -----------------------------------------------------
# Drifting pixels have no lifespan at all in this sketch. The ONLY way one
# leaves the field is by falling into a black hole, which means the total
# population is an equilibrium between the membrane's crossing rate and the
# black holes' catch rate - and because the catch rate scales with how many
# pixels are around to be caught, it self-corrects. A crowded field is eaten
# faster; a sparse one is barely touched. Nothing sets the number directly.
#
# BH_RANGE and BH_LIFE_FRAMES are the two dials that set where that
# equilibrium lands. Both raise the catch rate, so both LOWER the steady
# population - if the field is thinning out, shrink them.
BH_EVERY_N_BARS = 6       # one opens every sixth bar cycle
BH_LIFE_FRAMES = 300      # ~3.3s open, then it closes and the field is quiet
BH_RANGE = 34             # px of reach - beyond this the wind is untouched
BH_FALLOFF = 1.5          # >1 concentrates the pull near the middle of that reach
BH_SPEED = 0.6            # px/frame of infall at the edge of its reach
BH_ACCEL = 2.2            # how much faster the fall gets approaching the center
BH_SWIRL = 1.0            # sideways component of the infall. 0 = straight in,
                          # 1 = a 45-degree spiral. This is what makes it wind in
                          # rather than drop, and it's most of the character.
BH_HORIZON = 1.6          # px - inside this a pixel is gone
BH_MARGIN = 18            # px - keep it this far off the walls and the membrane

BH_IDLE_COLOR = (9, 9, 10)        # almost off - a dead pixel until it isn't
BH_FLASH_COLOR = (210, 215, 235)  # what it flares to when it swallows something
BH_FLASH_DECAY = 0.80             # per-frame decay of that flare
BH_IDLE_FLICKER = 0.006           # small chance per frame of a faint unprompted twitch

GAS_COLOR = (35, 45, 60)
GAS_COLOR_HI = (60, 75, 95)
BARRIER_COLOR = (70, 70, 70)

# Brighter than Sketch 02's leak colors: these never pile up into dense
# accreted masses, they stay spread out and moving, so they need more
# individual presence to read at all.
DRIFT_COLORS = [
    (110, 80, 80),   # red
    (80, 110, 80),   # green
    (80, 80, 110),   # blue
]

CHARGE_GLOW_COLORS = [
    (255, 40, 40),
    (40, 255, 40),
    (60, 90, 255),
]


# =======================================================================
# lower half - unchanged from Sketch 02
# =======================================================================

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
    """Full bar of fuel, ordered so pop() drains the top row first."""
    ordered = []
    for row in range(BAR_ROWS):
        y = H - 1 - row
        row_cells = [(x, y) for x in range(W)]
        random.shuffle(row_cells)
        ordered.extend(row_cells)
    return ordered


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def update_gas_vectorized(gas, t, W, H, barrier_y, pressure):
    """One frame of gas physics for the whole population (numpy).

    Returns a boolean mask of particles absorbed at the barrier this frame;
    the caller filters them out. Absorption is the population's only sink
    (the bar is its only source), and it's invisible because anything old
    enough to be absorbed has already faded to PRESS_FADE_FLOOR.
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

    vys += np.random.uniform(-JITTER, JITTER, n) * 0.015
    vys -= BUOYANCY
    vys *= DAMPING
    ys += vys

    if GAS_HORIZONTAL_DRIFT:
        vxs += np.sin(t * TURB_FREQ + phases) * TURB_AMP * 0.02
        vxs += np.random.uniform(-JITTER, JITTER, n) * 0.02
        vxs *= DAMPING
        xs += vxs
        left = xs < 0
        xs[left] = 0.0
        vxs[left] *= -0.5
        right = xs > (W - 1)
        xs[right] = float(W - 1)
        vxs[right] *= -0.5
    else:
        vxs[:] = 0.0

    floor = ys > (H - 1)
    ys[floor] = float(H - 1)
    vys[floor] *= -0.4

    hit = (ys <= barrier_y + 1) & (vys < 0)
    ys[hit] = float(barrier_y + 1)
    vys[hit] *= -0.25
    if np.any(hit):
        hit_cols = np.clip(xs[hit].astype(np.int64), 0, W - 1)
        np.add.at(pressure, hit_cols, PRESSURE_GAIN)
        np.clip(pressure, None, 40.0, out=pressure)

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
    """Batch draw of the gas, with the press-fade applied."""
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
    colors *= (1.0 - fade * (1.0 - PRESS_FADE_FLOOR))[:, None]
    colors = colors.astype(np.uint8)

    xi = np.round(xs).astype(np.int64)
    yi = np.round(ys).astype(np.int64)
    valid = (xi >= 0) & (xi < canvas.width) & (yi >= 0) & (yi < canvas.height)
    canvas.buffer[yi[valid], xi[valid]] = colors[valid]


# =======================================================================
# upper half - the affinity field
# =======================================================================

def new_drifter(x, y, color_index):
    return {
        "x": float(x),
        "y": float(y),
        "vx": random.uniform(-0.3, 0.3),
        "vy": -random.uniform(0.15, 0.35),
        "ci": color_index,
        "wander": 0,
        "phase": random.uniform(0, math.tau),
    }


def box_blur(a, passes):
    """Separable 3-tap box blur with edge clamping.

    Edge-clamped rather than wrapped on purpose: with np.roll the left edge
    of the screen would attract pixels toward the right edge, and clusters
    would visibly get tugged off the sides of the panel.
    """
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = (p[:-2, 1:-1] + p[1:-1, 1:-1] + p[2:, 1:-1]) * (1.0 / 3.0)
        p = np.pad(a, 1, mode="edge")
        a = (p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]) * (1.0 / 3.0)
    return a


def wind_at(xs, ys, t):
    """Large-scale flow velocity at each position.

    Derived from a stream function psi as (dpsi/dy, -dpsi/dx), which makes it
    divergence-free by construction: the flow swirls and reshapes over time but
    can never have a sink that pixels accumulate in. That matters here - a
    plain sum-of-sines velocity field would quietly herd everything into its
    convergence zones and undo the clustering we actually want to see.
    """
    ka = math.tau / WIND_SCALE_A
    kb = math.tau / WIND_SCALE_B
    aa = WIND_SPEED * 0.62 / ka
    ab = WIND_SPEED * 0.38 / kb

    pa_x = ka * xs + WIND_RATE_A * t
    pa_y = ka * ys - WIND_RATE_A * t
    pb_y = kb * ys + WIND_RATE_B * t
    pb_x = kb * xs - WIND_RATE_B * t

    # psi_a = sin(pa_x) * cos(pa_y);  psi_b = sin(pb_y) * cos(pb_x)
    dpsi_dy = -aa * ka * np.sin(pa_x) * np.sin(pa_y) + ab * kb * np.cos(pb_y) * np.cos(pb_x)
    dpsi_dx = aa * ka * np.cos(pa_x) * np.cos(pa_y) - ab * kb * np.sin(pb_y) * np.sin(pb_x)

    return dpsi_dy, -dpsi_dx


def build_affinity_fields(cxs, cys, cis, gw, gh):
    """Smoothed density maps per color, plus their gradients.

    Returns (wide, own_grad, other_grad, sharp_grad). `wide` is deliberately
    NOT normalized: its absolute value is "roughly how many same-color pixels
    are smeared into this cell", which is what lets cohesion and spin scale
    with the size of a clump instead of treating a lone pixel like a crowd.

    Cost is O(pixels + grid) rather than O(pixels^2), so the population can
    grow without the frame time following it.
    """
    n_colors = len(DRIFT_COLORS)
    dens = np.zeros((n_colors, gh, gw), dtype=np.float64)
    for c in range(n_colors):
        sel = cis == c
        if np.any(sel):
            np.add.at(dens[c], (cys[sel], cxs[sel]), 1.0)

    # personal space: relative is fine here, so this one IS normalized
    total_sharp = box_blur(dens.sum(axis=0), SEPARATION_BLUR)
    total_sharp /= (total_sharp.max() + 1e-6)
    sharp_grad = np.gradient(total_sharp)

    wide = [box_blur(dens[c], COHESION_BLUR) for c in range(n_colors)]
    own_grad = [np.gradient(wide[c]) for c in range(n_colors)]

    other_grad = []
    for c in range(n_colors):
        others = sum(wide[o] for o in range(n_colors) if o != c)
        peak = others.max() + 1e-6
        other_grad.append(np.gradient(others / peak))

    return wide, own_grad, other_grad, sharp_grad


def new_black_hole(W, barrier_y):
    return {
        "x": random.uniform(BH_MARGIN, W - 1 - BH_MARGIN),
        "y": random.uniform(BH_MARGIN, barrier_y - 1 - BH_MARGIN),
        "age": 0,
        "flash": 1.0,   # it announces itself as it opens
        "eaten": 0,
    }


def update_drifters(drifters, t, W, barrier_y, bh):
    """One frame of the affinity field.

    Returns (survivors, n_swallowed). `bh` is the open black hole or None.
    """
    n = len(drifters)
    if n == 0:
        return drifters, 0

    xs = np.fromiter((p["x"] for p in drifters), dtype=np.float64, count=n)
    ys = np.fromiter((p["y"] for p in drifters), dtype=np.float64, count=n)
    vxs = np.fromiter((p["vx"] for p in drifters), dtype=np.float64, count=n)
    vys = np.fromiter((p["vy"] for p in drifters), dtype=np.float64, count=n)
    cis = np.fromiter((p["ci"] for p in drifters), dtype=np.int64, count=n)
    wander = np.fromiter((p["wander"] for p in drifters), dtype=np.int64, count=n)

    gw = max(1, W // FIELD_CELL)
    gh = max(1, barrier_y // FIELD_CELL)
    cxs = np.clip((xs / FIELD_CELL).astype(np.int64), 0, gw - 1)
    cys = np.clip((ys / FIELD_CELL).astype(np.int64), 0, gh - 1)

    wide, own_grad, other_grad, sharp_grad = build_affinity_fields(cxs, cys, cis, gw, gh)

    # gather each pixel's own-color field values into flat arrays
    own_gx = np.zeros(n)
    own_gy = np.zeros(n)
    own_d = np.zeros(n)
    oth_gx = np.zeros(n)
    oth_gy = np.zeros(n)
    spin_sign = np.zeros(n)
    for c in range(len(DRIFT_COLORS)):
        sel = cis == c
        if not np.any(sel):
            continue
        cy_s, cx_s = cys[sel], cxs[sel]
        gy, gx = own_grad[c]
        own_gx[sel] = gx[cy_s, cx_s]
        own_gy[sel] = gy[cy_s, cx_s]
        own_d[sel] = wide[c][cy_s, cx_s]
        gy, gx = other_grad[c]
        oth_gx[sel] = gx[cy_s, cx_s]
        oth_gy[sel] = gy[cy_s, cx_s]
        spin_sign[sel] = SPIN_SIGNS[c]

    # "how much of my own kind is around me" - this is the mass term. A pixel
    # on its own gets almost nothing; a dense knot pulls and turns hard.
    clump = np.clip(own_d * CLUMP_GAIN, 0.0, CLUMP_MAX)

    # direction toward the local concentration of my own color
    gmag = np.hypot(own_gx, own_gy) + 1e-9
    ux = own_gx / gmag
    uy = own_gy / gmag

    # a pixel that's currently wandering ignores its own color completely,
    # but still respects personal space and still gets shoved by other colors
    caring = (wander <= 0).astype(np.float64)

    # Everything from here happens in the WIND'S frame of reference. A pixel's
    # velocity is split into "where the current is going" plus "what it's doing
    # on its own", and clustering, spin and damping all act only on the second
    # part. That's the difference between a clump that blows around intact and
    # one that gets dragged apart - and it means damping no longer works
    # against the wind, which was previously capping everything at about half
    # the flow speed no matter how hard the wind blew.
    wx, wy = wind_at(xs, ys, t)

    # A black hole doesn't push pixels around - it replaces the weather. Inside
    # its reach the ambient flow is blended toward an inward spiral, so a pixel
    # near it simply stops being carried by the wind and starts falling instead.
    # Doing it this way (rather than adding a gravity force) means the wind can
    # never rescue something that's already committed, and the infall reads as
    # the flow itself bending rather than as pixels being yanked.
    swallowed = np.zeros(n, dtype=bool)
    if bh is not None:
        dx = bh["x"] - xs
        dy = bh["y"] - ys
        dist = np.hypot(dx, dy)
        safe = np.maximum(dist, 1e-6)
        closeness = np.clip(1.0 - dist / BH_RANGE, 0.0, 1.0)
        infl = closeness ** BH_FALLOFF

        rx, ry = dx / safe, dy / safe          # straight at it
        ix = rx - ry * BH_SWIRL                # ...plus a sideways component
        iy = ry + rx * BH_SWIRL                #    which turns the fall into a spiral
        imag = np.hypot(ix, iy) + 1e-9
        fall = BH_SPEED * (1.0 + BH_ACCEL * closeness)

        wx = wx * (1.0 - infl) + (ix / imag) * fall * infl
        wy = wy * (1.0 - infl) + (iy / imag) * fall * infl

        swallowed = dist < BH_HORIZON

    rvx = vxs - wx
    rvy = vys - wy

    # radial pull inward...
    rvx += ux * (COHESION_STRENGTH * clump * caring)
    rvy += uy * (COHESION_STRENGTH * clump * caring)

    # ...and the same force turned 90 degrees, which is what makes a clump
    # rotate rather than just contract. Scaled by the same mass term, so a
    # big group visibly turns and a pair barely does.
    rvx += -uy * (SPIN_STRENGTH * clump * spin_sign * caring)
    rvy += ux * (SPIN_STRENGTH * clump * spin_sign * caring)

    rvx -= oth_gx * OTHER_REPEL_STRENGTH
    rvy -= oth_gy * OTHER_REPEL_STRENGTH

    gy, gx = sharp_grad
    rvx -= gx[cys, cxs] * SEPARATION_STRENGTH
    rvy -= gy[cys, cxs] * SEPARATION_STRENGTH

    # gentle lift, plus soft walls that keep the field off the membrane and
    # the ceiling instead of letting it settle against either
    rvy -= DRIFT_RISE
    near_top = np.clip(1.0 - ys / BOUNDARY_BAND, 0.0, 1.0)
    near_bot = np.clip(1.0 - (barrier_y - 1 - ys) / BOUNDARY_BAND, 0.0, 1.0)
    rvy += near_top * BOUNDARY_PUSH
    rvy -= near_bot * BOUNDARY_PUSH

    rvx += np.random.uniform(-DRIFT_NOISE, DRIFT_NOISE, n)
    rvy += np.random.uniform(-DRIFT_NOISE, DRIFT_NOISE, n)

    # break-aways: the kick is relative to the wind too, so breaking away
    # means leaving your group - not leaving the weather
    starting = (wander <= 0) & (np.random.random(n) < WANDER_CHANCE)
    n_start = int(np.count_nonzero(starting))
    if n_start:
        wander[starting] = np.random.randint(WANDER_FRAMES_MIN, WANDER_FRAMES_MAX + 1, n_start)
        angles = np.random.uniform(0, math.tau, n_start)
        rvx[starting] += np.cos(angles) * WANDER_KICK
        rvy[starting] += np.sin(angles) * WANDER_KICK
    wander = np.maximum(wander - 1, 0)

    rvx *= DRIFT_DAMP
    rvy *= DRIFT_DAMP

    rspeed = np.hypot(rvx, rvy)
    too_fast = rspeed > DRIFT_REL_MAX_SPEED
    if np.any(too_fast):
        scale = DRIFT_REL_MAX_SPEED / rspeed[too_fast]
        rvx[too_fast] *= scale
        rvy[too_fast] *= scale

    # back out of the wind's frame into actual screen motion
    vxs = wx + rvx
    vys = wy + rvy

    xs += vxs
    ys += vys

    # walls, ceiling, and the membrane underneath - all bounce, none stick
    left = xs < 0
    xs[left] = 0.0
    vxs[left] *= WALL_BOUNCE
    right = xs > (W - 1)
    xs[right] = float(W - 1)
    vxs[right] *= WALL_BOUNCE

    top = ys < 0
    ys[top] = 0.0
    vys[top] *= WALL_BOUNCE
    bottom = ys > (barrier_y - 1)
    ys[bottom] = float(barrier_y - 1)
    vys[bottom] *= WALL_BOUNCE

    survivors = []
    for i, p in enumerate(drifters):
        if swallowed[i]:
            continue
        p["x"] = xs[i]
        p["y"] = ys[i]
        p["vx"] = vxs[i]
        p["vy"] = vys[i]
        p["wander"] = int(wander[i])
        survivors.append(p)
    return survivors, int(np.count_nonzero(swallowed))


def draw_drifters(canvas, drifters, t):
    n = len(drifters)
    if n == 0:
        return
    xs = np.fromiter((p["x"] for p in drifters), dtype=np.float64, count=n)
    ys = np.fromiter((p["y"] for p in drifters), dtype=np.float64, count=n)
    cis = np.fromiter((p["ci"] for p in drifters), dtype=np.int64, count=n)
    phases = np.fromiter((p["phase"] for p in drifters), dtype=np.float64, count=n)

    base = np.array(DRIFT_COLORS, dtype=np.float64)[cis]
    shimmer = 1.0 - DRIFT_SHIMMER * (0.5 + 0.5 * np.sin(t * 1.7 + phases))
    colors = (base * shimmer[:, None]).astype(np.uint8)

    xi = np.round(xs).astype(np.int64)
    yi = np.round(ys).astype(np.int64)
    valid = (xi >= 0) & (xi < canvas.width) & (yi >= 0) & (yi < canvas.height)
    canvas.buffer[yi[valid], xi[valid]] = colors[valid]


# =======================================================================

def main():
    canvas = PixelCanvas(title="Sketch 04 - Accretion")
    W, H = canvas.width, canvas.height
    barrier_y = H // 2

    gas = []
    drifters = []
    pressure = np.zeros(W, dtype=np.float64)
    charges = []

    bar = new_bar(W, H)
    bar_total = len(bar)
    bar_frame = 0
    bars_done = 0        # completed bar cycles, counts toward the next black hole
    black_hole = None

    t = 0.0
    running = True
    while running:
        running = canvas.poll()
        t += 1 / 60

        # ---- the bass line: the bar spends itself into gas, then rests ----
        bar_frame += 1
        if bar_frame <= BAR_DRAIN_FRAMES:
            should_remain = max(0, int(round(bar_total * (1.0 - bar_frame / BAR_DRAIN_FRAMES))))
            while len(bar) > should_remain:
                bx, by = bar.pop()
                if len(gas) < GAS_SAFETY_CAP:
                    gas.append(new_gas_particle(bx, by))
        # else: the rest. `bar` is empty, so nothing draws and nothing is born.

        if bar_frame >= BAR_DRAIN_FRAMES + BAR_REST_FRAMES:
            bar = new_bar(W, H)
            bar_total = len(bar)
            bar_frame = 0
            bars_done += 1
            if bars_done % BH_EVERY_N_BARS == 0:
                black_hole = new_black_hole(W, barrier_y)

        pressure *= PRESSURE_DECAY

        absorbed = update_gas_vectorized(gas, t, W, H, barrier_y, pressure)
        if absorbed is not None and absorbed.any():
            gas = [p for i, p in enumerate(gas) if not absorbed[i]]

        # ---- pressure earns an attempt; the winner starts charging ----
        if len(charges) < MAX_CONCURRENT_CHARGES:
            for x in range(W):
                if pressure[x] <= 15.0:
                    continue
                if any(abs(c["x"] - x) < CHARGE_MIN_SEPARATION for c in charges):
                    continue
                prob = 1 - math.exp(-LEAK_K * pressure[x] * pressure[x])
                if random.random() < prob:
                    has_candidate = any(
                        abs(p["x"] - x) < 2 and p["y"] <= barrier_y + 3 for p in gas
                    )
                    if has_candidate:
                        if random.random() < PERMEABLE_FRACTION:
                            charges.append({
                                "x": x,
                                "color_index": random.randrange(len(DRIFT_COLORS)),
                                "progress": 0.0,
                            })
                            if len(charges) >= MAX_CONCURRENT_CHARGES:
                                break
                        else:
                            pressure[x] *= FAIL_RELIEF

        # ---- advance charges; a full one releases a drifter ----
        still_charging = []
        for ch in charges:
            ch["progress"] += 1.0 / CHARGE_FRAMES
            if ch["progress"] < 1.0:
                still_charging.append(ch)
                continue

            x = ch["x"]
            candidates = [p for p in gas if abs(p["x"] - x) < 2 and p["y"] <= barrier_y + 3]
            if candidates:
                chosen = random.choice(candidates)
                gas = [g for g in gas if g is not chosen]
                drifters.append(new_drifter(chosen["x"], barrier_y - 2, ch["color_index"]))
                pressure[x] *= LEAK_RESET
        charges = still_charging

        # ---- the affinity field, and whatever the black hole takes ----
        drifters, swallowed = update_drifters(drifters, t, W, barrier_y, black_hole)

        if black_hole is not None:
            black_hole["flash"] *= BH_FLASH_DECAY
            if swallowed:
                # it flares for what it takes - the only signal it gives
                black_hole["eaten"] += swallowed
                black_hole["flash"] = min(1.0, black_hole["flash"] + 0.55 * swallowed)
            elif random.random() < BH_IDLE_FLICKER:
                black_hole["flash"] = max(black_hole["flash"], 0.22)
            black_hole["age"] += 1
            if black_hole["age"] >= BH_LIFE_FRAMES:
                black_hole = None

        # ---- draw ----
        canvas.clear((0, 0, 0))

        for x in range(W):
            canvas.set_px(x, barrier_y, BARRIER_COLOR)

        for ch in charges:
            glow = CHARGE_GLOW_COLORS[ch["color_index"]]
            eased = ch["progress"] ** CHARGE_EASE
            for dx in range(-CHARGE_HALF_WIDTH, CHARGE_HALF_WIDTH + 1):
                xx = ch["x"] + dx
                if 0 <= xx < W:
                    falloff = 1.0 if dx == 0 else CHARGE_NEIGHBOR_FALLOFF
                    canvas.set_px(xx, barrier_y, lerp_color(BARRIER_COLOR, glow, eased * falloff))

        for (bx, by) in bar:
            canvas.set_px(bx, by, BAR_COLOR)

        draw_gas_vectorized(canvas, gas, t)
        draw_drifters(canvas, drifters, t)

        # drawn last so nothing spiralling in ever paints over it
        if black_hole is not None:
            canvas.set_px(
                black_hole["x"], black_hole["y"],
                lerp_color(BH_IDLE_COLOR, BH_FLASH_COLOR, black_hole["flash"]),
            )

        canvas.present(fps=60)

    canvas.quit()


if __name__ == "__main__":
    main()
