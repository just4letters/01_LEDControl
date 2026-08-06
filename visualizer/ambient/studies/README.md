# Ambient studies

Standalone sketches for prototyping one visual idea at a time on the
1.9mm / 0.5m panel (256x256, see `led_config.json`), before anything gets
folded into `patterns.py`. No camera, no voice, no AI - just run and watch.

Black background always; the point is that a single lit pixel should read
clearly as one pixel, not blur into a shape. `_pixel_canvas.py` renders with
hard-edged (non-antialiased) scaling to preserve that on a laptop preview.

The LED panel is a real second monitor (extended, not mirrored), driven at
1920x1080. `led_config.json` (produced by `tools/screen_calibration.py`)
says where the visible grid sits inside that signal: `width`/`height` is
the pixel grid (256x256), `offset_x`/`offset_y` is the top-left corner of
that grid within the 1920x1080 frame. Every study opens fullscreen on
display 1 and blits a Surface sized exactly `width` x `height` at
(`offset_x`, `offset_y`) - the same setup every other visualizer in this
repo already uses (see main.py, ambient/app.py). Nothing is ever scaled:
one buffer pixel is one physical LED.

Run any study directly:

    python study_01_barrier_drift.py

Esc or close the window to quit. Requires the LED monitor to be connected
as an extended display - if only one monitor is attached, it falls back to
fullscreen on that one instead (same fallback every app.py here uses), so
you'll see it full-screened on your laptop rather than pixel-accurate.

## Studies

- **01 - Barrier Drift**: a rigid 1px horizontal line at mid-height with
  loose pixels drifting and bumping against it from below, occasionally
  leaking through under pressure, then fluttering above until they stick to
  an edge. Geometric constraint, fluid behavior - a first pass at that
  "starts structured, ends fluid" arc.

- **02 - Selective Membrane**: same line as 01, but the two halves are
  built to feel like different kinds of thing. Below it the gas has no
  horizontal freedom at all - each pixel is locked to the column it was
  born in and can only rise and press - and it keeps strict time: every
  three seconds a solid bar forms across the bottom of the screen and
  spends itself into gas, losing exactly the pixel that just rose out of
  it. Then it rests - three quarters of a second where the floor is bare
  and nothing at all is born - before the next bar snaps in whole. That
  silence is what makes the rhythm legible: it puts a gap in the rising
  stream that climbs the whole lower half, widening as it goes because
  the pixels ahead of it are still picking up speed, and it's the thing
  that proves the gas is coming FROM the bar rather than just
  appearing. Above the line, nothing keeps time at all. And each
  crossing attempt only succeeds some of the time - a fresh coin flip
  every time, not a fixed tag on a pixel, so the system can't randomly
  drain itself of everything able to leak. Pixels that pass come out dim
  red, dim green, or dim blue, and every crossing is telegraphed before
  it happens: the line rests dim white, and where something is about to
  break through, three pixels glow brighter and brighter in the color
  that's coming, bursting only at full saturation. Just a few can build
  at once - a signal that fires constantly stops being a signal, so the
  scarcity matters as much as the glow. Gas pressing against the
  underside of the line dims slowly
  the longer it sits there without getting through, so it reads as a
  gradual, uneven glow that builds and fades in patches rather than a
  static second row of light - it only brightens back up by leaking
  through or giving up and sinking back into the open gas below. On
  launch, the gas population itself starts empty and bubbles up from the
  very bottom of the screen a few pixels at a time, filling in toward
  full density over several seconds instead of the whole population
  existing (and drifting upward together) from the first frame. Above
  the line, pixels feel a weak pull toward already-stuck pixels of their
  own color, bounce off the left/right walls, and now also bounce off
  any stuck pixel that *isn't* their color - a different color is a
  solid obstacle, not a surface. The only ways to actually come to rest
  are the very top edge or a same-color neighbor, so growth sorts into
  three separate color territories that can end up deflecting each other
  in fairly unpredictable ways.

- **03 - Affinity**: same bar, same gas, same telegraphed membrane as 02,
  but the top half never accumulates. Nothing sticks to anything, ever.
  A pixel that crosses just drifts forever, blown around by a wide, slow,
  swirling wind, and the only thing organizing the field is a preference -
  drawn toward wherever there's more of its own color, pushed away from
  concentrations of the other two, plus that same pull turned ninety
  degrees so a gathering doesn't only contract, it turns. Both scale with
  how much of your own kind is nearby: a lone pixel is barely affected, a
  dense knot holds itself hard and visibly rotates as the weather carries
  it. The affinity is computed in the wind's own frame of reference, so a
  group travels intact instead of being sheared apart.
  Groups form, drift as groups, deform, slide off each other, merge
  and shed stragglers, and none of it is fixed: every so often a pixel
  stops caring entirely, coasts off on its own momentum for a few
  seconds, and only later re-develops a preference. That defection is
  what keeps it from settling into three tidy blobs. Where 02 reads like
  an anthill in reverse, this one reads like weather. The affinity is
  computed from a coarse blurred density map per color rather than
  pixel-to-pixel, so it costs the same at fifty pixels or five thousand,
  and it groups more fluidly - clusters have soft edges and get pulled
  toward each other as wholes.

- **04 - Accretion**: 03 with a reason to disappear. In 03 the drifting
  pixels simply timed out - it balanced the books but nothing on screen
  explained it. Here they have no lifespan at all, and the only way out
  is a black hole: every sixth bar, one opens at a random spot up top.
  It's a single pixel, almost always dark, and what it does is bend the
  wind - inside its reach the prevailing flow is replaced by an inward
  spiral, so anything near it stops being carried by the weather and
  starts falling instead. Pixels wind in, accelerating, and vanish. It
  flares each time it takes one, which is the only thing that gives it
  away, then closes after a few seconds until the next opens elsewhere.
  Population becomes a real equilibrium rather than a set number, and it
  drifts: a hole that opens inside a clump takes ninety pixels, one that
  opens in a gap takes two, so the field visibly fills up and gets
  thinned again over minutes.

More studies land here as we explore other approaches (cellular-style
pixel-death/birth, strict grid patterns dissolving into noise, particle
swarms with rules, etc).
