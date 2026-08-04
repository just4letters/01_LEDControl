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

- **02 - Selective Membrane**: same line and gas physics as 01, but each
  crossing attempt only succeeds some of the time - a fresh coin flip
  every time, not a fixed tag on a pixel, so the system can't randomly
  drain itself of everything able to leak. Pixels that pass come out dim
  red, dim green, or dim blue; the barrier itself stays neutral gray, but
  the exact crossing point sparks a bright flash of that color and fades.
  Above the line, pixels feel a weak pull toward already-stuck pixels of
  their own color, bounce off the left/right walls, and now also bounce
  off any stuck pixel that *isn't* their color - a different color is a
  solid obstacle, not a surface. The only ways to actually come to rest
  are the very top edge or a same-color neighbor, so growth sorts into
  three separate color territories that can end up deflecting each other
  in fairly unpredictable ways.

More studies land here as we explore other approaches (cellular-style
pixel-death/birth, strict grid patterns dissolving into noise, particle
swarms with rules, etc).
