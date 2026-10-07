# Display layout

How the LED wall is physically connected and how every visualizer in this
repo talks to it. Read this before writing a new `app.py` or sketch.

## The hardware

- Panel: 0.5m square, 1.9mm pixel pitch -> 256 x 256 addressable pixels.
- Connection: the panel's driver presents itself to the computer as a
  second monitor, extended (not mirrored), currently signaled at
  1920x1080.
- The panel doesn't fill that whole 1920x1080 signal - it only reads a
  256x256 window starting at some offset within it. That offset was
  measured by hand with `tools/screen_calibration.py` and is saved in
  `led_config.json`:

```json
{
    "offset_x": 0,
    "offset_y": 25,
    "width": 256,
    "height": 256
}
```

- `width` / `height` - the panel's native pixel grid.
- `offset_x` / `offset_y` - top-left corner of that grid within the
  1920x1080 signal.

If the panel is ever repositioned, re-wired, or swapped for a different
size, rerun `tools/screen_calibration.py` (or `main.py` -> "Recalibrate
Screen") rather than hand-editing this file - it regenerates
`led_config.json` from a live on-screen calibration pass.

## The pattern every visualizer follows

1. Open a normal pygame fullscreen window on display `1` (the extended
   monitor) at 1920x1080, with a fallback to the primary display if a
   second one isn't attached:

   ```python
   pygame.init()
   OS_W, OS_H = 1920, 1080
   try:
       screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN, display=1)
   except Exception:
       screen = pygame.display.set_mode((OS_W, OS_H), pygame.FULLSCREEN)
   ```

2. Draw the actual animation onto a separate `pygame.Surface` sized
   *exactly* `width` x `height` from `led_config.json` - never bigger,
   never scaled up or down. One surface pixel is one physical LED, always.

3. Each frame, blit that small surface onto the fullscreen screen at
   `(offset_x, offset_y)` and clear everything else to black:

   ```python
   screen.fill((0, 0, 0))
   screen.blit(canvas, (offset_x, offset_y))
   pygame.display.flip()
   ```

That's it - no stretching, no letterboxing, no DPI scaling. The rest of
the 1920x1080 signal outside the panel's window just stays black and is
never seen.

This is what `main.py`, `visualizer/ambient/app.py`,
`visualizer/pixel_life/app.py`, and the calibration tools all already do.
`visualizer/ambient/_pixel_canvas.py` wraps steps 1-3 into a small
`PixelCanvas` class (`set_px`, `clear`, `present`) so a new sketch is just:

```python
from _pixel_canvas import PixelCanvas

canvas = PixelCanvas(title="my sketch")
while canvas.poll():
    canvas.clear()
    canvas.set_px(x, y, (255, 255, 255))
    canvas.present()
canvas.quit()
```

## Why this matters for sketches

The whole point of this panel is that individual pixels are visible - a
lit pixel should read as one distinct point of light on a black field, not
a blurred blob. That only holds if the code never resamples, scales, or
antialiases between the logical pixel buffer and what actually ships to
the panel. Always draw at native resolution (`width` x `height`) and let
the offset blit be the only positioning step.
