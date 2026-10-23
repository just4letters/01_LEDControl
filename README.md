# LEDControl — the light palette

A 0.5 m LED panel on the wall, 256×256 pixels at 1.9 mm pitch, run as a second
monitor. A webcam watches your hand, face or voice; the panel answers. Black
background always, and a lit pixel is meant to read as one pixel, never a blur.

This is the **light palette** of Palettes of the Periphery (POP): addressable
light × hand / face / voice. What POP is, and what the other palettes taught
this one, lives in [`../00_POP/PRACTICE.md`](../00_POP/PRACTICE.md). Output
tiers (Sketch → Study → Select) and the promotion log are in
[`CYCLE.md`](CYCLE.md). Nothing here has been promoted past Sketch.

(The GitHub remote is still named `01_LEDControl`; the numeric prefix was
dropped locally when the repos were grouped under `POP/`.)

---

## Run it

**You need:** a Mac or similar with Python 3.14 (what this has been run on), a
webcam, a microphone, and — for pixel-accurate output — the panel connected as
an *extended* display. Without the panel, everything falls back to full-screen
on your main display.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # slow the first time: Whisper pulls in torch
```

If you have an older `venv/` from before the folder was moved, delete it and
recreate it — its scripts still point at `~/Documents/01_LEDControl/venv`.

**Microphone.** `core/mic.py` records through PyAudio if it is installed and
through `sounddevice` if not, so a machine without Homebrew/PortAudio (an Intel
Mac, say) still works. `python tools/doctor.py` says which one it found.

**Keys.** Create `.env` in this folder (it is gitignored):

```
GEMINI_API_KEY=...          # voice → intent, used by every app that listens
RUNWAYML_API_SECRET=...     # only for visualizer/pixel_life_runway
```

Most apps build a Gemini client when they start, so set the key before
launching anything except the sketches.

**First time on a new panel or machine — calibrate before anything else.**
`led_config.json` (where the 256×256 window sits inside the 1920×1080 signal),
`finger_config.json` and `nose_config.json` are all gitignored, so a fresh clone
has none of them. Run `python tools/doctor.py` to see what's missing; the
launcher also notices a missing `led_config.json` and sends you to the
screen-calibration tool on its own.

```bash
python main.py
```

The launcher draws its menu on the panel. Pick by keyboard (`1`–`9`) or by
voice (say the name or number):

| # | App | What it is |
|---|-----|------------|
| 1 | Black Hole Sandbox (`pixel_life_face`) | Nose-pointing, blink to drop shapes, jaw opens the event horizon |
| 2 | Ambient (`ambient`) | A slow light field driven by weather, time of day and room motion; voice nudges palette / energy / brightness |
| 3 | Pixel Life (`pixel_life`) | A physics canvas you build by describing it out loud; your finger is the cursor |
| 4 | Pixel Life Runway (`pixel_life_runway`) | Pixel Life plus Runway video generation |
| 5 | Waterfall (`waterfall`) | Earlier piece, finger-driven |
| 6 | Waterfall Evolve (`waterfall_evolve`) | Waterfall variant with an `evolved_rules.txt` |

Numbering is the launcher's own: it lists folders under `visualizer/` that
contain an `app.py`, alphabetically, with the Black Hole Sandbox pinned first.
Add a folder and the numbers shift.

Other launcher commands: `S` / `N` / `F` (or say "recalibrate screen / nose /
finger"), `Esc` or `Q` to quit, or say "quit" / "shut down". "Clear screen"
works inside any app that uses the shared voice listener.

macOS will ask for camera and microphone permission for your terminal the
first time. Whisper downloads its `base.en` model on first listen.

### Setting up a second machine

*On an Intel Mac, use `requirements-intel-mac.txt` instead of `requirements.txt`
below: PyTorch has no newer Intel-Mac builds, which forces older NumPy and
OpenCV. Its header explains what was and wasn't verified.*

Git carries the code; it does not carry anything that depends on the room.
Each machine owns its own calibration, venv and keys.

1. Install Python 3.14.
2. `git clone https://github.com/just4letters/01_LEDControl.git LEDControl`
   (the remote still has the old name), then `cd LEDControl`.
3. `python3.14 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
   Slow the first time: Whisper pulls in torch (~2 GB).
4. Create `.env` with `GEMINI_API_KEY` (and `RUNWAYML_API_SECRET` if you use the
   runway app).
5. `python tools/doctor.py`. Expect failures for `led_config.json` and
   `finger_config.json` on a fresh clone; that is the next step, not a bug.
6. Calibrate in order: screen, finger, nose (below). Grant camera and mic
   permission to your terminal when macOS asks.
7. Check `FINGER_CAMERA_INDEX` / `FACE_CAMERA_INDEX` in `core/config.py`.
   macOS can number cameras differently on each machine; doctor lists what it
   finds at each index.
8. `python main.py`. The first spoken command downloads Whisper's `base.en`.

Calibration files never sync, so a `git pull` on either machine will not
disturb the other's.

### Calibration, in order

Calibration is an interaction, not a settings file — you do it by looking at
the panel and talking to it, because your hands are busy holding the camera.

1. **Screen** (`tools/screen_calibration.py`) → `led_config.json`. Finds the
   panel's grid inside the display signal. See [`DISPLAY_LAYOUT.md`](DISPLAY_LAYOUT.md).
2. **Finger** (`tools/finger_calibrate_voice.py`) → `finger_config.json`. The
   panel shows ArUco markers; the camera reads them; a homography maps camera
   space to panel space.
3. **Nose** (`tools/nose_calibration.py`) → `nose_config.json`. Point your nose
   at the four corners; same idea for head pose. The launcher only insists on
   it for the Black Hole Sandbox.

### Just the sketches

The four ambient sketches need no camera, mic, key or calibration — only
`numpy` and `pygame-ce`:

```bash
cd visualizer/ambient/sketches
python sketch_04_accretion.py          # Esc to quit
```

They resolve `led_config.json` and the shared canvas themselves, so a sketch
can move between tiers with a plain `git mv`. What each one does is in
[`visualizer/ambient/sketches/README.md`](visualizer/ambient/sketches/README.md).

---

## What's where

```
main.py          launcher: menu on the panel, voice + keyboard, pre-flight config checks
core/            shared by every app
  config.py        paths, camera indices, mic threshold
  vision.py        HandTracker / FaceTracker (MediaPipe + the homographies)
  voice.py         VoiceListener: Whisper → fast-path parser → Gemini → queue
tools/           the three calibration tools
visualizer/      the pieces (see the table above)
  ambient/         also holds _pixel_canvas.py, patterns.py, data_sources/, and
                   the sketches/ studies/ selects/ tiers
Demos/           MediaPipe reference demos (body, face, hands, object)
models/          MediaPipe .task files (tracked)
writing/         documentation strategy and post drafts
CYCLE.md         tiers, gates, promotion log
DISPLAY_LAYOUT.md  how the panel is driven — read before writing a new app
JOURNAL.md       dated work log
```

Config files in the root. All of them are per-machine: every camera position,
panel and room is different, so none of them is shared through git.

| File | Written by | In git? |
|------|-----------|---------|
| `led_config.json` | screen calibration | no — per panel/machine |
| `finger_config.json` | finger calibration | no — per camera position |
| `nose_config.json` | nose calibration | no — per camera position |
| `.env` | you | no |

`python tools/doctor.py` checks that all of these exist and look sane, along
with packages, models, displays, cameras and mic. Run it first on any new
machine, and again whenever something that was working stops.

------|-----------|---------|
| `led_config.json` | screen calibration | no — per panel/machine |
| `finger_config.json` | finger calibration | yes |
| `nose_config.json` | nose calibration | yes, although it is also listed in `.gitignore` (it was tracked first) — so it shows as modified every time you recalibrate |
| `.env` | you | no |

---

## The technical part

*A little more technical from here; the above is enough to get it running.*

**Display.** Every app opens a fullscreen pygame window on display 1, draws
into a `Surface` of exactly `width × height` from `led_config.json`, and blits
it at `(offset_x, offset_y)` over black. Never scaled. The full pattern is in
[`DISPLAY_LAYOUT.md`](DISPLAY_LAYOUT.md); `PixelCanvas` in
`visualizer/ambient/_pixel_canvas.py` wraps it for new sketches. The code
imports `pygame` but the package is **`pygame-ce`** — same namespace, different
project; don't install both.

**Voice.** `core/voice.py` runs a listener thread: `speech_recognition` records,
local Whisper (`base.en`) transcribes, then in order — universal kill words,
"clear screen", the app's own Python fast-path parser, and only then Gemini with
the app's prompt file and a response schema, so the model translates rather
than authors. Anything unresolved goes to a sentence queue for the app's own
workers. `MIC_ENERGY_THRESHOLD` in `core/config.py` is fixed (dynamic
adjustment is off); tune it to the room. The prompt files
(`visualizer/*/instructions.txt`) carry the aesthetics as much as the schema does.

**Vision.** `core/vision.py` smooths with an exponential moving average
(α = 0.35) and waits 15 frames of no detection before declaring a hand or face
lost. The same constant turned up independently in the arm palette.

**Cameras.** `core/config.py`: `FINGER_CAMERA_INDEX = 0`, `FACE_CAMERA_INDEX = 1`
(a second webcam). A tracker that can't get its camera falls back to index 0,
and a small registry stops two trackers grabbing the same one.

**Models.** Hand, face and pose `.task` files are in `models/`; the `Demos/`
scripts download any that are missing. Gemini model names are hard-coded per
file as fallback lists (`grep -rn fallback_models`) and have drifted apart —
`core/voice.py` and the apps don't list the same ones.

**Can this machine run it?** `python tools/bench.py` measures hand-tracking
frame rate, Whisper time per phrase, and both at once (the case that hurts on a
dual-core machine). Each section skips itself with a reason if a package won't
install, so it still reports something on a machine where the full
requirements can't be met. `--no-camera` and `--mic` are the useful flags.

**Headless smoke test.** `visualizer/ambient/sketches/_smoke_headless.py` runs
each sketch's real `main()` with no window (SDL's dummy video driver) for a few
thousand frames and checks that nothing raises and something gets lit. It
exits 1 on failure. Run it after any refactor, or on a new machine:

```bash
python visualizer/ambient/sketches/_smoke_headless.py                  # all sketches
python visualizer/ambient/sketches/_smoke_headless.py sketch_03_affinity --profile
```

Frame times exclude drawing to a real display, so they measure the simulation
alone: good for comparing sketches and machines, not a promise about the panel.

---

## Known rough edges

- `models/vosk-model-small-en-us-0.15/` is a leftover from an earlier speech
  approach. Nothing imports Vosk; it is gitignored.
- `visualizer/pixel_life_face/` has its own `requirements.txt` and `README.md`
  that predate the move to shared `tools/` and `main.py`. The requirements
  there list `pygame`, which conflicts with `pygame-ce`, and the README refers
  to scripts that are no longer in that folder. Use the root `requirements.txt`.
- `visualizer/waterfall_evolve/temp_frames/` holds committed scratch images.
