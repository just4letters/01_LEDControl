# Pointing at Light

*Notes from a practice — 01. Teaching a wall of LEDs where it is, and then talking to it.*

---

A baby points before it can talk. Pointing is one of the earliest things we do, and it only works because of something we never think about: the other person and I are in the same room, and we both know it. I point, you look. That shared space is doing all the work.

I have a half-meter LED panel on my wall. 256 by 256 pixels, 1.9mm apart — close enough that a single lit pixel still reads as a single pixel rather than dissolving into a glow. I wanted to point at it.

The problem is that the panel and the webcam watching me have no idea the other one exists.

> **[video: the panel on the wall, hand entering frame, particles scattering away from a fingertip]**

---

## Two machines, one room

The webcam sees a rectangle of light somewhere in its frame, at an angle, probably keystoned, definitely not where it thinks it is. The panel just displays whatever it's handed. Neither knows the geometry of the room they're both sitting in. Before anything else can happen — before a finger can push a particle, before "make it bigger" can mean anything — those two have to agree on where things are.

The trick is to let the panel tell the camera.

The rules are short:

- **The panel displays a grid of printed markers.** Not a photo of markers — the panel itself lights them up, dozens of small machine-readable squares (ArUco markers, the same kind robots use), each one with an ID that maps to a known position on the panel.
- **The camera looks for them.** Whichever markers it can find, it now knows the pixel coordinates of. It's reading the panel's mind, because the panel is broadcasting its own coordinate system as an image.
- **The visible markers define the bounds.** Take the leftmost and rightmost, topmost and bottommost markers the camera can see, and you've roughly found the panel in camera space.
- **Then you nudge it by hand until it's exact.** Rough is not enough at this pitch; being off by three pixels is visible.
- **Four corners give you a homography.** With four known correspondences — this point in the camera, that point on the panel — you can compute a single 3×3 matrix that converts *any* camera coordinate into a panel coordinate. That matrix is the shared space. It's the whole thing.

After that, a fingertip found in the camera image can be multiplied by that matrix and land, correctly, on a specific LED. Pointing works.

---

## Talking, because your hands are full

Here's the part I didn't plan and now can't imagine doing without.

Calibration is a physical activity. You're behind the camera, or adjusting the camera, or stepping back to check the angle, or holding something. Your hands are *in the room*, not on the keyboard. So the obvious interface — nudge these edges with the arrow keys — is exactly the wrong one. You have to keep walking back to the laptop to move a boundary by one pixel.

So calibration listens.

> *"Make the markers bigger."*
> *"Move the top down five pixels."*
> *"Looks good."*

Speech gets transcribed locally by Whisper, then handed to a language model that is **not** allowed to answer freely. It's constrained to a rigid schema — an action from a fixed list, an edge (top/bottom/left/right), a direction, an integer amount. That's the entire vocabulary. The model's only job is to turn a sentence into one of those, or to say *ignore*.

That constraint is the design. A model that can return anything is a model you have to defend against. A model that can only return `{"action": "move_edge", "edge": "top", "direction": "down", "amount": 5}` is a translator, and translators are easy to trust.

Keyboard controls still work. They're just no longer the only way in.

---

## A world you can talk to

The same idea, pushed further, is a sketch I call **Pixel Life**.

The panel holds a small physics world. Particles with velocity and a lifespan. Nodes — dots, lines, triangles, squares — that emit particles, or attract them, or bounce them off their exact perimeter. Global gravity. And your fingertip, which is not a mouse pointer but a *force*: it can attract, repel, or spray particles into the world depending on what mode it's in.

You build the world by describing it.

> *"Spawn a red dot here."*
> *"Make it shoot faster."*
> *"Now make my finger repel things."*
> *"Gravity goes up."*
> *"Save this as bees."*

Same architecture as calibration: speech in, strict JSON out, the app executes a bounded operation. Nothing improvises. The model never touches the physics loop — it only ever picks from a menu I wrote.

Two details make it feel like a conversation instead of a command line.

**It knows what's on screen.** Every request gets the current world state injected alongside it — how much gravity there is, how many nodes exist, and the full properties of whatever node is currently selected. So *"make it bigger"* is answerable: the model can see the thing is currently size 4, and knows that "bigger" from 4 means something different than "bigger" from 20. Relative language needs a referent, and the referent is the room.

**It knows the local physics.** This is my favourite file in the whole project, and it's not code:

```
GRAVITY: Pygame adds gravity every frame. Use TINY decimals.
Earth gravity is 0.1. "A little gravity" is 0.05. NEVER use 9.8.
```

That's not configuration. That's me teaching a model the physical intuitions of a world I invented — a world where earth gravity happens to be 0.1, where "a lot" of emission means fifty particles a second, where a "dot" means precisely one pixel and never a circle. The prompt file is where the aesthetics of the piece actually live. Everything downstream is just arithmetic.

---

## What surprised me

I expected the voice layer to feel like dictation — a slower keyboard. It doesn't. It feels like *describing*, which is a different mental act. You end up saying things you wouldn't have thought to build, because a sentence is cheaper than a menu. "Make it shoot faster and go blue" is one breath and two parameter changes you'd never have bothered to hunt down.

And the finger turning out to be a *force* rather than a cursor changed the whole register of it. A cursor selects. A force disturbs. You wave at a cloud of particles and they scatter, and the gesture that produces it is nothing like using a computer.

---

## Two things I got wrong first *(slightly more technical)*

**A tracked finger jitters, so selection has to be sticky.** My first version grabbed whatever node was within 20 pixels of the fingertip. It flickered constantly between neighbours, because hand tracking is noisy at the pixel level. The fix is hysteresis: it takes 20 pixels to *grab* a node, but 35 pixels to *lose* it. Easy to catch, hard to drop. Any tracked-input system needs some version of this.

**Raw tracking is unusable; smoothed tracking is fine.** The fingertip position is exponentially smoothed toward its target — each frame moves about a third of the remaining distance. And when the hand leaves the frame, the cursor doesn't vanish immediately; it waits about fifteen frames first, so a momentary detection dropout doesn't make your finger blink out of existence mid-gesture. Both numbers were found by being annoyed.

---

## Where this goes

Things I want to try: the panel calibrating itself *continuously* rather than once, so the camera can be moved mid-session. Voice that modifies the physics *rules* and not just the objects. Two people pointing at once.

If you want to build something like this, the pieces are all off-the-shelf — MediaPipe for hand landmarks, OpenCV for markers and homography, Whisper for speech, any model that supports structured output for intent, Pygame to draw. The interesting work isn't in any one of those. It's in deciding what the machine is *allowed to be asked*, and writing that down carefully.

**Code:** [`tools/finger_calibrate_voice.py`](https://github.com/just4letters/01_LEDControl/blob/main/tools/finger_calibrate_voice.py) · [`core/vision.py`](https://github.com/just4letters/01_LEDControl/blob/main/core/vision.py) · [`core/voice.py`](https://github.com/just4letters/01_LEDControl/blob/main/core/voice.py) · [`visualizer/pixel_life/`](https://github.com/just4letters/01_LEDControl/tree/main/visualizer/pixel_life)

Everything is in [01_LEDControl](https://github.com/just4letters/01_LEDControl). It is not tidy.
