# Documentation & Sharing — working brief

**Purpose of this file:** cold-start context for a chat dedicated to documenting and
sharing the LED panel practice. Read this first, then work from it. It is meant to be
edited as things change — if a decision here turns out to be wrong, change the file.

---

## Who / where

Michael Luck Schneider (Luck) — interaction designer, artist, educator. Background in
physical computing, assistive tech, museum/exhibit work; taught at NYU ITP and City
Tech. Long history of building interactive things in physical space.

| Channel | URL | Role |
|---|---|---|
| GitHub | https://github.com/just4letters/01_LEDControl | Code, source of truth. Public. |
| YouTube | https://www.youtube.com/@just4letters | Video archive. Everything links here. |
| Instagram | https://www.instagram.com/just_4letters/ | Frequent, low-effort sketch feed. |
| WordPress | https://www.just4letters.com/projects/ | Long-form writing. Currently only holds an old portfolio grid ("Projects – Old", "New to come") — the new work wants its own section, not a retrofit of that grid. |

## What's being documented

An ongoing practice around a physical LED panel: 0.5m square, 256×256, 1.9mm pitch,
run as an extended second monitor. The work is about **natural human interaction and
language as the interface** — webcam hand/face tracking, voice, and language models
translating speech into bounded operations on a physics system. Plus a parallel track
of ambient generative "sketches" that run without any interaction at all.

Aesthetic constraint throughout: black background, individual pixels must read as
individual pixels, never scaled or antialiased.

---

## Decisions already made

### 1. Three platforms, three tempos — do not cross-post the same thing

**Instagram = the sketchbook.** Highest frequency, lowest effort. 15–45s screen
recording, posted the day it's made, one or two plain sentences of caption. No
editing, no intro, no music. Modeled on Zach Lieberman's daily sketches, which he
describes as "an intimate studio journal, a space for experimentation without the
weight of the finished artwork." Failures and half-states are *better* posts than
resolved ones — they show the medium has behavior being discovered rather than
executed.

**YouTube = the evidence.** Fewer, longer. Critically: **the physical panel in a
room, with a hand in frame.** Instagram crops and compresses; the whole point of this
work is a light object in physical space, which needs the wider frame and longer take.
Everything else links here.

**WordPress = the thinking. Irregular, NOT weekly.** Post when a piece resolves into
an idea worth a few paragraphs. Could be every two weeks, could be every two months.
A weekly cadence on the slowest platform turns into homework and produces thin posts.

**GitHub** already works — good conventional commit messages (`feat:`, `fix:`,
`refactor:`). Missing a root README, which should become a hub linking each
sketch to its video.

### 2. Register: between inspiration, art, and DIY

Not precious, not polished, but genuinely artful and clearly reproducible. The
reference point is **Anders Hoff / inconvergent.net** — see
https://inconvergent.net/2016/shepherding-random-growth/ for the canonical structure.
Secondary reference: Tyler Hobbs' essays.

The moves Hoff makes, which we're copying:

- Opens with a **natural-world hook**, not a technical one (differential growth is
  introduced via walnuts, cabbage, flower petals — you understand the phenomenon
  before any math).
- Reduces the system to **a short bulleted list of rules in plain language**. That
  list *is* the DIY. Someone could reimplement from it in any language.
- **Marks the deep end explicitly** ("this section is a little more technical") so a
  casual reader can bail at the right moment without feeling stupid.
- Gives **one or two hard-won practical tips** — the thing that proves he actually
  built it (for him: use k-d trees for neighbor lookups).
- **Ends with an invitation, not a conclusion** — "what if you extended this to
  grids?" — plus a pointer outward.
- **Links generously to others** doing related work, early in the piece.
- **Never dumps code inline.** Code is linked. The rules carry the explanation.

### 3. Series structure: instrument vs. pieces

This is the key organizing idea. The work splits into:

- **The instrument** — calibration → homography → finger-as-cursor → voice → strict
  structured intent. Shared substrate across sketches.
- **The pieces played on it** — pixel_life, ambient sketches, waterfall, face/nose
  variants.

So: **one foundation post explains the instrument properly. Every later sketch post
opens with a single sentence linking back to it and spends its words on what's
specific to that sketch.** This is exactly Hoff's own pattern — he wrote "On
Generative Algorithms" once and later posts just reference it.

### 4. Post template

1. Natural-world hook (no tech)
2. The clip — panel in the room, physical
3. The rules — 4–6 plain-language bullets, no code
4. The surprise — what the rules did that wasn't designed. This is the artful part
   and it must be honest, not curatorial.
5. Technical aside, explicitly marked as such
6. Invitation — open questions, what to try next
7. Code links

---

## Current status

**Post 01 drafted:** `writing/01-pointing-at-light.md` — covers webcam calibration
(ArUco markers displayed *by the panel itself*, homography) plus voice-driven
calibration plus Pixel Life's speech→JSON physics control. Does double duty as the
foundation post and the first sketch.

Blocking / open on that post:
- Needs a new video shot. Existing placeholder: https://youtu.be/7xv4GKCXPEo
  The needed shot is the wide one — panel on the wall, hand entering frame.
- **The series needs a name.** Currently a placeholder: "Notes from a practice — 01."
- Confirm the link-back structure before writing post 02.

**Not yet built (proposed, not agreed):**
- `JOURNAL.md` — running dated log in the repo. Habit: at the end of a work session,
  say "log this," and the entry gets drafted from the git diff and the session's
  conversation. Two minutes, not a writing exercise. This is what feeds the
  WordPress posts later so they never start cold.
- A capture script / hotkey for recording just the LED display, dumping timestamped
  files into `capture/raw/`.
- Root `README.md` as a hub.

**Candidate subjects for future posts:**
- **Ambient Sketches 03 + 04 (Affinity / Accretion) — strongest material, probably
  ONE post not two.** The arc writes itself: built a field where pixels drift
  forever and group by color, realized nothing on screen explained why anything ever
  left, fixed it by making the exit visible (a black hole that bends the wind into an
  inward spiral instead of applying a gravity force). Keepers: damping was silently
  capping everything at 53% of wind speed until the physics moved into the wind's own
  reference frame; grouping via a coarse blurred density map so cost is flat from 50
  pixels to 5000; the "stops caring" defection rule that keeps the field from
  collapsing into three tidy blobs; and the `CLUMP_GAIN` saturation trap, where
  doubling the population silently flattened the loner-vs-clump distinction into
  uniform mush. See JOURNAL 2026-08-05.
- Ambient Sketch 02 (Selective Membrane) — a membrane that only lets some pixels
  through, and now *telegraphs* it: the line rests dim white and a crossing builds
  visibly across three pixels before bursting at full saturation. Pixels gain a color
  by crossing; above the line they can only rest against their own color and bounce
  off everything else, which sorts the space into color territories nobody designed.
  Includes a genuinely interesting bug (an absorbing state where the system could
  permanently seal itself if permeability was a fixed per-particle tag instead of a
  fresh roll each attempt).
- The bass line, shared by 02/03/04 — a fuel bar that forms at the bottom, spends
  itself into gas losing exactly the pixel that just rose out of it, then *rests* on a
  bare floor. The rest is the good bit: it sends a gap climbing through the rising
  stream that widens as it goes (measured 16px → 32px), because the pixels ahead of it
  are still accelerating. It's what proves the gas comes FROM the bar.
- **Measuring instead of eyeballing** — possible short methodology piece. Repeatedly
  the measurement contradicted the obvious guess: raising wind speed made things move
  *slower*; widening the swirls was what actually helped; clustering had to be checked
  against a shuffled-color baseline to prove it wasn't just a density artifact.
- Face / nose / blink tracking as input.

---

## Working agreements

- **Draft, don't interview.** Produce a real draft to react to rather than asking a
  lot of questions first.
- **Ground everything in the actual code.** Read the source before writing about it.
  The good details are in there (e.g. the `instructions.txt` prompt file, where
  `Earth gravity is 0.1. NEVER use 9.8` is doing real aesthetic work — it's teaching
  a model the physical intuitions of an invented world).
- **The un-preciousness is the point.** Admitting what surprised or broke is more
  valuable than presenting a resolved result.
- **Don't invent facts about the work.** If a detail isn't in the code or the
  conversation, ask or leave a marked placeholder.
- Drafts live in `writing/` in the LEDControl repo and version with the code.

## Relevant files

```
LEDControl/
├── writing/
│   ├── DOCUMENTATION-BRIEF.md      <- this file
│   └── 01-pointing-at-light.md     <- post 01 draft
├── DISPLAY_LAYOUT.md               <- how the panel is driven (2nd monitor, offsets)
├── tools/finger_calibrate_voice.py <- ArUco + homography + voice calibration
├── core/vision.py                  <- HandTracker / FaceTracker, homography applied
├── core/voice.py                   <- Whisper listener + constrained-schema intent
├── visualizer/pixel_life/          <- app.py, physics.py, instructions.txt (the prompt)
└── visualizer/ambient/sketches/    <- standalone generative sketches (+ studies/, selects/)
```
