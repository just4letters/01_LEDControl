# Cycle — output tiers for this palette

This repo is one palette in **Palettes of the Periphery (POP)** — addressable
light (0.5m / 256×256 / 1.9mm pitch) × hand, face and voice tracking.

POP is my personal creative exploration of the same intersection
[The Periphery Center](https://www.peripherycenter.com/) is built around:
technology, built environment and cultural ritual. **POP does not run the
Culture Cycle** — there's no cohort, no guides, no Exchanges. It borrows the
vocabulary so the two don't drift apart: a *palette* is a medium plus an
interaction layer, and output moves through the same three tiers.

| Tier | Gate | Lives in |
|---|---|---|
| **Sketch** | None. It runs. Ugly is fine, abandoned is fine — volume is the point. | `visualizer/ambient/sketches/` |
| **Study** | It met someone who isn't me and survived it — went out publicly, got a response, and I still believed in it afterward. | `visualizer/ambient/studies/` |
| **Select** | It earns a slot in what the panel plays when nothing else is running. | `visualizer/ambient/selects/` |

Promotion is a `git mv` plus one line in the log below. The shared canvas
harness sits at `visualizer/ambient/_pixel_canvas.py` and is resolved relative
to each piece's parent directory, so moving a file between tiers never breaks
its imports.

A tier is not a quality score. A sketch can be better than a study; it just
hasn't been out of the room yet.

---

## Current state

**Sketches** (4)

- `sketch_01_barrier_drift` — a rigid line, buoyant gas pressing from below,
  occasional leaks that flutter up and stick to the edges.
- `sketch_02_selective_membrane` — crossings only sometimes succeed and are
  telegraphed before they happen; a pixel gains a color identity by crossing,
  and above the line can only rest against its own kind.
- `sketch_03_affinity` — nothing sticks. Crossed pixels drift permanently in a
  slow wind, grouping by color, clumps rotating as they travel.
- `sketch_04_accretion` — 03 with a reason to disappear: a black hole that
  bends the wind into an inward spiral rather than pulling.

**Studies** (0) · **Selects** (0)

---

## Promotion log

Newest first. One line each: date, piece, tier it moved to, and what gated it.

_Nothing promoted yet._
