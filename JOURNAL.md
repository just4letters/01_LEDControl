# Journal

Running dated log, drafted from the git diff and the session at "log this" time.
Two minutes, not a writing exercise. This is what post drafts get pulled from —
see `writing/DOCUMENTATION-BRIEF.md` and `writing/RHYTHM.md`.

---

## 2026-08-05

**Commit:** [`23d81dd`](https://github.com/just4letters/01_LEDControl/commit/23d81dd) — "implement affinity study and hub page while updating calibration configs"

Reworked Study 02 (Selective Membrane)'s crossing mechanic: it now telegraphs —
the barrier line rests dim white, and a crossing about to happen builds visibly
across three pixels before bursting at full saturation, capped to a few
concurrent charges so the signal stays legible. Paired it with a strict
timekeeper: a fuel bar forms across the bottom every few seconds and dissolves
into gas, losing exactly the pixel that just rose out of it.

Built Study 03 (Affinity) on the same bass line — same bar, same gas, same
telegraphed membrane — but the top half no longer accumulates. Crossed pixels
drift permanently in a slow swirling wind, pulled toward their own color and
pushed from the others via a coarse blurred density map (not pixel-to-pixel,
so cost stays flat from 50 pixels to 5000). Groups hold together in the wind's
frame, rotate as they travel, merge, slide off each other, and shed stragglers
— a pixel periodically stops caring about color entirely and drifts on
momentum alone, which is what keeps the field from collapsing into three tidy
blobs. Where 02 reads like an anthill in reverse, this reads like weather.

Also recalibrated both homography configs (finger and nose) — new corner
correspondences in `finger_config.json` and `nose_config.json`.

### Later the same session

Gave the bar a **rest**: it now spends itself over 110 frames and then the floor
sits bare for 55 before the next one snaps in. That silence is the whole point —
it puts a gap in the rising stream that climbs the full lower half, *widening*
as it goes because the pixels ahead of it are still accelerating toward terminal
velocity. Measured it climbing from 16px to 32px over one cycle. It's what
proves to a viewer that the gas comes FROM the bar rather than just appearing.
Applied to both 02 and 03 since the bass line is shared.

Chased more pixels in 03's upper field and found the ceiling wasn't where I
assumed: the charge system was already running at 82% of its theoretical
throughput, so lifespan was the only lever that adds pixels without making the
membrane busier. Doubled it. The non-obvious catch — `CLUMP_GAIN` converts local
density into the cohesion "mass" term and it's capped, so doubling the population
silently pinned ~68% of pixels at the cap and flattened the whole
loner-vs-knot distinction into uniform mush. Halving the gain restored it.
Population-dependent constants like that are a trap worth writing about.

Built **Study 04 (Accretion)**, which is 03 with a reason to disappear. In 03
pixels just timed out — it balanced the books but nothing on screen explained
it. Now they have no lifespan at all and the only exit is a black hole: every
sixth bar, one opens at a random spot up top. It's a single almost-always-dark
pixel, and rather than applying a gravity force it **replaces the weather** —
inside its reach the prevailing wind is blended into an inward spiral, so
anything near it stops being carried and starts falling. Did it that way
because a gravity force competes with the wind and pixels get half-pulled then
rescued; bending the flow means anything committed stays committed. It flares
only when it actually swallows something, so every flicker is one pixel gone.

Population is now a true equilibrium rather than a set number. Verified it's
stable from both directions — seeded at 600 it climbed to ~775, seeded at 1400
it fell to ~1214, converging near 1000. But it *wanders*, and that's inherent:
a hole opening inside a clump takes 90 pixels, one opening in a gap takes 2.
Logged that full range.

**Worth pulling into a post:** Studies 03 and 04 together are the strongest
material yet, and they're arguably one post rather than two — the arc is "I
built a system where things accumulate forever, realized nothing explained why
anything left, and fixed it by making the exit visible." Specific things worth
keeping: the frame-relative wind (damping was quietly capping everything at 53%
of flow speed until the physics moved into the wind's own reference frame), the
coarse density-map trick for flat-cost grouping, the "stops caring" defection
rule, the black hole bending flow instead of pulling, and the `CLUMP_GAIN`
saturation trap. Same shape as the Selective Membrane absorbing-state bug
already on the candidates list: here's the rule, here's what it did that I
didn't design.

Also a recurring methodology point that might deserve its own short piece: most
of tonight's decisions were settled by *measuring* rather than eyeballing —
wind speed, swirl wavelength, whether clustering was real or just a density
artifact (checked against a shuffled-color baseline), whether the population
equilibrium was stable. Several times the measurement contradicted the obvious
guess: raising wind speed made things move *slower*, and widening the swirls
was what actually helped.

**Open:** no video shot of any study yet. Post 01 is still blocked on the wide
in-room shot — same blocker, unaddressed today. Studies 03/04 would both want
a screen capture rather than an in-room take.

---
