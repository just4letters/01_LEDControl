"""
Headless smoke test and profiler for the ambient sketches.

Runs each sketch's real main() with no window: SDL's "dummy" video driver, and
the canvas's present()/poll() replaced so the loop runs flat out for N frames.
Checks that nothing raises and that something actually gets lit.

    python _smoke_headless.py                     # every sketch_*.py, 1500 frames each
    python _smoke_headless.py sketch_04_accretion # just one
    python _smoke_headless.py --frames 6000       # longer soak
    python _smoke_headless.py sketch_03_affinity --profile   # hottest functions

Works from any tier folder (sketches/, studies/, selects/): it runs whatever
sketch_*.py sit next to it, so after a `git mv` this file travels with the
folder or you can run it from where it is with --dir.

Exit code is 1 if any sketch fails, so it can gate a commit.
Frame times here exclude drawing to a real display, so treat them as the
cost of the simulation alone -- useful for comparing sketches and machines.
"""
import argparse
import cProfile
import importlib.util
import os
import pstats
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="sketch module names (default: all sketch_*.py)")
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--profile", action="store_true", help="cProfile each sketch, print top functions")
    ap.add_argument("--dir", default=HERE, help="folder holding the sketches (default: this file's folder)")
    args = ap.parse_args()

    sketches_dir = os.path.abspath(args.dir)
    ambient_dir = os.path.dirname(sketches_dir)       # holds _pixel_canvas.py
    sys.path.insert(0, ambient_dir)
    sys.path.insert(0, sketches_dir)

    import _pixel_canvas as pc

    state = {"n": 0, "lit_peak": 0}

    def fake_present(self, fps=60):
        state["n"] += 1
        # Counting lit pixels costs real time, so sample it; it only has to
        # prove the sketch draws something, not track every frame.
        if state["n"] % 30 == 0:
            state["lit_peak"] = max(state["lit_peak"], int((self.buffer.max(axis=2) > 0).sum()))

    def fake_poll(self):
        return state["n"] < args.frames

    pc.PixelCanvas.present = fake_present
    pc.PixelCanvas.poll = fake_poll

    found = sorted(f[:-3] for f in os.listdir(sketches_dir) if f.startswith("sketch_") and f.endswith(".py"))
    names = [n[:-3] if n.endswith(".py") else n for n in args.names] or found
    if not names:
        print(f"no sketch_*.py in {sketches_dir}")
        return 1

    print(f"led_config: {pc.LED_CONFIG_PATH}")
    print(f"{args.frames} frames each, dummy display\n")
    failed = False
    for name in names:
        path = os.path.join(sketches_dir, name + ".py")
        state.update(n=0, lit_peak=0)
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            prof = cProfile.Profile() if args.profile else None
            t0 = time.perf_counter()
            if prof:
                prof.enable()
            mod.main()
            if prof:
                prof.disable()
            dt = time.perf_counter() - t0
            ok = state["n"] >= args.frames and state["lit_peak"] > 0
            failed |= not ok
            why = "" if ok else ("  <-- never lit a pixel" if state["lit_peak"] == 0 else "  <-- stopped early")
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {state['n']} frames, "
                  f"peak {state['lit_peak']} lit px, {dt / max(state['n'], 1) * 1000:.2f} ms/frame{why}")
            if prof:
                pstats.Stats(prof).sort_stats("cumulative").print_stats(15)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
