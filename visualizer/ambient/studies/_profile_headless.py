"""
Scratch profiling harness - stubs pygame, runs study_02 headless under
cProfile for a fixed number of frames (with a decent backlog of stuck
pixels already accumulated, since that's what the user says gets slow),
and prints the hottest functions by cumulative time. Not part of the
studies collection; safe to delete.
"""

import sys
import types
import cProfile
import pstats
import os as _os

STUDIES_DIR = "/sessions/relaxed-dreamy-brahmagupta/mnt/01_LEDControl/visualizer/ambient/studies"
_STOP_AFTER = int(_os.environ.get("STOP_FRAMES", "1200"))


class _FakeSurface:
    def __init__(self, size):
        self._size = size

    def get_size(self):
        return self._size

    def fill(self, *a, **k):
        pass

    def blit(self, *a, **k):
        pass


class _FakeClock:
    def tick(self, fps=60):
        return 16


_frame_counter = {"n": 0}


def _fake_event_get(*a, **k):
    _frame_counter["n"] += 1
    if _frame_counter["n"] > _STOP_AFTER:
        return [types.SimpleNamespace(type="QUIT")]
    return []


pg = types.ModuleType("pygame")
pg.QUIT = "QUIT"
pg.KEYDOWN = "KEYDOWN"
pg.K_ESCAPE = "K_ESCAPE"
pg.FULLSCREEN = "FULLSCREEN"
pg.init = lambda: None
pg.quit = lambda: None
pg.mouse = types.SimpleNamespace(set_visible=lambda v: None)
pg.display = types.SimpleNamespace(
    set_caption=lambda t: None,
    set_mode=lambda size, *a, **k: _FakeSurface(size),
    flip=lambda: None,
)
pg.Surface = _FakeSurface
pg.time = types.SimpleNamespace(Clock=_FakeClock)
pg.event = types.SimpleNamespace(get=_fake_event_get)
pg.transform = types.SimpleNamespace(scale=lambda surf, size: _FakeSurface(size))
pg.surfarray = types.SimpleNamespace(blit_array=lambda surf, arr: None)

sys.modules["pygame"] = pg
sys.path.insert(0, STUDIES_DIR)

mod_name = sys.argv[1] if len(sys.argv) > 1 else "study_02_selective_membrane"
study = __import__(mod_name)

profiler = cProfile.Profile()
profiler.enable()
study.main()
profiler.disable()

print(f"Profiled {_frame_counter['n'] - 1} simulated frames.")
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(15)
