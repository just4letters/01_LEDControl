"""
Time-of-day data source for the ambient visualizer.
No external API needed - just blends a small set of color/brightness
keyframes across the 24-hour clock (deep night -> dawn -> day -> dusk -> night).

--- Data source interface contract ---
    .start()        -> no-op (kept for symmetry with other sources)
    .get_state()    -> computed fresh each call, always safe

get_state() returns:
    {
        "tint_color": (r, g, b),     # warm/cool tint for the current time of day
        "brightness_factor": float,  # 0-1 multiplier (dim at night, bright midday)
    }
"""

import datetime

# (hour, color, brightness) keyframes, wrapping at 24 -> 0
_KEYFRAMES = [
    (0.0, (5, 10, 40), 0.20),
    (5.0, (10, 15, 50), 0.25),
    (6.5, (255, 150, 80), 0.50),
    (8.0, (255, 235, 200), 0.90),
    (12.0, (255, 255, 255), 1.00),
    (17.0, (255, 220, 180), 0.90),
    (19.0, (255, 120, 80), 0.55),
    (21.0, (20, 20, 60), 0.30),
    (24.0, (5, 10, 40), 0.20),
]


def _lerp(a, b, t):
    return a + (b - a) * t


def get_time_of_day_state(now=None):
    now = now or datetime.datetime.now()
    hour = now.hour + now.minute / 60.0 + now.second / 3600.0

    for (h0, c0, b0), (h1, c1, b1) in zip(_KEYFRAMES, _KEYFRAMES[1:]):
        if h0 <= hour <= h1:
            t = (hour - h0) / (h1 - h0) if h1 != h0 else 0
            color = tuple(int(_lerp(c0[i], c1[i], t)) for i in range(3))
            brightness = _lerp(b0, b1, t)
            return {"tint_color": color, "brightness_factor": round(brightness, 3)}

    return {"tint_color": (255, 255, 255), "brightness_factor": 1.0}


class TimeOfDayService:
    """Thin wrapper so app.py can treat every data source the same way."""

    def start(self):
        pass

    def get_state(self):
        return get_time_of_day_state()
