"""
RSS feed data source - PLACEHOLDER, not yet implemented.

Intended behavior: poll a list of feed URLs, detect new items since the
last check, and expose a brief "pulse" so app.py can flash an accent
color or ripple when fresh headlines land.

--- Interface contract to implement ---
    .start()      -> begin a polling thread
    .get_state()  -> {
        "available": bool,
        "pulse": bool,           # True for a brief window right after a new item arrives
        "accent_color": (r,g,b), # color to flash, if pulse is True
        "latest_title": str | None,
    }

To wire this in for real:
    1. Add feed URLs to ambient_config.json, e.g. "rss_feeds": ["https://..."]
    2. `pip install feedparser` (into the project's real venv/ - see the
       README note in app.py about which of venv/ vs .venv/ is populated)
       and poll each feed on an interval (a thread, same pattern as
       WeatherService in weather.py).
    3. Track seen entry ids/links so only *new* items trigger a pulse.
    4. In app.py, merge get_state() into ambient_state the same way
       weather/time_of_day/motion already are.
"""


class RSSService:
    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):
        pass

    def get_state(self):
        return {
            "available": False,
            "pulse": False,
            "accent_color": (255, 255, 255),
            "latest_title": None,
        }
