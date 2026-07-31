"""
Calendar / events data source - PLACEHOLDER, not yet implemented.

Intended behavior: check for upcoming or in-progress calendar events (e.g.
via Google Calendar API or an ICS feed URL) and let app.py cue a distinct
color/pulse shortly before something starts.

--- Interface contract to implement ---
    .start()      -> begin a polling thread
    .get_state()  -> {
        "available": bool,
        "upcoming_soon": bool,   # True if something starts within the next N minutes
        "in_progress": bool,
        "accent_color": (r,g,b),
        "event_title": str | None,
    }

To wire this in for real, follow the WeatherService pattern in weather.py:
a background thread refreshing on an interval, a thread-safe get_state(),
then merge the result into ambient_state inside app.py.
"""


class EventsService:
    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):
        pass

    def get_state(self):
        return {
            "available": False,
            "upcoming_soon": False,
            "in_progress": False,
            "accent_color": (255, 255, 255),
            "event_title": None,
        }
