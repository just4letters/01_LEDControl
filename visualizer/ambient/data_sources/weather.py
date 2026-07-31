"""
Weather data source for the ambient visualizer.
Uses Open-Meteo (https://open-meteo.com) - free, no API key required.

--- Data source interface contract ---
Every module in data_sources/ exposes a "Service" class with:
    .start()        -> begins a background daemon thread (safe to call once)
    .get_state()    -> thread-safe dict snapshot, ALWAYS returns something
                       sane even before the first successful fetch.
This lets app.py merge multiple sources every frame without blocking on
network calls or worrying about partial failures.

WeatherService.get_state() returns:
    {
        "available": bool,
        "location_name": str | None,
        "temperature_c": float | None,
        "weather_code": int | None,
        "cloud_cover": float | None,   # 0-100
        "is_day": bool,
        "base_color": (r, g, b),       # dim ambient hue for current conditions
        "accent_color": (r, g, b),     # brighter peak hue for current conditions
        "brightness": float,           # 0-1 multiplier (storms/overcast dim things down)
    }
"""

import json
import os
import threading
import time

import requests

AMBIENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(AMBIENT_DIR, "ambient_config.json")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes -> broad category buckets
WEATHER_PALETTES = {
    "clear":   {"day":   {"base": (40, 80, 160),  "accent": (255, 200, 120), "brightness": 1.0},
                "night": {"base": (5, 10, 40),    "accent": (80, 100, 180),  "brightness": 0.35}},
    "cloudy":  {"day":   {"base": (90, 100, 110), "accent": (180, 190, 200), "brightness": 0.7},
                "night": {"base": (20, 25, 35),   "accent": (60, 70, 90),    "brightness": 0.25}},
    "fog":     {"day":   {"base": (120, 120, 120),"accent": (200, 200, 200), "brightness": 0.5},
                "night": {"base": (30, 30, 35),   "accent": (90, 90, 100),   "brightness": 0.2}},
    "drizzle": {"day":   {"base": (60, 90, 120),  "accent": (140, 180, 210), "brightness": 0.6},
                "night": {"base": (10, 20, 40),   "accent": (60, 90, 130),   "brightness": 0.22}},
    "rain":    {"day":   {"base": (40, 70, 110),  "accent": (100, 150, 200), "brightness": 0.55},
                "night": {"base": (8, 15, 35),    "accent": (50, 80, 120),   "brightness": 0.2}},
    "snow":    {"day":   {"base": (150, 160, 180),"accent": (255, 255, 255), "brightness": 0.85},
                "night": {"base": (40, 45, 70),   "accent": (180, 190, 220), "brightness": 0.3}},
    "storm":   {"day":   {"base": (50, 40, 70),   "accent": (200, 80, 255),  "brightness": 0.5},
                "night": {"base": (15, 10, 30),   "accent": (140, 60, 200),  "brightness": 0.25}},
}


def classify_weather_code(code):
    if code == 0:
        return "clear"
    if code in (1, 2, 3):
        return "cloudy"
    if code in (45, 48):
        return "fog"
    if code in (51, 53, 55, 56, 57):
        return "drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "storm"
    return "cloudy"


class WeatherService:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.lock = threading.Lock()
        self.running = False
        self.location_name = None
        self.lat = None
        self.lon = None
        self.refresh_seconds = 900
        self._state = self._default_state()
        self._load_config()

    def _default_state(self):
        return {
            "available": False,
            "location_name": None,
            "temperature_c": None,
            "weather_code": None,
            "cloud_cover": None,
            "is_day": True,
            "base_color": (10, 15, 40),
            "accent_color": (60, 90, 150),
            "brightness": 0.3,
        }

    def _load_config(self):
        try:
            with open(self.config_path, "r") as f:
                cfg = json.load(f)
            self.location_name = cfg.get("location_name")
            self.lat = cfg.get("latitude")
            self.lon = cfg.get("longitude")
            self.refresh_seconds = cfg.get("weather_refresh_seconds", 900)
        except Exception:
            pass

    def _save_config(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump({
                    "location_name": self.location_name,
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "weather_refresh_seconds": self.refresh_seconds,
                }, f, indent=4)
        except Exception as e:
            print(f"[weather] Could not save config: {e}")

    def set_location(self, query):
        """Geocode a free-text location (e.g. 'Austin, TX') via voice command
        and sync future weather fetches to it. Returns (success, resolved_name)."""
        try:
            resp = requests.get(
                GEOCODE_URL,
                params={"name": query, "count": 1, "language": "en", "format": "json"},
                timeout=8,
            )
            resp.raise_for_status()
            results = resp.json().get("results")
            if not results:
                return False, None
            top = results[0]
            self.lat = top["latitude"]
            self.lon = top["longitude"]
            region = top.get("admin1")
            country = top.get("country")
            label_parts = [top["name"]]
            if region and region != top["name"]:
                label_parts.append(region)
            if country:
                label_parts.append(country)
            self.location_name = ", ".join(label_parts)
            self._save_config()
            self._fetch_now()
            return True, self.location_name
        except Exception as e:
            print(f"[weather] Geocode failed for '{query}': {e}")
            return False, None

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def get_state(self):
        with self.lock:
            return dict(self._state)

    def _loop(self):
        while self.running:
            if self.lat is not None and self.lon is not None:
                self._fetch_now()
            time.sleep(max(60, self.refresh_seconds))

    def _fetch_now(self):
        if self.lat is None or self.lon is None:
            return
        try:
            resp = requests.get(
                FORECAST_URL,
                params={
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "current": "temperature_2m,cloud_cover,weather_code,is_day,wind_speed_10m",
                    "timezone": "auto",
                },
                timeout=8,
            )
            resp.raise_for_status()
            current = resp.json().get("current", {})
            code = current.get("weather_code", 0)
            is_day = bool(current.get("is_day", 1))
            cloud = current.get("cloud_cover", 0) or 0

            category = classify_weather_code(code)
            palette = WEATHER_PALETTES[category]["day" if is_day else "night"]
            brightness = palette["brightness"] * max(0.3, 1 - 0.3 * (cloud / 100))

            with self.lock:
                self._state = {
                    "available": True,
                    "location_name": self.location_name,
                    "temperature_c": current.get("temperature_2m"),
                    "weather_code": code,
                    "cloud_cover": cloud,
                    "is_day": is_day,
                    "base_color": palette["base"],
                    "accent_color": palette["accent"],
                    "brightness": round(brightness, 3),
                }
            print(f"[weather] Updated: {category} ({'day' if is_day else 'night'}), "
                  f"{current.get('temperature_2m')}C, cloud {cloud}%")
        except Exception as e:
            print(f"[weather] Fetch failed: {e}")
