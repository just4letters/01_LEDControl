"""
tools/doctor.py -- preflight for this machine.

Run it after a fresh clone, after moving folders, or whenever something
"was working yesterday". It checks the things git can't carry:

    python tools/doctor.py                 # everything, including camera + mic
    python tools/doctor.py --no-hardware   # skip camera / mic / display probes

It reads, never writes. API key *values* are never printed.
Exit code is 1 if anything FAILed, so it can gate a script.
"""
import argparse
import importlib.util
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

results = []


def report(level, name, detail=""):
    results.append(level)
    print(f"  {level:<4}  {name}" + (f" -- {detail}" if detail else ""))


def section(title):
    print(f"\n{title}")


# import name -> pip name, for the packages requirements.txt pins
REQUIRED = {
    "numpy": "numpy", "pygame": "pygame-ce", "cv2": "opencv-contrib-python",
    "mediapipe": "mediapipe", "PIL": "pillow", "dotenv": "python-dotenv",
    "google.genai": "google-genai", "speech_recognition": "SpeechRecognition",
    "whisper": "openai-whisper", "requests": "requests",
    "runwayml": "runwayml",
    # Imported lazily by SpeechRecognition inside recognize_whisper(), so nothing
    # fails until the first spoken word. Checked here so it fails *now* instead.
    "soundfile": "soundfile",
}


def check_python():
    section("Python")
    v = sys.version_info
    report("ok" if v >= (3, 13) else "WARN", f"Python {v.major}.{v.minor}.{v.micro}",
           "" if v >= (3, 13) else "repo was built and pinned on 3.14; older may differ")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    report("ok" if in_venv else "WARN", "virtualenv active" if in_venv else "not in a virtualenv")
    if in_venv:
        # A venv whose folder was moved keeps absolute paths baked into it.
        want = os.path.join(ROOT_DIR, "venv")
        here = os.path.realpath(sys.prefix)
        if os.path.realpath(want) != here and os.path.isdir(want):
            report("WARN", "running a different venv than ./venv", here)


def check_packages():
    section("Packages (requirements.txt)")
    for mod, pipname in REQUIRED.items():
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        report("ok" if found else "FAIL", pipname, "" if found else f"pip install {pipname}")
    # Microphone backend: core/mic.py uses PyAudio if present, else sounddevice.
    have = [m for m in ("pyaudio", "sounddevice") if importlib.util.find_spec(m) is not None]
    report("ok" if have else "FAIL", "microphone backend: " + (" / ".join(have) if have else "none"),
           "" if have else "pip install sounddevice  (no Homebrew or PortAudio needed)")
    # pygame vs pygame-ce share a namespace; both installed is a classic trap.
    try:
        from importlib import metadata
        names = {d.metadata["Name"].lower() for d in metadata.distributions()}
        cv = sorted(n for n in names if n.startswith("opencv-"))
        if len(cv) > 1:
            report("WARN", "more than one OpenCV package installed: " + ", ".join(cv),
                   "they overwrite each other's cv2/ folder; keep only opencv-contrib-python "
                   "(pip uninstall -y <all of them>, then pip install opencv-contrib-python==5.0.0.93)")
        if "pygame" in names and "pygame-ce" in names:
            report("FAIL", "both pygame and pygame-ce installed",
                   "pip uninstall pygame pygame-ce, then reinstall pygame-ce only")
    except Exception:
        pass


def check_files():
    section("Config, models, keys")
    # led_config.json is per panel/machine -- the launcher sends you to calibrate if absent.
    cfg_paths = {
        "led_config.json": ("screen calibration: python tools/screen_calibration.py", True),
        "finger_config.json": ("finger calibration: python tools/finger_calibrate_voice.py", True),
        "nose_config.json": ("nose calibration (only the Black Hole Sandbox needs it)", False),
    }
    for fn, (hint, needed) in cfg_paths.items():
        p = os.path.join(ROOT_DIR, fn)
        if not os.path.isfile(p):
            report("FAIL" if needed else "WARN", fn, "missing -- " + hint)
            continue
        try:
            data = json.load(open(p))
            report("ok", fn, "")
            if fn == "led_config.json":
                w, h = int(data["width"]), int(data["height"])
                ox, oy = int(data.get("offset_x", 0)), int(data.get("offset_y", 0))
                fits = ox >= 0 and oy >= 0 and ox + w <= 1920 and oy + h <= 1080
                report("ok" if fits else "FAIL", f"grid {w}x{h} at ({ox},{oy}) inside 1920x1080")
        except Exception as e:
            report("FAIL", fn, f"unreadable: {e}")
    for fn in ("hand_landmarker.task", "face_landmarker.task", "pose_landmarker.task"):
        p = os.path.join(ROOT_DIR, "models", fn)
        ok = os.path.isfile(p) and os.path.getsize(p) > 100_000
        report("ok" if ok else "FAIL", "models/" + fn, "" if ok else "missing or truncated (git lfs? partial clone?)")
    env = os.path.join(ROOT_DIR, ".env")
    keys = {}
    if os.path.isfile(env):
        for line in open(env):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip().strip("'\"")
    for k, needed in (("GEMINI_API_KEY", True), ("RUNWAYML_API_SECRET", False)):
        have = bool(keys.get(k) or os.environ.get(k))
        report("ok" if have else ("FAIL" if needed else "WARN"), k,
               "set" if have else ("missing -- every voice app needs it" if needed else "missing -- only pixel_life_runway"))


def check_hardware():
    section("Display")
    try:
        import pygame
        os.environ.setdefault("SDL_VIDEO_ALLOW_SCREENSAVER", "1")
        pygame.display.init()
        n = pygame.display.get_num_displays()
        sizes = pygame.display.get_desktop_sizes()
        report("ok" if n >= 2 else "WARN", f"{n} display(s): {sizes}",
               "" if n >= 2 else "no second display -- apps fall back to your main one, not pixel-accurate")
        if n >= 2 and tuple(sizes[1]) != (1920, 1080):
            report("WARN", f"display 1 is {sizes[1]}, panel expects 1920x1080")
        pygame.display.quit()
    except Exception as e:
        report("FAIL", "pygame display probe", str(e))

    section("Cameras (core/config.py: finger=0, face=1)")
    try:
        import cv2
        from core import config
        found = 0
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    found += 1
                    h, w = frame.shape[:2]
                    tag = " (FINGER_CAMERA_INDEX)" if i == config.FINGER_CAMERA_INDEX else ""
                    tag += " (FACE_CAMERA_INDEX)" if i == config.FACE_CAMERA_INDEX else ""
                    report("ok", f"camera {i}: {w}x{h}{tag}")
                else:
                    report("WARN", f"camera {i} opened but gave no frame",
                           "macOS: grant Camera permission to your terminal")
            cap.release()
        if found == 0:
            report("FAIL", "no camera delivered a frame")
        for label, idx in (("finger", config.FINGER_CAMERA_INDEX), ("face", config.FACE_CAMERA_INDEX)):
            if idx >= found and found:
                report("WARN", f"{label} camera index {idx} but only {found} camera(s) found",
                       "tracker will fall back to 0; edit core/config.py on this machine")
    except Exception as e:
        report("FAIL", "camera probe", str(e))

    section("Microphone")
    try:
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            ins = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
            ins = [d for d in ins if d.get("maxInputChannels", 0) > 0]
            pa.terminate()
            names, via = [d["name"] for d in ins], "pyaudio"
        except ImportError:
            import sounddevice as sd
            names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
            via = "sounddevice"
        if names:
            report("ok", f"{len(names)} input device(s) via {via}", ", ".join(names[:3]))
        else:
            report("FAIL", "no input devices", "macOS: grant Microphone permission to your terminal")
    except Exception as e:
        report("FAIL", "microphone probe", str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hardware", action="store_true", help="skip display, camera and mic probes")
    args = ap.parse_args()
    print(f"LEDControl doctor -- {ROOT_DIR}")
    check_python()
    check_packages()
    check_files()
    if not args.no_hardware:
        check_hardware()
    fails, warns = results.count("FAIL"), results.count("WARN")
    print(f"\n{results.count('ok')} ok, {warns} warning(s), {fails} failure(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
