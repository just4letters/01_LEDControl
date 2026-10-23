"""
tools/bench.py -- can this machine run the light palette? Measure, don't guess.

    python tools/bench.py                 # everything it can, using the real camera
    python tools/bench.py --no-camera     # synthetic frames (no camera needed)
    python tools/bench.py --mic           # also record 5 s of your voice for Whisper
    python tools/bench.py --seconds 15    # longer tracking runs

Each section is independent and skips itself, saying why, if a package is
missing -- so it still tells you something on a machine where the full
requirements.txt won't install. Paste the final SUMMARY block back to compare
machines.

What it measures, in the order that matters for the real rig:
  1. sketches   simulation cost per frame (numpy + pygame only)
  2. hand       MediaPipe HandLandmarker in VIDEO mode, the same calls
                core/vision.py makes: camera read + colour convert + detect
  3. whisper    recognize_whisper(base.en) on 5 s of audio, the same call
                core/voice.py makes; load time and warm per-phrase time
  4. together   hand tracking while Whisper transcribes in the background --
                the situation on a dual-core machine, and the one that hurts

For the hand sections, hold your hand up to the camera so MediaPipe has
something to find; detection is slower with a hand in frame than without.
Verdict thresholds are rules of thumb chosen for this rig (the app asks the
camera for 60 fps; its smoothing, alpha = 0.35, was tuned at normal webcam
rates), not published limits.
"""
import argparse
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SUMMARY = []   # (label, value, verdict)
ISOLATED = {}  # fps measured in section 2, before Whisper has touched the CPU


def hdr(t):
    print(f"\n=== {t}")


def skip(section, why):
    print(f"  SKIPPED: {why}")
    SUMMARY.append((section, "skipped", why.split("\n")[0][:70]))


def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p))]


def sysinfo():
    hdr("machine")
    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    mem = ""
    try:
        mem = f", {int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)) / 2**30:.0f} GB RAM"
    except Exception:
        pass
    print(f"  {platform.system()} {platform.mac_ver()[0] or platform.release()} {platform.machine()}")
    print(f"  {cpu}, {os.cpu_count()} logical cores{mem}")
    print(f"  Python {platform.python_version()}")
    SUMMARY.append(("machine", f"{cpu}, {os.cpu_count()} cores{mem}, {platform.machine()}, Python {platform.python_version()}", ""))


# ---------------------------------------------------------------- sketches
def bench_sketches():
    hdr("1. sketches (simulation cost, no display)")
    script = os.path.join(ROOT_DIR, "visualizer", "ambient", "sketches", "_smoke_headless.py")
    if not os.path.isfile(script):
        return skip("sketches", "_smoke_headless.py not found")
    if not os.path.isfile(os.path.join(ROOT_DIR, "led_config.json")):
        return skip("sketches", "led_config.json is missing (it is gitignored, so a fresh clone has none).\n"
                                "  For a benchmark any panel's file will do: AirDrop one from the main Mac into this folder.")
    try:
        out = subprocess.run([sys.executable, script, "--frames", "600"], capture_output=True, text=True, timeout=300)
    except Exception as e:
        return skip("sketches", f"{type(e).__name__}: {e}")
    worst = 0.0
    for line in out.stdout.splitlines():
        if line.startswith(("PASS", "FAIL")):
            print("  " + line)
            m = re.search(r"([\d.]+) ms/frame", line)
            if m:
                worst = max(worst, float(m.group(1)))
    if out.returncode != 0 and not worst:
        return skip("sketches", (out.stderr or out.stdout).strip().splitlines()[-1][:120])
    # 60 fps budget is 16.7 ms per frame for everything.
    verdict = "ok" if worst < 5 else ("marginal" if worst < 10 else "slow")
    SUMMARY.append(("sketches, worst ms/frame", f"{worst:.2f}", verdict))


# ------------------------------------------------------------ hand tracking
class HandRig:
    """The same camera + MediaPipe setup as core/vision.py HandTracker, minus the thread."""

    def __init__(self, use_camera):
        import cv2
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision as mp_vision
        from core import config
        self.cv2, self.mp, self.np, self.config = cv2, mp, np, config
        opts = mp_vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH),
            num_hands=1, running_mode=mp_vision.RunningMode.VIDEO)
        self.det = mp_vision.HandLandmarker.create_from_options(opts)
        self.cap = None
        self.t0 = time.time()
        self.last_ts = -1
        if use_camera:
            self.cap = cv2.VideoCapture(config.FINGER_CAMERA_INDEX)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
            else:
                print("  (camera did not open; using synthetic frames)")
                self.cap = None
        self.synthetic = np.random.randint(0, 255, (config.CAM_HEIGHT, config.CAM_WIDTH, 3), dtype=np.uint8)

    def step(self):
        """One loop iteration. Returns (read_s, convert_s, detect_s, hand_found)."""
        cv2, mp = self.cv2, self.mp
        t = time.perf_counter()
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                frame = self.synthetic
        else:
            frame = self.synthetic
        t_read = time.perf_counter() - t
        t = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_conv = time.perf_counter() - t
        ts = max(int((time.time() - self.t0) * 1000), self.last_ts + 1)   # VIDEO mode needs rising timestamps
        self.last_ts = ts
        t = time.perf_counter()
        res = self.det.detect_for_video(img, ts)
        t_det = time.perf_counter() - t
        return t_read, t_conv, t_det, bool(res.hand_landmarks)

    def close(self):
        if self.cap is not None:
            self.cap.release()


def run_hand(rig, seconds):
    reads, convs, dets, found, stamps = [], [], [], 0, []
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        r, c, d, f = rig.step()
        reads.append(r); convs.append(c); dets.append(d); found += f; n += 1
        stamps.append(time.perf_counter())
    return reads, convs, dets, found, stamps


def fps_verdict(fps):
    return "good" if fps >= 25 else ("marginal" if fps >= 15 else "poor")


def bench_hand(args):
    hdr(f"2. hand tracking ({'camera' if not args.no_camera else 'synthetic frames'}, {args.seconds}s)")
    try:
        rig = HandRig(not args.no_camera)
    except Exception as e:
        skip("hand tracking", f"{type(e).__name__}: {e}\n  pip install mediapipe opencv-contrib-python")
        return None
    if rig.cap is not None:
        print("  Hold your hand up to the camera now...")
        time.sleep(2)
    for _ in range(5):          # warm-up: first frames pay one-time setup
        rig.step()
    reads, convs, dets, found, stamps = run_hand(rig, args.seconds)
    n = len(dets)
    fps = n / args.seconds
    print(f"  {n} frames, loop {fps:.1f} fps | read {statistics.mean(reads)*1000:.1f} ms, "
          f"convert {statistics.mean(convs)*1000:.1f} ms, detect {statistics.mean(dets)*1000:.1f} ms "
          f"(p95 {pct(dets, .95)*1000:.1f})")
    if rig.cap is not None:
        print(f"  hand found in {found}/{n} frames" + ("  <-- hold a hand up for a fair test" if found < n * .5 else ""))
    ISOLATED["fps"] = fps
    SUMMARY.append(("hand tracking, loop fps", f"{fps:.1f}", fps_verdict(fps)))
    SUMMARY.append(("hand tracking, detect ms", f"{statistics.mean(dets)*1000:.1f} (p95 {pct(dets,.95)*1000:.1f})", ""))
    return rig


# ------------------------------------------------------------------ whisper
def make_audio(use_mic):
    import speech_recognition as sr
    import numpy as np
    if use_mic:
        r = sr.Recognizer()
        from core.mic import open_microphone
        with open_microphone(sample_rate=16000) as src:
            print("  Recording 5 s -- say something like 'make it rain heavily' ...")
            return r.record(src, duration=5)
    rng = np.random.default_rng(0)
    pcm = (rng.standard_normal(16000 * 5) * 300).astype("<i2").tobytes()
    return sr.AudioData(pcm, 16000, 2)


def whisper_once(rec, audio):
    t = time.perf_counter()
    text = rec.recognize_whisper(audio, model="base.en")
    return time.perf_counter() - t, text


def bench_whisper(args):
    hdr("3. whisper base.en (same call as core/voice.py)")
    try:
        import speech_recognition as sr
        import whisper  # noqa: F401
        import soundfile  # noqa: F401
    except Exception as e:
        hint = "pip install openai-whisper soundfile SpeechRecognition"
        if "numpy" in str(e).lower():
            hint = "torch and NumPy disagree: pip install 'numpy<2' (and an OpenCV 4.x, since OpenCV 5 needs NumPy 2)"
        return skip("whisper", f"{type(e).__name__}: {e}\n  {hint}")
    try:
        import numpy as np
        import torch
        torch.from_numpy(np.zeros(1, dtype="float32"))
    except Exception as e:
        return skip("whisper", f"torch cannot exchange data with this NumPy ({type(e).__name__}: {e})\n"
                               "  torch 2.2.x needs NumPy 1.x: pip install 'numpy==1.26.4' 'opencv-contrib-python==4.10.0.84'")
    try:
        audio = make_audio(args.mic)
    except Exception as e:
        print(f"  (mic unavailable: {e}\n   using synthetic audio; check microphone permission for your terminal)")
        args.mic = False
        audio = make_audio(False)
    rec = sr.Recognizer()
    if not args.mic:
        print("  (synthetic noise stands in for speech; use --mic for a real phrase)")
    print("  first call loads the model (and downloads ~140 MB the very first time)...")
    try:
        t_first, _ = whisper_once(rec, audio)
        warm = []
        for _ in range(3):
            t, text = whisper_once(rec, audio)
            warm.append(t)
    except Exception as e:
        skip("whisper", f"{type(e).__name__}: {e}")
        return None
    med = statistics.median(warm)
    print(f"  first call {t_first:.1f} s, then {', '.join(f'{w:.2f}' for w in warm)} s per 5 s phrase  -> '{text.strip()}'")
    SUMMARY.append(("whisper, warm s per 5 s phrase", f"{med:.2f}", "good" if med < 2 else ("marginal" if med < 5 else "slow")))
    SUMMARY.append(("whisper, first-call load s", f"{t_first:.1f}", ""))
    return rec, audio


# ----------------------------------------------------------------- together
def bench_together(args, rig, wh):
    hdr("4. together: hand tracking while whisper transcribes")
    if rig is None or wh is None:
        return skip("together", "needs both hand tracking and whisper to have run")
    rec, audio = wh
    stop = threading.Event()
    started = threading.Event()

    def bg():
        started.wait()
        while not stop.is_set():
            try:
                whisper_once(rec, audio)
            except Exception:
                return

    th = threading.Thread(target=bg, daemon=True)
    th.start()
    time.sleep(3)    # let Whisper's worker threads and the CPU settle after section 3
    quiet = max(5.0, args.seconds / 2)
    _, _, d0, _, s0 = run_hand(rig, quiet)
    started.set()
    _, _, d1, _, s1 = run_hand(rig, args.seconds)
    stop.set()
    fps0, fps1 = len(d0) / quiet, len(d1) / args.seconds
    worst_gap = max((b - a for a, b in zip(s1, s1[1:])), default=0)
    iso = ISOLATED.get("fps")
    print(f"  isolated (section 2): {iso:.1f} fps | just before Whisper: {fps0:.1f} fps | "
          f"with Whisper running: {fps1:.1f} fps | longest stall {worst_gap*1000:.0f} ms")
    if iso and fps0 < 0.85 * iso:
        print(f"  NOTE: the pre-Whisper baseline ({fps0:.1f}) is well below the isolated run ({iso:.1f}), so the machine "
              "was already slowed (heat, or threads left over from section 3). Judge by the isolated number.")
    base = max(fps0, iso or 0)
    print(f"  slowdown from Whisper, against the better baseline ({base:.1f} fps): {(1 - fps1 / base) * 100:.0f}%")
    SUMMARY.append(("together, loop fps while whisper runs", f"{fps1:.1f}", fps_verdict(fps1)))
    SUMMARY.append(("together, slowdown vs best baseline", f"{(1 - fps1 / base) * 100:.0f}%", ""))
    SUMMARY.append(("together, longest stall ms", f"{worst_gap*1000:.0f}",
                    "ok" if worst_gap < .25 else ("noticeable" if worst_gap < .6 else "cursor freezes")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-camera", action="store_true", help="use synthetic frames instead of the webcam")
    ap.add_argument("--mic", action="store_true", help="record 5 s of real speech for the Whisper test")
    ap.add_argument("--seconds", type=int, default=10, help="length of each tracking run (default 10)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["sketches", "hand", "whisper", "together"])
    args = ap.parse_args()

    sysinfo()
    if "sketches" not in args.skip:
        bench_sketches()
    rig = bench_hand(args) if "hand" not in args.skip else None
    wh = bench_whisper(args) if "whisper" not in args.skip else None
    if "together" not in args.skip:
        bench_together(args, rig, wh)
    if rig:
        rig.close()

    print("\n" + "=" * 60 + "\nSUMMARY  (paste this block back)\n" + "=" * 60)
    for label, value, verdict in SUMMARY:
        print(f"  {label:<42} {value:<28} {verdict}")
    print("\nThresholds are rules of thumb: >=25 fps good, 15-25 marginal, <15 poor;"
          "\nwhisper <2 s good, 2-5 s marginal, >5 s slow.")


if __name__ == "__main__":
    main()
