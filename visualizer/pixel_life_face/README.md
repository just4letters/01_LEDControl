# LED Matrix Gaze & Nose Tracking Interface

A highly optimized, hardware-accelerated computer vision interface designed specifically for LED matrix displays. It uses MediaPipe's 3D facial transformation matrices to track head movement (nose-pointing) and facial blendshapes for blink detection and mouth-scaled UI elements.

## Features
* **Rigid 3D Nose Tracking:** Uses a 4x4 facial transformation matrix to eliminate "phantom pitch" when blinking or squinting.
* **Blink-Click & Freeze Engine:** Detects blinks via `eyeBlink` blendshapes and freezes the cursor matrix to prevent tracking jitter during the blink.
* **Black Hole Physics Reticle:** A procedural, true-2D physics particle system that acts as a gravity well, scaling dynamically with the user's `jawOpen` blendshape.
* **Pixel-Perfect Rendering:** Optimized for low-resolution LED canvases with 1:1 pixel drawing.

## Prerequisites
Ensure you have Python 3.9+ installed and a working webcam.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Step 1: Calibration**
You must calibrate the 3D matrix to your specific screen/webcam setup first.
```bash
python nose_calibration.py
```
* Point your nose at the red dots and press `SPACE`.
* This generates `nose_config.json`.

**Step 2: Run the App**
```bash
python nose_blink_app.py
```
* **Look around:** Moves the Black Hole reticle.
* **Open Mouth:** Expands the Event Horizon.
* **Blink:** Drops a physics-enabled shape on the canvas.
* **Press 'X':** Clears the canvas.
* **Press 'ESC':** Quits the application.

## Configuration Files
* `led_config.json`: Defines the hardware canvas size (e.g., `{"width": 256, "height": 256}`).
* `nose_config.json`: Auto-generated homography matrix from the calibration step. Do not edit manually.