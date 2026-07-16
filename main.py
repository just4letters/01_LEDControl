import os
import subprocess
import sys

# --- THE CARTRIDGE SLOT ---
ACTIVE_APP = "waterfall" 

print(f"🚀 Booting engine for visualizer: {ACTIVE_APP}")

# 1. Get the path to the app (using your exact folder name 'visualizer')
script_path = os.path.join("visualizer", ACTIVE_APP, "app.py")

if not os.path.exists(script_path):
    print(f"❌ Error: Could not find {script_path}")
    sys.exit()

# 2. Tell Python the root folder exists so it can find 'core'
env = os.environ.copy()
env["PYTHONPATH"] = os.getcwd()

# 3. Launch the app!
try:
    subprocess.run([sys.executable, script_path], env=env)
except KeyboardInterrupt:
    print("\n🛑 Shutting down engine.")