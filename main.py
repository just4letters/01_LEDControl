import os
import subprocess
import sys

def get_available_apps():
    """Scans the visualizer directory for available game cartridges."""
    vis_dir = "visualizer"
    
    if not os.path.exists(vis_dir):
        print(f"❌ Error: Could not find the '{vis_dir}' directory.")
        sys.exit()

    # Get all folders inside the visualizer directory, ignoring hidden/cache folders
    apps = [d for d in os.listdir(vis_dir) 
            if os.path.isdir(os.path.join(vis_dir, d)) and not d.startswith('_') and not d.startswith('.')]
    
    apps.sort()
    return apps

def display_menu(apps):
    """Shows the interactive terminal menu."""
    print("\n" + "="*35)
    print(" 🎮 LED PHYSICS ENGINE LAUNCHER 🎮")
    print("="*35)
    
    if not apps:
        print("❌ No visualizer apps found in the 'visualizer' folder!")
        sys.exit()

    for i, app in enumerate(apps):
        # Capitalize the names and replace underscores with spaces for a cleaner look
        clean_name = app.replace("_", " ").title()
        print(f"  [{i + 1}] - {clean_name}")
        
    print("  [ Q ] - Quit")
    print("="*35)

def main():
    apps = get_available_apps()
    display_menu(apps)
    
    while True:
        choice = input("\nSelect a visualizer to boot: ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            sys.exit()
            
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(apps):
                ACTIVE_APP = apps[choice_idx]
                break
            else:
                print("❌ Invalid number. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number or 'Q' to quit.")

    print(f"\n🚀 Booting engine for visualizer: {ACTIVE_APP.upper()}...")

    # 1. Get the exact path to the app
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

if __name__ == "__main__":
    main()