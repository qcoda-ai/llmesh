import os
import sys
import asyncio
import time
import subprocess
import platform
import io
import threading
from PIL import Image, ImageDraw
import dotenv

# Optional Tkinter for settings dialog
try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    tk = None

# Add project root to sys.path so `from lib.agent.client import ...` resolves
# when this example is run directly with `python examples/desktop-tray/app.py`.
# This file lives at <project_root>/examples/desktop-tray/app.py — three
# dirname() hops on abspath(__file__) reach the project root. PyInstaller-
# bundled runs use _MEIPASS instead.
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    project_root = sys._MEIPASS
else:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import pystray, warn if missing
try:
    import pystray
    from pystray import MenuItem as item
except ImportError:
    print("Error: pystray or Pillow not found. Please install them: pip install pystray Pillow")
    sys.exit(1)

# Import our actual agent logic
try:
    from lib.agent.client import main as agent_main
except ImportError as e:
    print(f"Error importing LLMesh agent client: {e}")
    sys.exit(1)

# Global state
is_running = False
loop_thread = None
agent_loop = None

class LogBuffer(io.StringIO):
    def __init__(self):
        super().__init__()
        self.max_lines = 1000
        self.logs = []

    def write(self, s):
        if s:
            # Also write to the original stdout/stderr so we don't go blind in the terminal
            sys.__stdout__.write(s)
            self.logs.append(s)
            if len(self.logs) > self.max_lines:
                self.logs.pop(0)

    def get_content(self):
        return "".join(self.logs)

log_buffer = LogBuffer()
sys.stdout = log_buffer
sys.stderr = log_buffer

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(project_root, "lib", "agent", "desktop", relative_path)

def create_image(color1, color2):
    """
    Generate a simple 64x64 dynamic icon for the OS Tray.
    Attempts to load icon.png if it exists.
    """
    icon_path = get_resource_path("icon.png")
    if os.path.exists(icon_path):
        try:
            return Image.open(icon_path)
        except Exception:
            pass

    image = Image.new('RGB', (64, 64), color1)
    dc = ImageDraw.Draw(image)
    dc.rectangle((16, 16, 48, 48), fill=color2)
    return image

ICONS = {
    'stopped': create_image('black', 'red'),
    'running': create_image('black', 'green'),
}

def start_agent_thread(icon):
    """
    Runs the agent's asyncio loop in a separate daemon thread
    so the pystray blocking icon loop doesn't freeze.
    """
    global is_running, agent_loop
    is_running = True
    
    icon.icon = ICONS['running']
    
    # We must create a new event loop for this background thread
    agent_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(agent_loop)
    
    try:
        agent_loop.run_until_complete(agent_main())
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Agent loop crashed: {e}")
    finally:
        icon.icon = ICONS['stopped']
        is_running = False

def get_macos_input(title, prompt, default_value="", hidden=False):
    """
    Fallback for macOS: Uses osascript (AppleScript) to show an input dialog
    when tkinter is not available.
    """
    hidden_str = "with hidden answer" if hidden else ""
    cmd = f'display dialog "{prompt}" default answer "{default_value}" with title "{title}" {hidden_str} buttons {{"Cancel", "OK"}} default button "OK"'
    try:
        result = subprocess.run(['osascript', '-e', cmd], capture_output=True, text=True)
        if result.returncode == 0:
            # Output format is "button returned:OK, text returned:InputContent"
            out = result.stdout.strip()
            if "text returned:" in out:
                return out.split("text returned:")[1]
        return None
    except Exception:
        return None

def show_settings_window(icon, item):
    """
    Spawns a native settings dialog. Uses tkinter if available, 
    falls back to AppleScript for macOS if tkinter is missing.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), ".env")
    current_url = os.getenv("HUB_URL", "http://localhost:8000")
    current_key = os.getenv("LLMESH_API_KEY", "")

    if tk:
        # Tkinter implementation
        root = tk.Tk()
        root.title("LLMesh Settings")
        root.geometry("350x200")
        root.attributes('-topmost', 1)
        root.after(100, lambda: root.attributes('-topmost', 0))
        
        tk.Label(root, text="Hub URL:").pack(pady=(10, 0))
        url_entry = tk.Entry(root, width=40)
        url_entry.insert(0, current_url)
        url_entry.pack(pady=5)
        
        tk.Label(root, text="API Key:").pack(pady=(10, 0))
        key_entry = tk.Entry(root, width=40, show="*")
        key_entry.insert(0, current_key)
        key_entry.pack(pady=5)
        
        def save_settings():
            new_url = url_entry.get().strip()
            new_key = key_entry.get().strip()
            if not new_key:
                messagebox.showerror("Error", "API Key is required!")
                return
            
            dotenv.set_key(env_path, "HUB_URL", new_url)
            dotenv.set_key(env_path, "LLMESH_API_KEY", new_key)
            os.environ["HUB_URL"] = new_url
            os.environ["LLMESH_API_KEY"] = new_key
            
            from lib.agent import client
            client.HUB_URL = new_url
            client.API_KEY = new_key
            
            messagebox.showinfo("Saved", "Settings saved successfully! Restart agent if it is currently running.")
            root.destroy()
            
        tk.Button(root, text="Save & Close", command=save_settings).pack(pady=10)
        root.mainloop()
    elif platform.system() == "Darwin":
        # macOS Fallback using AppleScript
        new_url = get_macos_input("LLMesh Settings", "Enter Hub URL:", current_url)
        if new_url is None: return # Cancelled
        
        new_key = get_macos_input("LLMesh Settings", "Enter API Key:", current_key, hidden=True)
        if new_key is None: return # Cancelled

        dotenv.set_key(env_path, "HUB_URL", new_url)
        dotenv.set_key(env_path, "LLMESH_API_KEY", new_key)
        os.environ["HUB_URL"] = new_url
        os.environ["LLMESH_API_KEY"] = new_key
        
        from lib.agent import client
        client.HUB_URL = new_url
        client.API_KEY = new_key
        
        subprocess.run(['osascript', '-e', 'display notification "LLMesh settings saved successfully!" with title "LLMesh"'])
    else:
        print("Error: Tkinter is required for settings on this platform.")

def show_log_window(icon, item):
    """
    Shows a window with the latest console logs.
    """
    if not tk:
        # Fallback for macOS
        if platform.system() == "Darwin":
            subprocess.run(['osascript', '-e', 'display notification "Log Viewer requires Tkinter. View output in Terminal." with title "LLMesh"'])
        return

    root = tk.Toplevel() if tk._default_root else tk.Tk()
    root.title("LLMesh Agent Logs")
    root.geometry("600x400")
    root.attributes('-topmost', 1)

    text_area = tk.Text(root, bg="black", fg="lightgreen", font=("Courier", 12))
    text_area.pack(expand=True, fill='both')

    def update_logs():
        if not root.winfo_exists():
            return
        content = log_buffer.get_content()
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, content)
        text_area.see(tk.END)
        root.after(1000, update_logs)

    update_logs()
    root.mainloop()
        # On Windows, we'd usually have tk, but we can't easily fallback to a native GUI there without more libs.

def on_start(icon, item):
    global loop_thread
    if not is_running:
        loop_thread = threading.Thread(target=start_agent_thread, args=(icon,), daemon=True)
        loop_thread.start()

def on_stop(icon, item):
    global is_running, agent_loop
    if is_running and agent_loop:
        # We must cancel all pending tasks in the agent's background loop
        for task in asyncio.all_tasks(agent_loop):
            task.cancel()
        agent_loop.call_soon_threadsafe(agent_loop.stop)
        is_running = False
        icon.icon = ICONS['stopped']

def on_quit(icon, item):
    on_stop(icon, item)
    icon.stop()

def setup_tray():
    # Attempt to load .env next to executable if it exists
    env_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), ".env")
    dotenv.load_dotenv(env_path)

    # Setup initial state menu
    menu = (
        item('Start LLMesh Agent', on_start, default=True),
        item('Stop LLMesh Agent', on_stop),
        item('View Logs', show_log_window),
        item('Settings', show_settings_window),
        pystray.Menu.SEPARATOR,
        item('Quit', on_quit)
    )

    # Initialize the tray app icon
    icon = pystray.Icon("LLMesh", ICONS['stopped'], "LLMesh Agent", menu)
    
    # Run the blocking GUI loop
    icon.run()

if __name__ == '__main__':
    setup_tray()
