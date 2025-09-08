import os
import sys
import json
import platform
import tempfile
import datetime
import subprocess
import time
from typing import Optional, Dict, List, Any

# Ensure parent directory is in path to allow import of py_classes
# This helps with standalone testing of the utility.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.utils_manager.util_base import UtilBase

# Use a try-except block for dependencies to allow the tool to be listed
# even if some platform-specific libraries are missing. A comprehensive check
# is performed within the run method.
_MSS_INSTALLED = False
try:
    import mss
    _MSS_INSTALLED = True
except ImportError:
    mss = None

_PILLOW_INSTALLED = False
try:
    from PIL import Image
    _PILLOW_INSTALLED = True
except ImportError:
    Image = None

# Placeholder for checking platform-specific dependencies.
# The actual check is performed within the run method when needed.
_PLATFORM_DEPS_CHECKED = False
_PLATFORM_DEPS_INSTALLED = False


class TakeScreenshotUtil(UtilBase):
    """
    A utility to take screenshots or list window titles.
    This tool uses `mss` for cross-platform screen capturing and platform-specific
    methods to identify and capture application windows or list their titles.

    Dependencies:
    - Core: python-mss, Pillow
    - Windows-specific: pywin32
    - Linux-specific (capture): xdotool, xwininfo
    - Linux-specific (list titles): wmctrl
    - macOS-specific: OS-level AppleScript support (built-in)
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["capture screen", "screenshot", "get window", "screen grab", "snapshot", "UI", "visual", "documentation", "bug report", "error screen", "terminal output", "command result", "visual proof", "share screen", "show progress", "demo", "tutorial", "evidence"],
            "use_cases": [
                "Take a screenshot of the entire desktop.",
                "Capture the current VS Code window.",
                "List all the currently open application windows.",
                "Take a screenshot to document this error.",
                "Capture the terminal showing the installation process.",
                "Screenshot the Docker container status for debugging.",
                "Get a visual of the model download progress."
            ],
            "arguments": {
                "window_query": "A substring of the window title to capture. If omitted, captures all screens.",
                "list_window_titles": "If true, lists all window titles instead of taking a screenshot.",
                "reload": "If true, attempts to refresh the window (send F5/Cmd+R) before capturing."
            }
        }

    @staticmethod
    def _check_platform_deps_for_capture():
        """Checks for platform-specific dependencies required for window capture."""
        global _PLATFORM_DEPS_CHECKED, _PLATFORM_DEPS_INSTALLED
        if _PLATFORM_DEPS_CHECKED:
            return

        system = platform.system()
        if system == "Windows":
            try:
                import win32gui
                _PLATFORM_DEPS_INSTALLED = True
            except ImportError:
                _PLATFORM_DEPS_INSTALLED = False
        elif system == "Linux":
            from shutil import which
            if which('xdotool') and which('xwininfo'):
                _PLATFORM_DEPS_INSTALLED = True
            else:
                _PLATFORM_DEPS_INSTALLED = False
        elif system == "Darwin":  # macOS
            _PLATFORM_DEPS_INSTALLED = True
        
        _PLATFORM_DEPS_CHECKED = True

    @staticmethod
    def _list_windows_windows() -> List[str]:
        """List visible window titles on Windows."""
        import win32gui
        titles = set()
        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                # Filter out empty titles and non-interactive windows
                if title and win32gui.GetWindowRect(hwnd) != (0, 0, 0, 0):
                    titles.add(title)
            return True
        win32gui.EnumWindows(enum_callback, None)
        return sorted(list(titles))

    @staticmethod
    def _list_windows_linux() -> List[str]:
        """List visible window titles on Linux using wmctrl."""
        from shutil import which
        if not which('wmctrl'):
            raise FileNotFoundError("The 'wmctrl' command-line tool is required for listing windows on Linux. Please install it (e.g., 'sudo apt-get install wmctrl').")
        
        result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, check=True)
        titles = []
        for line in result.stdout.strip().splitlines():
            # Line format: 0x...  0 hostname  Window Title -> extract title part
            parts = line.split(maxsplit=3)
            if len(parts) > 3:
                titles.append(parts[3])
        return sorted(titles)

    @staticmethod
    def _list_windows_mac() -> List[str]:
        """List visible window titles on macOS."""
        script = '''
            set windowTitles to ""
            tell application "System Events"
                set procs to processes whose background only is false
                repeat with proc in procs
                    try
                        if (count of windows of proc) > 0 then
                            repeat with win in windows of proc
                                set t to name of win
                                if t is not "" and t is not missing value then
                                    set windowTitles to windowTitles & t & "\\n"
                                end if
                            end repeat
                        end if
                    end try
                end repeat
            end tell
            return windowTitles
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=True)
        return sorted([title for title in result.stdout.strip().split('\n') if title])

    @staticmethod
    def _get_window_coords_windows(title_query: str) -> tuple[Optional[Dict[str, int]], Optional[Any]]:
        """Find window by title and get its coordinates on Windows."""
        import win32gui
        
        target_hwnd = None
        
        def enum_callback(hwnd, _):
            nonlocal target_hwnd
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                if title_query.lower() in win32gui.GetWindowText(hwnd).lower():
                    if win32gui.GetWindowRect(hwnd) != (0, 0, 0, 0):
                        target_hwnd = hwnd
                        return False
            return True

        win32gui.EnumWindows(enum_callback, None)
        
        if target_hwnd:
            rect = win32gui.GetWindowRect(target_hwnd)
            return {'left': rect[0], 'top': rect[1], 'width': rect[2] - rect[0], 'height': rect[3] - rect[1]}, target_hwnd
        return None, None

    @staticmethod
    def _get_window_coords_linux(title_query: str) -> tuple[Optional[Dict[str, int]], Optional[str]]:
        """Find window by title and get its coordinates on Linux (X11)."""
        try:
            result = subprocess.run(['xdotool', 'search', '--name', title_query], capture_output=True, text=True, check=True)
            window_ids = result.stdout.strip().split()
            if not window_ids: return None, None
            
            window_id = None
            for wid in window_ids:
                type_check = subprocess.run(['xprop', '-id', wid, '_NET_WM_WINDOW_TYPE'], capture_output=True, text=True)
                if '_NET_WM_WINDOW_TYPE_NORMAL' in type_check.stdout or type_check.returncode != 0:
                     window_id = wid
                     break
            if not window_id: return None, None

            info_result = subprocess.run(['xwininfo', '-id', window_id], capture_output=True, text=True, check=True)
            
            coords = {}
            for line in info_result.stdout.splitlines():
                line = line.strip()
                if 'Absolute upper-left X:' in line: coords['left'] = int(line.split(':')[1].strip())
                elif 'Absolute upper-left Y:' in line: coords['top'] = int(line.split(':')[1].strip())
                elif 'Width:' in line: coords['width'] = int(line.split(':')[1].strip())
                elif 'Height:' in line: coords['height'] = int(line.split(':')[1].strip())
            
            if all(k in coords for k in ['left', 'top', 'width', 'height']): return coords, window_id
            
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
            return None, None
        return None, None

    @staticmethod
    def _get_window_coords_mac(title_query: str) -> Optional[Dict[str, int]]:
        """Find window by title and get its coordinates on macOS."""
        script = f'''
            set windowDetails to ""
            tell application "System Events"
                repeat with proc in (processes whose background only is false)
                    try
                        repeat with win in windows of proc
                            if name of win contains "{title_query}" then
                                set {{x, y}} to position of win
                                set {{w, h}} to size of win
                                return "" & x & "," & y & "," & w & "," & h
                            end if
                        end repeat
                    end try
                end repeat
            end tell
            return windowDetails
        '''
        try:
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            if not output: return None
            
            parts = output.split(',')
            return {'left': int(parts[0]), 'top': int(parts[1]), 'width': int(parts[2]), 'height': int(parts[3])}
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
            return None

    @staticmethod
    def _reload_window_windows(hwnd):
        """Send F5 key to reload window on Windows"""
        import win32gui
        import win32con
        import win32api
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.1) # Give a short delay for window to become active
        win32api.keybd_event(win32con.VK_F5, 0, 0, 0)
        win32api.keybd_event(win32con.VK_F5, 0, win32con.KEYEVENTF_KEYUP, 0)

    @staticmethod
    def _reload_window_linux(window_id):
        """Send F5 key to reload window on Linux"""
        from shutil import which
        if not which('xdotool'):
            raise FileNotFoundError("The 'xdotool' command-line tool is required for reloading windows on Linux. Please install it (e.g., 'sudo apt-get install xdotool').")
        subprocess.run(['xdotool', 'windowactivate', window_id], check=True)
        time.sleep(0.1) # Give a short delay for window to become active
        subprocess.run(['xdotool', 'key', 'F5'], check=True)

    @staticmethod
    def _reload_window_mac(title):
        """Send Cmd+R to reload window on macOS"""
        script = f'''
        tell application "System Events"
            set found to false
            repeat with proc in (processes whose background only is false)
                try
                    repeat with win in windows of proc
                        if name of win contains "{title}" then
                            set frontmost of proc to true
                            delay 0.1 -- Give a short delay for window to become active
                            keystroke "r" using command down
                            set found to true
                            exit repeat
                        end if
                    end repeat
                    if found then exit repeat
                end try
            end repeat
        end tell
        '''
        subprocess.run(['osascript', '-e', script], check=True)

    @staticmethod
    def _run_logic(window_query: Optional[str] = None, 
            list_window_titles: bool = False,
            reload: bool = False) -> str:
        """
        Takes a screenshot or lists visible window titles.

        - If `list_window_titles` is True, it returns a list of all visible window titles.
        - If `window_query` is provided, captures the first visible window matching the query.
        - If neither is provided, captures all connected displays into a single image file.

        Args:
            window_query (Optional[str]): A part of the window title to search for.
            list_window_titles (bool): If True, ignores other options and lists all window titles.
            reload (bool): If True, attempts to reload the specified window before capturing.
                           This is only applicable when `window_query` is provided.
        Returns:
            str: A JSON string with a 'result' key on success, or an 'error' key on failure.
        """
        system = platform.system()

        if list_window_titles:
            try:
                titles = []
                if system == "Windows":
                    import win32gui
                    titles = TakeScreenshotUtil._list_windows_windows()
                elif system == "Linux":
                    titles = TakeScreenshotUtil._list_windows_linux()
                elif system == "Darwin":
                    titles = TakeScreenshotUtil._list_windows_mac()
                else:
                    return json.dumps({"error": f"Listing windows is not supported on this OS: {system}"})
                
                return json.dumps({"result": {
                    "status": "Success",
                    "message": f"Found {len(titles)} visible window(s).",
                    "window_titles": titles}}, indent=2)
            except (FileNotFoundError, ImportError, subprocess.CalledProcessError) as e:
                return json.dumps({"error": f"Failed to list windows: {str(e)}"})
            except Exception as e:
                return json.dumps({"error": f"An unexpected error occurred while listing windows: {str(e)}"})

        if not _MSS_INSTALLED or not _PILLOW_INSTALLED:
            return json.dumps({"error": "Dependencies not installed. Run: pip install python-mss pillow"})

        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with mss.mss() as sct:
                if not window_query:
                    # Capture all displays into a single image file
                    monitor_all = sct.monitors[0]
                    if not monitor_all or monitor_all.get('width', 0) <= 0:
                        return json.dumps({"error": "No monitors found or dimensions are invalid."})
                    
                    filename = os.path.join(temp_dir, f"screenshot_all-monitors_{timestamp}.png")
                    sct_img = sct.grab(monitor_all)
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    img.save(filename)
                    
                    return json.dumps({"result": {
                        "status": "Success",
                        "message": "Captured all displays into a single image.",
                        "file": filename}}, indent=2)
                else:
                    # Capture a specific window
                    TakeScreenshotUtil._check_platform_deps_for_capture()
                    if not _PLATFORM_DEPS_INSTALLED:
                        dep_map = {"Windows": "pywin32", "Linux": "xdotool and xwininfo"}
                        return json.dumps({"error": f"Window capture dependencies not found for {system}. Install: {dep_map.get(system, 'required libraries')}."})

                    coords = None
                    window_id_or_title = None # This will hold hwnd for Windows, window_id for Linux, and title for macOS reload

                    if system == "Windows": 
                        coords, window_id_or_title = TakeScreenshotUtil._get_window_coords_windows(window_query)
                    elif system == "Linux": 
                        coords, window_id_or_title = TakeScreenshotUtil._get_window_coords_linux(window_query)
                    elif system == "Darwin": 
                        coords = TakeScreenshotUtil._get_window_coords_mac(window_query)
                        window_id_or_title = window_query # For macOS, we pass the title for reload
                    else: 
                        return json.dumps({"error": f"Window capture not supported on this OS: {system}"})
                    
                    if not coords:
                        return json.dumps({"error": f"Window matching '{window_query}' not found or its geometry is invalid."})

                    # NEW: Reload window if requested
                    if reload and window_id_or_title is not None:
                        try:
                            if system == "Windows":
                                TakeScreenshotUtil._reload_window_windows(window_id_or_title)
                            elif system == "Linux":
                                TakeScreenshotUtil._reload_window_linux(window_id_or_title)
                            elif system == "Darwin":
                                TakeScreenshotUtil._reload_window_mac(window_id_or_title)
                            
                            # Wait for reload to complete
                            time.sleep(2)  
                        except Exception as e:
                            # We don't want to fail the entire capture if reload fails, so just print warning and continue
                            print(f"Warning: Failed to reload window '{window_query}': {str(e)}", file=sys.stderr)

                    filename = os.path.join(temp_dir, f"screenshot_window_{timestamp}.png")
                    sct_img = sct.grab(coords)
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    img.save(filename)
                    
                    return json.dumps({"result": {
                        "status": "Success",
                        "message": f"Captured window matching '{window_query}'.",
                        "file": filename}}, indent=2)
        
        except Exception as e:
            return json.dumps({"error": f"An unexpected screenshot error occurred: {str(e)}"})


# Module-level run function for CLI-Agent compatibility
def run(path: str, window_query: str = None, list_windows: bool = False) -> str:
    """
    Module-level wrapper for TakeScreenshotUtil._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        path (str): The file path to save the screenshot
        window_query (str): Window title or ID to capture specific window
        list_windows (bool): Whether to list available windows
        
    Returns:
        str: JSON string with result or error
    """
    return TakeScreenshotUtil._run_logic(path=path, window_query=window_query, list_windows=list_windows)

# Example usage for standalone testing
if __name__ == '__main__':
    print(f"Running on platform: {platform.system()}")

    print("\n--- Test Case 1: List all visible window titles ---")
    list_result = TakeScreenshotUtil._run_logic(list_window_titles=True)
    print(list_result)
    try:
        data = json.loads(list_result)
        if 'result' in data and 'window_titles' in data['result']:
            print(f"Success! Found {len(data['result']['window_titles'])} windows.")
        else:
            print(f"Failed. Response: {data.get('error', 'Unknown error')}")
    except (json.JSONDecodeError, KeyError):
        print("Failed to parse JSON response.")

    print("\n--- Test Case 2: Capture all screens (single file) ---")
    all_screens_result = TakeScreenshotUtil._run_logic()
    print(all_screens_result)
    try:
        data = json.loads(all_screens_result)
        if 'result' in data and data['result'].get('file'):
            print(f"Success! Screenshot saved to: {data['result']['file']}")
        else:
            print(f"Failed. Response: {data.get('error', 'Unknown error')}")
    except (json.JSONDecodeError, KeyError):
        print("Failed to parse JSON response.")

    print("\n--- Test Case 3: Capture a specific window (e.g., this terminal/IDE) ---")
    window_name_query = "code" if platform.system() != "Windows" else "Explorer"
    print(f"Attempting to capture window with title containing: '{window_name_query}'")
    window_result = TakeScreenshotUtil._run_logic(window_query=window_name_query)
    print(window_result)
    try:
        data = json.loads(window_result)
        if 'result' in data and data['result'].get('file'):
            print(f"Success! Window screenshot saved to: {data['result']['file']}")
        else:
            print(f"Failed. Response: {data.get('error', 'Unknown error')}")
    except (json.JSONDecodeError, KeyError):
        print("Failed to parse JSON response.")

    print("\n--- Test Case 4: Capture a non-existent window ---")
    non_existent_window_query = "ajskdhaslkdjhalskdjasxxxx"
    not_found_result = TakeScreenshotUtil._run_logic(window_query=non_existent_window_query)
    print(not_found_result)
    try:
        data = json.loads(not_found_result)
        if 'error' in data:
            print("Success! Correctly reported an error for a non-existent window.")
        else:
            print("Failed. Did not report an error as expected.")
    except (json.JSONDecodeError, KeyError):
        print("Failed to parse JSON response.")