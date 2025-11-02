"""
Computer control tool for CLI Agent.
Provides window management, screenshot functionality, and basic computer automation.
Works with any application window, not just browsers.
"""

import os
import re
import subprocess
import time


from typing import Optional
class computercontrol:
    """
    A tool to control computer windows and take screenshots.
    Supports window detection, activation, and screenshot capture of specific windows.
    """

    @staticmethod
    def get_delim() -> str:
        return "computercontrol"

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "computercontrol",
            "description": "Control computer windows and take screenshots - list windows, screenshot specific windows by title, activate windows",
            "example": "<computercontrol><action>screenshot</action><window>Firefox</window></computercontrol>"
        }

    @staticmethod
    def run(content: str) -> str:
        """Execute computer control commands."""
        try:
            # Parse action from content
            action_match = re.search(r'<action>(.*?)</action>', content, re.DOTALL)
            if not action_match:
                return "Error: No action specified. Available actions: screenshot, list_windows, activate_window"

            action = action_match.group(1).strip()

            # Parse optional window parameter
            window_match = re.search(r'<window>(.*?)</window>', content, re.DOTALL)
            window_name = window_match.group(1).strip() if window_match else None

            # Parse optional file parameter for screenshot
            file_match = re.search(r'<file>(.*?)</file>', content, re.DOTALL)
            filename = file_match.group(1).strip() if file_match else None

            if action == 'screenshot':
                return computercontrol._take_screenshot(window_name, filename)
            elif action == 'list_windows':
                return computercontrol._list_windows()
            elif action == 'activate_window':
                if not window_name:
                    return "Error: Window name required for activate_window action"
                return computercontrol._activate_window(window_name)
            else:
                return f"Error: Unknown action '{action}'. Available: screenshot, list_windows, activate_window"

        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _get_window_list() -> list[dict[str, str]]:
        """Get list of all open windows using multiple methods."""
        windows = []

        # Method 1: Try wmctrl
        try:
            result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(None, 3)
                    if len(parts) >= 4:
                        window_id = parts[0]
                        desktop = parts[1]
                        hostname = parts[2]
                        title = parts[3]
                        windows.append({
                            'id': window_id,
                            'title': title,
                            'method': 'wmctrl'
                        })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Method 2: Try xdotool if wmctrl failed or found nothing
        if not windows:
            try:
                # Get all window IDs
                result = subprocess.run(['xdotool', 'search', '--onlyvisible', '.'],
                                      capture_output=True, text=True, check=True)
                window_ids = result.stdout.strip().split('\n')

                for window_id in window_ids:
                    if window_id.strip():
                        try:
                            # Get window title
                            title_result = subprocess.run(['xdotool', 'getwindowname', window_id],
                                                        capture_output=True, text=True, check=True)
                            title = title_result.stdout.strip()
                            if title:
                                windows.append({
                                    'id': window_id,
                                    'title': title,
                                    'method': 'xdotool'
                                })
                        except subprocess.CalledProcessError:
                            continue
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        # Method 3: Try Python libraries if available
        if not windows:
            try:
                import pygetwindow as gw
                for window in gw.getAllWindows():
                    if window.title.strip():
                        windows.append({
                            'id': str(window._hWnd) if hasattr(window, '_hWnd') else 'unknown',
                            'title': window.title,
                            'method': 'pygetwindow'
                        })
            except ImportError:
                pass

        return windows

    @staticmethod
    def _find_window_by_name(window_name: str) -> Optional[dict[str, str]]:
        """Find a window by partial title match (case-insensitive)."""
        windows = computercontrol._get_window_list()
        window_name_lower = window_name.lower()

        # First try exact match
        for window in windows:
            if window['title'].lower() == window_name_lower:
                return window

        # Then try partial match
        for window in windows:
            if window_name_lower in window['title'].lower():
                return window

        return None

    @staticmethod
    def _get_window_geometry(window_info: dict[str, str]) -> Optional[tuple[int, int, int, int]]:
        """Get window geometry (x, y, width, height) using multiple methods."""
        window_id = window_info['id']
        method = window_info['method']

        try:
            if method == 'wmctrl':
                # Use wmctrl to get geometry
                result = subprocess.run(['wmctrl', '-lG'], capture_output=True, text=True, check=True)
                for line in result.stdout.strip().split('\n'):
                    if line.startswith(window_id):
                        parts = line.split()
                        if len(parts) >= 7:
                            x, y, width, height = map(int, parts[2:6])
                            return (x, y, width, height)

            elif method == 'xdotool':
                # Use xdotool to get geometry
                result = subprocess.run(['xdotool', 'getwindowgeometry', '--shell', window_id],
                                      capture_output=True, text=True, check=True)
                geometry = {}
                for line in result.stdout.strip().split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        geometry[key] = int(value)

                if all(k in geometry for k in ['X', 'Y', 'WIDTH', 'HEIGHT']):
                    return (geometry['X'], geometry['Y'], geometry['WIDTH'], geometry['HEIGHT'])

            elif method == 'pygetwindow':
                # Try to get window by title again for geometry
                try:
                    import pygetwindow as gw
                    windows = gw.getWindowsWithTitle(window_info['title'])
                    if windows:
                        window = windows[0]
                        return (window.left, window.top, window.width, window.height)
                except ImportError:
                    pass

        except (subprocess.CalledProcessError, ValueError):
            pass

        return None

    @staticmethod
    def _take_screenshot(window_name: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Take a screenshot of the entire screen or a specific window."""
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = int(time.time())
                if window_name:
                    safe_name = re.sub(r'[^\w\-_]', '_', window_name)
                    filename = f"/tmp/screenshot_{safe_name}_{timestamp}.png"
                else:
                    filename = f"/tmp/screenshot_{timestamp}.png"

            success = False

            if window_name:
                # Try to take screenshot of specific window
                window_info = computercontrol._find_window_by_name(window_name)
                if not window_info:
                    return f"Error: Window '{window_name}' not found. Use list_windows to see available windows."

                # Method 1: Try Python libraries for window screenshot
                try:
                    import pyscreenshot as ImageGrab
                    geometry = computercontrol._get_window_geometry(window_info)
                    if geometry:
                        x, y, width, height = geometry
                        # Take screenshot of the window region
                        im = ImageGrab.grab(bbox=(x, y, x + width, y + height))
                        im.save(filename)
                        success = True
                except ImportError:
                    pass

                # Method 2: Try system tools for window screenshot
                if not success:
                    try:
                        # Try to activate window first
                        computercontrol._activate_window(window_name)
                        time.sleep(0.5)  # Give time for activation

                        # Try gnome-screenshot with window selection
                        result = subprocess.run(['gnome-screenshot', '-w', '-f', filename],
                                              capture_output=True, check=True)
                        success = True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        pass

                # Method 3: Try scrot with window selection
                if not success:
                    try:
                        window_id = window_info['id']
                        if window_info['method'] in ['wmctrl', 'xdotool']:
                            subprocess.run(['scrot', '-s', filename], check=True)
                            success = True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        pass

            else:
                # Take full screen screenshot
                screenshot_tools = [
                    ['gnome-screenshot', '-f', filename],
                    ['scrot', filename],
                    ['import', '-window', 'root', filename],  # ImageMagick
                    ['xwd', '-root', '-out', filename]
                ]

                for tool_cmd in screenshot_tools:
                    try:
                        subprocess.run(tool_cmd, check=True, capture_output=True)
                        success = True
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue

                # Try Python libraries as fallback
                if not success:
                    try:
                        import pyscreenshot as ImageGrab
                        im = ImageGrab.grab()
                        im.save(filename)
                        success = True
                    except ImportError:
                        pass

            if success and os.path.exists(filename):
                size = os.path.getsize(filename)
                if window_name:
                    return f"Screenshot of '{window_name}' saved to {filename} ({size} bytes)"
                else:
                    return f"Full screen screenshot saved to {filename} ({size} bytes)"
            else:
                return "Error: Failed to take screenshot. Install gnome-screenshot, scrot, or imagemagick."

        except Exception as e:
            return f"Screenshot error: {str(e)}"

    @staticmethod
    def _list_windows() -> str:
        """List all open windows."""
        try:
            windows = computercontrol._get_window_list()

            if not windows:
                return "No windows found. Install wmctrl, xdotool, or pygetwindow for window detection."

            # Format window list
            result = f"Found {len(windows)} open windows:\n"
            for i, window in enumerate(windows, 1):
                title = window['title'][:60] + "..." if len(window['title']) > 60 else window['title']
                result += f"{i:2d}. {title}\n"

            result += "\nTo screenshot a specific window, use:\n<computercontrol><action>screenshot</action><window>Window Title</window></computercontrol>"
            return result

        except Exception as e:
            return f"Error listing windows: {str(e)}"

    @staticmethod
    def _activate_window(window_name: str) -> str:
        """Activate/focus a specific window."""
        try:
            window_info = computercontrol._find_window_by_name(window_name)
            if not window_info:
                return f"Error: Window '{window_name}' not found"

            window_id = window_info['id']
            method = window_info['method']

            # Try different activation methods
            success = False

            if method == 'wmctrl':
                try:
                    subprocess.run(['wmctrl', '-ia', window_id], check=True)
                    success = True
                except subprocess.CalledProcessError:
                    pass

            if not success and method == 'xdotool':
                try:
                    subprocess.run(['xdotool', 'windowactivate', window_id], check=True)
                    success = True
                except subprocess.CalledProcessError:
                    pass

            if not success:
                # Try Python libraries
                try:
                    import pygetwindow as gw
                    windows = gw.getWindowsWithTitle(window_info['title'])
                    if windows:
                        windows[0].activate()
                        success = True
                except ImportError:
                    pass

            if success:
                return f"Activated window: {window_info['title']}"
            else:
                return f"Error: Could not activate window '{window_name}'"

        except Exception as e:
            return f"Error activating window: {str(e)}"