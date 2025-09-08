# utils/process_manager.py

"""
A utility for managing long-running background processes using tmux windows.
"""
import subprocess
import markpickle
from typing import Dict, Any, Literal, Optional

# For standalone testing, create a mock UtilBase if the real one isn't available.
try:
    from agent.utils_manager.util_base import UtilBase
except ImportError:
    print("Warning: Could not import UtilBase. Using a mock class for standalone testing.")
    class UtilBase:
        pass

class ProcessManager(UtilBase):
    """
    Manages long-running background processes by creating, listing, viewing, and stopping
    dedicated tmux windows. This is the standard way to run servers, daemons, or any
    task that needs to persist in the background independently of the main agent loop.
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": [
                "background process", "long-running task", "daemon", "server", "manage processes",
                "tmux", "background job", "start server", "stop process", "view logs", "process control",
                "interactive command", "yeoman", "menu selection", "keystroke", "terminal input"
            ],
            "use_cases": [
                "Start a web server in the background and continue working.",
                "Run a long compilation or data processing task without blocking the agent.",
                "List all currently running background processes.",
                "Check the console output of a running background server.",
                "Stop a background process that is no longer needed.",
                "Send input to interactive commands (menus, prompts, wizards).",
                "Control interactive processes that were moved to background due to timeout.",
                "Send keystrokes to interactive terminals like Yeoman generators.",
                "Navigate menu selections in background processes."
            ],
            "arguments": {
                "action": "The operation to perform. Must be one of: 'start', 'list', 'view', 'stop', 'send'.",
                "window_name": "A unique name for the process window (e.g., 'web-server', 'data-processor'). For interactive commands moved to background, this is provided by the timeout handler.",
                "command": "For 'start': the shell command to execute. For 'send': the input to send to the interactive process. Use empty string '' to send just Enter key. For arrow-key menus, send specific keys or just Enter to select default option."
            }
        }

    @staticmethod
    def _ensure_tmux_session():
        """Ensure we have a tmux session to work with."""
        try:
            # Check if we're already in a tmux session
            result = subprocess.run("tmux list-sessions", shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                # No sessions exist, check if we're in one by environment
                import os
                if 'TMUX' not in os.environ:
                    raise ValueError("ProcessManager requires running inside a tmux session. Please start the agent which automatically creates one.")
        except Exception:
            raise ValueError("ProcessManager requires running inside a tmux session. Please start the agent which automatically creates one.")

    @staticmethod
    def _run_logic(
        action: Literal["start", "list", "view", "stop", "send"],
        window_name: Optional[str] = None,
        command: Optional[str] = None
    ) -> str:
        """
        The core implementation for managing tmux-based background processes.
        """
        try:
            # Argument validation
            if action in ["start", "view", "stop", "send"] and not window_name:
                raise ValueError(f"'window_name' is required for the '{action}' action.")
            if action in ["start", "send"] and command is None:
                raise ValueError(f"'command' is required for the '{action}' action.")

            # Ensure we have a tmux session
            ProcessManager._ensure_tmux_session()

            # Action dispatcher
            if action == "start":
                # Check if window already exists to prevent errors
                try:
                    # The `^...` ensures an exact match on the window name
                    result = subprocess.run(f"tmux list-windows -F '#W' | grep -q '^{window_name}$'", shell=True)
                    if result.returncode == 0:
                        raise ValueError(f"A process window named '{window_name}' already exists.")
                except subprocess.CalledProcessError:
                    # Window does not exist, which is what we want. Continue.
                    pass
                
                subprocess.run(f"tmux new-window -d -n {window_name} '{command}'", shell=True, check=True)
                result = {"status": "Success", "message": f"Process '{window_name}' started in the background."}

            elif action == "list":
                output = subprocess.check_output("tmux list-windows -F '#W'", shell=True, text=True)
                windows = [line for line in output.strip().split('\n') if line and line.strip()]
                result = {"status": "Success", "running_processes": windows}

            elif action == "view":
                output = subprocess.check_output(f"tmux capture-pane -p -t {window_name}", shell=True, text=True)
                result = {"status": "Success", "window_name": window_name, "output": output.strip()}
            
            elif action == "stop":
                subprocess.run(f"tmux kill-window -t {window_name}", shell=True, check=True)
                result = {"status": "Success", "message": f"Process '{window_name}' stopped."}

            elif action == "send":
                # The C-m sequence sends the Enter key
                # Escape single quotes in the command to prevent shell issues
                escaped_command = command.replace("'", "'\"'\"'")
                subprocess.run(f"tmux send-keys -t {window_name} '{escaped_command}' C-m", shell=True, check=True)
                result = {"status": "Success", "message": f"Sent input to '{window_name}'."}

            else:
                raise ValueError(f"Invalid action '{action}'.")

            return markpickle.dumps({"result": result})

        except (subprocess.CalledProcessError, ValueError, Exception) as e:
            error_message = str(e)
            if isinstance(e, subprocess.CalledProcessError):
                error_message = e.stderr.strip() if e.stderr else (e.stdout.strip() if e.stdout else str(e))
            return markpickle.dumps({"error": f"Process manager action '{action}' failed. Reason: {error_message}"})


def run(
    action: Literal["start", "list", "view", "stop", "send"],
    window_name: Optional[str] = None,
    command: Optional[str] = None
) -> str:
    """
    Manage background processes in tmux windows. Handle interactive commands moved to background.
    
    Args:
        action: Operation to perform - 'start', 'list', 'view', 'stop', 'send'
        window_name: Unique window identifier (provided by timeout handler for interactive commands)
        command: For 'start': shell command. For 'send': input to send (use '' for Enter key)
    
    Usage Examples:
        # View an interactive command that was moved to background
        result = process_manager.run('view', window_name='interactive-08-31-49-f1519528')
        
        # Send Enter key to select default option in Yeoman menu
        process_manager.run('send', window_name='interactive-08-31-49-f1519528', command='')
        
        # Send specific text input
        process_manager.run('send', window_name='interactive-08-31-49-f1519528', command='my-extension-name')
        
        # Stop the background process when done
        process_manager.run('stop', window_name='interactive-08-31-49-f1519528')
    
    Returns:
        Serialized result dict with status and data or error information
    """
    return ProcessManager._run_logic(action=action, window_name=window_name, command=command)