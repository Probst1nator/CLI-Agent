#!/usr/bin/env python3

from datetime import datetime
import json
import os
import sys
import tempfile
import time
from typing import Tuple, Any, Optional, Dict, Callable, List
import jupyter_client
from jupyter_client.kernelspec import KernelSpecManager
import threading
from queue import Queue, Empty
import logging
import subprocess

# Import pexpect for interactive shell command support
try:
    import pexpect
    PEXPECT_AVAILABLE = True
except ImportError:
    PEXPECT_AVAILABLE = False
    logging.warning("pexpect not available. Interactive shell commands will not work. Install with: pip install pexpect")

from py_classes.globals import Globals, g
logger = logging.getLogger(__name__)

class PythonSandbox:
    """
    A unified Python sandbox for executing both Python code and shell commands with full interactive support.
    This provides process isolation, maintains state across executions, and automatically handles user input.
    
    Key Features:
    - **Unified execution**: One execute() method handles both Python code and shell commands (prefix with !)
    - **Interactive support**: Automatically handles input prompts with configurable callbacks
    - **Environment consistency**: Shell commands use the same virtual environment as Python code
    - **Timeout management**: Configurable timeouts with helpful user hints
    - **Real-time streaming**: Live output via callbacks with proper interrupt handling
    
    The sandbox automatically detects command type:
    - Python code: Executed in persistent Jupyter kernel with global state preservation
    - Shell commands: Commands starting with ! are executed with full interactivity support
    
    Usage Examples:
    
    # Basic Python execution
    sandbox = PythonSandbox()
    stdout, stderr, result = sandbox.execute("print('Hello World')")
    
    # Shell command with automatic interaction handling
    stdout, stderr, exit_code = sandbox.execute(
        "!curl -fsSL https://example.com/script.sh | sudo bash",
        input_callback=lambda prompt: "y"  # Auto-answer prompts
    )
    
    # Python code with input handling
    stdout, stderr, result = sandbox.execute(
        "name = input('Enter name: '); print(f'Hello {name}')",
        input_callback=lambda prompt: "Alice"
    )
    
    # Long-running operation with timeout
    stdout, stderr, result = sandbox.execute(
        "!apt update && apt install -y docker.io",
        timeout=300,  # 5 minute timeout
        show_timeout_hints=True  # Show progress hints
    )
    
    Environment Consistency:
    The sandbox ensures shell commands run in the same environment as Python:
    - Virtual environment activation is automatic
    - pip installs affect the same Python environment
    - Environment variables are shared between Python and shell
    """
    
    def __init__(self) -> None:
        """
        Initialize the Python sandbox with a Jupyter kernel.
        
        Args:
            cwd: Optional current working directory to use for resolving relative paths
        """
        # Flag to prevent recursive calls during initialization
        self._initializing = True
        
        self.ensure_kernel_available()
        if g.USE_SANDBOX:
            self.cwd = g.AGENTS_SANDBOX_DIR
        else:
            self.cwd = os.getcwd()
        self._start_kernel()
        
        # Initialization complete
        self._initializing = False
        
    def ensure_kernel_available(self) -> None:
        """Ensure the Python kernel is available and install required dependencies."""
        ksm = KernelSpecManager()
        if 'python3' not in ksm.find_kernel_specs():
            logger.warning("Python3 kernel not found. Attempting to install it.")
            try:
                import ipykernel
                ipykernel.kernelspec.install(user=True)
                logger.info("Successfully installed Python3 kernel.")
            except Exception as e:
                logger.error(f"Failed to install Python3 kernel: {str(e)}")
                raise RuntimeError("Cannot initialize Python sandbox: Python3 kernel not available")
        
        # Install additional packages during sandbox initialization
        try:
            # Check if ipywidgets is installed by importing it
            import importlib.util
            if importlib.util.find_spec('ipywidgets') is None:
                logger.info("Installing ipywidgets...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ipywidgets"])
                logger.info("Successfully installed ipywidgets.")
        except Exception as e:
            # Just log the error but don't fail - this is not critical
            logger.warning(f"Failed to install ipywidgets: {str(e)}. Some warnings may appear.")
    
    def _execute_initialization_code(self) -> None:
        """Execute initialization code for the sandbox without attempting to sync globals."""
        # Execute initialization code and set working directory
        self._raw_execute("import sys, os, json, io, traceback, subprocess, threading")
        self._raw_execute(f"os.chdir('{self.cwd}')")
        self._raw_execute("print(f'Python sandbox initialized with working directory: {os.getcwd()}')")
        
        # Capture current environment information for shell consistency
        self._raw_execute("""
# Store environment information for shell command consistency
import sys
import os
import subprocess

# Get current Python environment details
SANDBOX_PYTHON_EXECUTABLE = sys.executable
SANDBOX_PYTHON_PATH = sys.path.copy()
SANDBOX_VIRTUAL_ENV = os.environ.get('VIRTUAL_ENV', '')
SANDBOX_CONDA_DEFAULT_ENV = os.environ.get('CONDA_DEFAULT_ENV', '')
SANDBOX_PATH = os.environ.get('PATH', '')

# Detect if we're in a virtual environment
def detect_virtual_env():
    if SANDBOX_VIRTUAL_ENV:
        return 'venv', SANDBOX_VIRTUAL_ENV
    elif SANDBOX_CONDA_DEFAULT_ENV:
        return 'conda', SANDBOX_CONDA_DEFAULT_ENV
    elif hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return 'venv', sys.prefix
    else:
        return 'system', ''

SANDBOX_ENV_TYPE, SANDBOX_ENV_PATH = detect_virtual_env()

# Enhanced shell execution function that maintains environment consistency
def run_shell_command_with_env(command, capture_output=True, text=True, check=False, **kwargs):
    '''
    Execute shell command with the same environment as current Python kernel.
    This ensures pip installs and other environment changes are consistent.
    '''
    import subprocess
    import os
    
    # Prepare environment
    env = os.environ.copy()
    
    # Ensure Python executable path is in PATH
    if SANDBOX_PYTHON_EXECUTABLE:
        python_dir = os.path.dirname(SANDBOX_PYTHON_EXECUTABLE)
        current_path = env.get('PATH', '')
        if python_dir not in current_path:
            env['PATH'] = f"{python_dir}:{current_path}"
    
    # Set virtual environment variables
    if SANDBOX_VIRTUAL_ENV:
        env['VIRTUAL_ENV'] = SANDBOX_VIRTUAL_ENV
        venv_bin = os.path.join(SANDBOX_VIRTUAL_ENV, 'bin')
        if venv_bin not in env['PATH']:
            env['PATH'] = f"{venv_bin}:{env['PATH']}"
    
    if SANDBOX_CONDA_DEFAULT_ENV:
        env['CONDA_DEFAULT_ENV'] = SANDBOX_CONDA_DEFAULT_ENV
    
    # Set PYTHONPATH
    if SANDBOX_PYTHON_PATH:
        env['PYTHONPATH'] = ':'.join(SANDBOX_PYTHON_PATH)
    
    # Prepare activation command
    activation_cmd = ''
    if SANDBOX_ENV_TYPE == 'conda' and SANDBOX_CONDA_DEFAULT_ENV:
        activation_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {SANDBOX_CONDA_DEFAULT_ENV} && "
    elif SANDBOX_ENV_TYPE == 'venv' and SANDBOX_VIRTUAL_ENV:
        activate_script = os.path.join(SANDBOX_VIRTUAL_ENV, 'bin', 'activate')
        if os.path.exists(activate_script):
            activation_cmd = f"source {activate_script} && "
    
    # Execute command with proper environment
    full_command = f"bash -c '{activation_cmd}{command}'" if activation_cmd else command
    
    return subprocess.run(full_command, env=env, capture_output=capture_output, 
                         text=text, check=check, shell=True, **kwargs)

# Make the enhanced function available globally in the kernel
globals()['run_shell_command_with_env'] = run_shell_command_with_env

print(f'Environment type: {SANDBOX_ENV_TYPE}')
if SANDBOX_ENV_PATH:
    print(f'Environment path: {SANDBOX_ENV_PATH}')
print(f'Python executable: {SANDBOX_PYTHON_EXECUTABLE}')
print('Enhanced shell command execution available via run_shell_command_with_env()')
""")
        
        # Suppress tqdm warnings about ipywidgets
        self._raw_execute("import warnings")
        self._raw_execute("warnings.filterwarnings('ignore', category=UserWarning, module='tqdm')")
        
        # Add project directories to Python path to access project modules
        self._raw_execute(f"sys.path.append('{g.PROJ_DIR_PATH}')")  # Add main project dir
        self._raw_execute(f"sys.path.append('{os.path.dirname(g.PROJ_DIR_PATH)}')")  # Add parent dir
        
        # Import common modules
        self._raw_execute("import os, pathlib")
        self._raw_execute("from utils import *")
        self._raw_execute("home_dir = str(pathlib.Path.home())")  # Make home dir available
        
        # Import g and json (required for sync_globals)
        self._raw_execute("from py_classes.globals import g")
        self._raw_execute("import json")
        
        # Import utils
        self._raw_execute("from utils import *")
        
        # Store environment info for shell commands
        self._get_environment_info()
    
    def _get_environment_info(self) -> None:
        """Get environment information from the kernel for shell command consistency."""
        try:
            # Get environment information from the kernel
            stdout, stderr, result = self._raw_execute("""
import json
env_info = {
    'python_executable': SANDBOX_PYTHON_EXECUTABLE,
    'virtual_env': SANDBOX_VIRTUAL_ENV,
    'conda_env': SANDBOX_CONDA_DEFAULT_ENV,
    'env_type': SANDBOX_ENV_TYPE,
    'env_path': SANDBOX_ENV_PATH,
    'path': SANDBOX_PATH,
    'python_path': SANDBOX_PYTHON_PATH
}
print(json.dumps(env_info))
""")
            
            if stdout.strip():
                try:
                    self.env_info = json.loads(stdout.strip().split('\n')[-1])
                    logger.info(f"Detected environment: {self.env_info['env_type']}")
                    if self.env_info['env_path']:
                        logger.info(f"Environment path: {self.env_info['env_path']}")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse environment info, using defaults")
                    self.env_info = self._get_default_env_info()
            else:
                self.env_info = self._get_default_env_info()
                
        except Exception as e:
            logger.warning(f"Failed to get environment info: {e}")
            self.env_info = self._get_default_env_info()
    
    def _get_default_env_info(self) -> Dict[str, Any]:
        """Get default environment information."""
        return {
            'python_executable': sys.executable,
            'virtual_env': os.environ.get('VIRTUAL_ENV', ''),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV', ''),
            'env_type': 'system',
            'env_path': '',
            'path': os.environ.get('PATH', ''),
            'python_path': sys.path
        }
    
    def _prepare_shell_environment(self) -> Dict[str, str]:
        """Prepare environment variables for shell commands to match Python environment."""
        env = os.environ.copy()
        
        # Ensure we use the same Python executable
        if self.env_info.get('python_executable'):
            python_dir = os.path.dirname(self.env_info['python_executable'])
            
            # Add Python directory to PATH if not already there
            current_path = env.get('PATH', '')
            if python_dir not in current_path:
                env['PATH'] = f"{python_dir}:{current_path}"
        
        # Set virtual environment variables if applicable
        if self.env_info.get('virtual_env'):
            env['VIRTUAL_ENV'] = self.env_info['virtual_env']
            # Activate virtual environment in PATH
            venv_bin = os.path.join(self.env_info['virtual_env'], 'bin')
            if venv_bin not in env['PATH']:
                env['PATH'] = f"{venv_bin}:{env['PATH']}"
        
        if self.env_info.get('conda_env'):
            env['CONDA_DEFAULT_ENV'] = self.env_info['conda_env']
        
        # Set PYTHONPATH to match kernel's sys.path
        if self.env_info.get('python_path'):
            env['PYTHONPATH'] = ':'.join(self.env_info['python_path'])
        
        return env
    
    def _get_shell_activation_command(self) -> str:
        """Get shell commands to activate the same environment as Python kernel."""
        activation_commands = []
        
        if self.env_info.get('env_type') == 'conda' and self.env_info.get('conda_env'):
            # For conda environments
            activation_commands.append(f"source $(conda info --base)/etc/profile.d/conda.sh")
            activation_commands.append(f"conda activate {self.env_info['conda_env']}")
        elif self.env_info.get('env_type') == 'venv' and self.env_info.get('virtual_env'):
            # For virtual environments
            activate_script = os.path.join(self.env_info['virtual_env'], 'bin', 'activate')
            if os.path.exists(activate_script):
                activation_commands.append(f"source {activate_script}")
        
        return ' && '.join(activation_commands) if activation_commands else ''
    
    def execute_shell_command_with_env(self, command: str,
                                     stdout_callback: Optional[Callable[[str], None]] = None,
                                     stderr_callback: Optional[Callable[[str], None]] = None,
                                     timeout: int = 300,
                                     show_timeout_hints: bool = True) -> Tuple[str, str, int]:
        """
        Execute a shell command using the same environment as the Python kernel.
        This is a simpler alternative to execute_interactive_shell_command for non-interactive commands.
        
        Args:
            command: Shell command to execute
            stdout_callback: Optional callback function to receive stdout in real-time
            stderr_callback: Optional callback function to receive stderr in real-time
            timeout: Maximum timeout in seconds for the entire command (default: 300 = 5 minutes)
            show_timeout_hints: Whether to show timeout hints after 30 seconds (default: True)
            
        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        try:
            # Prepare environment to match Python kernel
            env = self._prepare_shell_environment()
            
            # Prepare the command with environment activation
            activation_cmd = self._get_shell_activation_command()
            if activation_cmd:
                full_command = f"bash -c '{activation_cmd} && {command}'"
            else:
                full_command = command
            
            logger.debug(f"Executing shell command with environment: {full_command}")
            
            # Execute using subprocess with proper environment
            process = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.cwd,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_content = []
            stderr_content = []
            start_time = time.time()
            last_output_time = time.time()
            timeout_hint_shown = False
            
            try:
                # Read output in real-time with timeout handling
                while True:
                    try:
                        # Check if process has finished
                        if process.poll() is not None:
                            # Read any remaining output
                            remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                            if remaining_stdout:
                                stdout_content.append(remaining_stdout)
                                if stdout_callback:
                                    stdout_callback(remaining_stdout)
                            if remaining_stderr:
                                stderr_content.append(remaining_stderr)
                                if stderr_callback:
                                    stderr_callback(remaining_stderr)
                            break
                        
                        # Read available output (non-blocking)
                        stdout_line = ""
                        stderr_line = ""
                        
                        # Try to read stdout
                        try:
                            stdout_line = process.stdout.readline()
                            if stdout_line:
                                stdout_content.append(stdout_line)
                                if stdout_callback:
                                    stdout_callback(stdout_line)
                                last_output_time = time.time()
                                timeout_hint_shown = False
                        except:
                            pass
                        
                        # Try to read stderr
                        try:
                            stderr_line = process.stderr.readline()
                            if stderr_line:
                                stderr_content.append(stderr_line)
                                if stderr_callback:
                                    stderr_callback(stderr_line)
                                last_output_time = time.time()
                                timeout_hint_shown = False
                        except:
                            pass
                        
                        # Check timeout conditions
                        current_time = time.time()
                        total_elapsed = current_time - start_time
                        elapsed_since_output = current_time - last_output_time
                        
                        # Show timeout hint if no output for 30 seconds
                        if show_timeout_hints and elapsed_since_output > 30 and not timeout_hint_shown:
                            hint_msg = (
                                f"\n⏱️  Command has been running for {total_elapsed:.0f} seconds with no output.\n"
                                f"💡 You can:\n"
                                f"   • Wait for the command to complete\n"
                                f"   • Press Ctrl+C to abort the command\n"
                                f"   • The command will timeout after {timeout} seconds total\n"
                                f"🔄 Continuing to wait...\n"
                            )
                            if stdout_callback:
                                stdout_callback(hint_msg)
                            else:
                                print(hint_msg, flush=True)
                            timeout_hint_shown = True
                        
                        # Check total timeout
                        if total_elapsed > timeout:
                            timeout_msg = f"\n⏰ Command timed out after {timeout} seconds. Terminating...\n"
                            if stderr_callback:
                                stderr_callback(timeout_msg)
                            else:
                                print(timeout_msg, flush=True)
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            return ''.join(stdout_content), ''.join(stderr_content) + "Command timed out", 124  # Standard timeout exit code
                        
                        # Small sleep to prevent busy waiting
                        time.sleep(0.1)
                    
                    except Exception as e:
                        logger.warning(f"Error reading process output: {e}")
                        break
                        
            except KeyboardInterrupt:
                interrupt_msg = (
                    f"\n🛑 KeyboardInterrupt received (Ctrl+C).\n"
                    f"🔄 Terminating command: {command[:50]}{'...' if len(command) > 50 else ''}\n"
                )
                if stderr_callback:
                    stderr_callback(interrupt_msg)
                else:
                    print(interrupt_msg, flush=True)
                
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                return ''.join(stdout_content), ''.join(stderr_content) + "Command interrupted by user (Ctrl+C)", 130  # Standard exit code for Ctrl+C
            
            exit_code = process.returncode
            total_time = time.time() - start_time
            completion_msg = f"\n✅ Command completed in {total_time:.1f} seconds with exit code: {exit_code}\n"
            logger.debug(completion_msg.strip())
            
            return ''.join(stdout_content), ''.join(stderr_content), exit_code
            
        except Exception as e:
            error_msg = f"Error executing shell command with environment: {str(e)}"
            logger.error(error_msg)
            return "", error_msg, 1
    
    def _start_kernel(self) -> None:
        """Start a new Jupyter kernel."""
        # Start the kernel
        self.km = jupyter_client.KernelManager(kernel_name='python3')
        self.km.start_kernel()
        
        # Create a client
        self.kc = self.km.client()
        self.kc.start_channels()
        
        # Wait for kernel to be ready
        self.kc.wait_for_ready(timeout=30)
        logger.info("Python sandbox kernel started successfully")
        
        # Execute initialization code without attempting to sync globals
        self._execute_initialization_code()
        
        # Now that everything is initialized, sync the globals
        self.sync_globals()
    
    def _raw_execute(self, code: str) -> Tuple[str, str, Any]:
        """
        Execute code without syncing globals - used during initialization.
        This is a stripped-down version of execute() that doesn't call sync_globals().
        """
        try:
            # Send code to the kernel
            msg_id = self.kc.execute(code)
            
            # Get the reply
            try:
                reply = self.kc.get_shell_msg(timeout=30)
                if reply['content']['status'] == 'error':
                    error_msg = f"Error executing code: {reply['content']['ename']}: {reply['content']['evalue']}"
                    logging.error(error_msg)
            except Exception as e:
                logging.error(f"Error getting shell message: {str(e)}")
            
            # Collect output (simplified version without callbacks)
            # Use None as the timeout to prevent automatic termination
            return self._collect_output(timeout=None, show_timeout_hints=False)
        except Exception as e:
            logging.error(f"Error in _raw_execute: {str(e)}")
            return "", str(e), None
    
    def sync_globals(self) -> None:
        """
        Synchronize all globals from the main runtime to the sandbox runtime.
        This keeps the sandbox's g object in sync with the main runtime's g object.
        """
        # Skip if we're still initializing
        if hasattr(self, '_initializing') and self._initializing:
            return
        
        # Get all attributes of the g object that are not callables or private
        globals_to_sync = []
        for attr_name in dir(g):
            # Skip private attributes, methods, and builtins
            if attr_name.startswith('_') or callable(getattr(g, attr_name)):
                continue
            
            attr_value = getattr(g, attr_name)
            # Try to serialize the value to ensure it can be passed to the sandbox
            try:
                # For primitive types, just convert to string
                if isinstance(attr_value, (bool, int, float, str, type(None))):
                    globals_to_sync.append((attr_name, attr_value))
                # For lists and dicts, use JSON to ensure serializability
                elif isinstance(attr_value, (list, dict)):
                    json.dumps(attr_value)  # This will raise an exception if not serializable
                    globals_to_sync.append((attr_name, attr_value))
                # Skip other types that may not be serializable
            except (TypeError, ValueError):
                logger.debug(f"Skipping global '{attr_name}' as it's not serializable")
        
        # Now synchronize each global
        for attr_name, attr_value in globals_to_sync:
            try:
                if isinstance(attr_value, (bool, int, float, type(None))):
                    self._raw_execute(f"g.{attr_name} = {attr_value}")
                elif isinstance(attr_value, str):
                    # Escape single quotes for string values
                    escaped_value = attr_value.replace("'", "\\'")
                    self._raw_execute(f"g.{attr_name} = '{escaped_value}'")
                elif isinstance(attr_value, (list, dict)):
                    # Convert complex types to JSON
                    json_str = json.dumps(attr_value)
                    self._raw_execute(f"g.{attr_name} = json.loads('''{json_str}''')")
            except Exception as e:
                logger.warning(f"Failed to sync global '{attr_name}': {str(e)}")
        
    def _collect_output(self, timeout: Optional[float] = None, stdout_callback: Optional[Callable[[str], None]] = None, 
                       stderr_callback: Optional[Callable[[str], None]] = None,
                       input_callback: Optional[Callable[[str], str]] = None,
                       show_timeout_hints: bool = True) -> Tuple[str, str, Any]:
        """
        Collect all output from the kernel execution.
        
        Args:
            timeout: Maximum time to wait for execution completion (seconds), None means wait indefinitely
            stdout_callback: Optional callback function to handle stdout in real-time
            stderr_callback: Optional callback function to receive stderr in real-time
            input_callback: Optional callback function to handle input requests
            show_timeout_hints: Whether to show timeout hints after 30 seconds
            
        Returns:
            Tuple of (stdout, stderr, result)
        """
        stdout_content = []
        stderr_content = []
        seen_outputs = set()  # Track outputs we've seen to prevent duplication
        result = None
        error = None
        execution_completed = False
        
        # Only set up timeout tracking if timeout is actually specified
        if timeout is not None:
            start_time = time.time()
            last_message_time = time.time()
            timeout_hint_shown = False
        
        try:
            while True:
                try:
                    msg = self.kc.get_iopub_msg(timeout=0.1)
                    if timeout is not None:
                        last_message_time = time.time()  # Update last message time
                        timeout_hint_shown = False  # Reset hint flag on any message
                    
                    msg_type = msg['header']['msg_type']
                    content = msg['content']
                    
                    if msg_type == 'stream':
                        if content['name'] == 'stderr':
                            # Prioritize stderr
                            error_text = content['text']
                            # Add to stderr regardless of whether seen before
                            stderr_content.append(error_text)
                            # Mark as seen
                            seen_outputs.add(error_text)
                            if stderr_callback:
                                stderr_callback(error_text)
                        elif content['name'] == 'stdout':
                            output_text = content['text']
                            # Only add to stdout if not already in stderr
                            if output_text not in seen_outputs:
                                stdout_content.append(output_text)
                                seen_outputs.add(output_text)
                                if stdout_callback:
                                    stdout_callback(output_text)
                            
                    elif msg_type == 'execute_result':
                        result = content['data'].get('text/plain', '')
                        
                    elif msg_type == 'display_data':
                        result = content['data'].get('text/plain', '')
                        
                    elif msg_type == 'error':
                        error_name = content['ename']
                        error_value = content['evalue']
                        error_traceback = '\n'.join(content['traceback'])
                        error = f"{error_name}: {error_value}\n{error_traceback}"
                        # Add to seen outputs to prevent duplication
                        seen_outputs.add(error)
                        if stderr_callback:
                            stderr_callback(error)
                        
                    elif msg_type == 'status' and content['execution_state'] == 'idle':
                        # Kernel has finished processing
                        execution_completed = True
                        break
                        
                except Empty:
                    # Check for input requests on stdin channel
                    try:
                        stdin_msg = self.kc.get_stdin_msg(timeout=0.01)
                        if stdin_msg['header']['msg_type'] == 'input_request':
                            prompt = stdin_msg['content']['prompt']
                            password = stdin_msg['content'].get('password', False)
                            
                            # Handle input request
                            if input_callback:
                                user_input = input_callback(prompt)
                            else:
                                # Default behavior: print prompt and wait for input
                                if stdout_callback:
                                    stdout_callback(prompt)
                                user_input = input()  # This will block until user provides input
                            
                            # Send the input reply
                            self.kc.input(user_input)
                            if timeout is not None:
                                last_message_time = time.time()  # Reset timeout
                            continue
                            
                    except Empty:
                        pass
                    
                    # Only do timeout checking if timeout is set
                    if timeout is not None:
                        current_time = time.time()
                        total_elapsed = current_time - start_time
                        elapsed_since_message = current_time - last_message_time
                        
                        # Only show timeout hint if explicitly enabled and we've been waiting a while
                        if (show_timeout_hints and elapsed_since_message > 30 and not timeout_hint_shown):
                            hint_msg = (
                                f"\n⏱️  Python execution has been running for {total_elapsed:.0f} seconds with no output.\n"
                                f"💡 You can:\n"
                                f"   • Wait for the execution to complete\n"
                                f"   • Press Ctrl+C to interrupt the execution\n"
                                f"   • The execution will timeout after {timeout} seconds total\n"
                                f"🔄 Continuing to wait...\n"
                            )
                            
                            if stdout_callback:
                                stdout_callback(hint_msg)
                            else:
                                print(hint_msg, flush=True)
                            timeout_hint_shown = True
                        
                        # Check if we've hit the total timeout
                        if total_elapsed > timeout:
                            timeout_msg = f"\n⏰ Python execution timed out after {timeout} seconds. Interrupting...\n"
                            if stderr_callback:
                                stderr_callback(timeout_msg)
                            else:
                                print(timeout_msg, flush=True)
                            
                            # Try to interrupt the kernel
                            try:
                                self.km.interrupt_kernel()
                            except Exception as e:
                                logger.warning(f"Failed to interrupt kernel: {e}")
                            
                            return ''.join(stdout_content), ''.join(stderr_content) + "Python execution timed out", None
                    
                    continue
            
            # Show completion message only for long-running executions with timeout enabled
            if timeout is not None and show_timeout_hints:
                total_time = time.time() - start_time
                if total_time > 30:  # Only show for executions longer than 30 seconds
                    completion_msg = f"\n✅ Python execution completed in {total_time:.1f} seconds\n"
                    logger.debug(completion_msg.strip())
            
            return ''.join(stdout_content), ''.join(stderr_content), result if not error else error
            
        except KeyboardInterrupt:
            interrupt_msg = (
                f"\n🛑 KeyboardInterrupt received during output collection.\n"
                f"🔄 Interrupting Python execution...\n"
            )
            if stderr_callback:
                stderr_callback(interrupt_msg)
            else:
                print(interrupt_msg, flush=True)
            
            # Try to interrupt the kernel
            try:
                self.km.interrupt_kernel()
            except Exception as e:
                logger.warning(f"Failed to interrupt kernel: {e}")
            
            return ''.join(stdout_content), ''.join(stderr_content) + "Execution interrupted by user (Ctrl+C)", None
    
    def execute(self, code: str, stdout_callback: Optional[Callable[[str], None]] = None,
               stderr_callback: Optional[Callable[[str], None]] = None, 
               input_callback: Optional[Callable[[str], str]] = None,
               timeout: Optional[float] = None,
               show_timeout_hints: bool = True) -> Tuple[str, str, Any]:
        """
        Execute the given code in the kernel and return stdout, stderr, and result.
        Automatically handles both Python code and shell commands with full interactive support.
        
        Args:
            code: Python code or shell command to execute (shell commands start with !)
            stdout_callback: Optional callback function to receive stdout in real-time
            stderr_callback: Optional callback function to receive stderr in real-time
            input_callback: Optional callback function to handle input requests (receives prompt, returns input)
            timeout: Maximum time in seconds for the entire execution (None for no limit)
            show_timeout_hints: Whether to show timeout hints after 30 seconds for long operations
            
        Returns:
            Tuple of (stdout, stderr, result) - for shell commands, result is the exit code
        """
        # Detect if this is a shell command
        if code.strip().startswith('!'):
            # Extract the shell command (remove the !)
            shell_command = code.strip()[1:].strip()
            return self._execute_shell_command(shell_command, stdout_callback, stderr_callback, 
                                             input_callback, timeout, show_timeout_hints)
        else:
            # Execute as Python code
            return self._execute_python_code(code, stdout_callback, stderr_callback,
                                           input_callback, timeout, show_timeout_hints)
    
    def _execute_shell_command(self, command: str, 
                              stdout_callback: Optional[Callable[[str], None]] = None,
                              stderr_callback: Optional[Callable[[str], None]] = None,
                              input_callback: Optional[Callable[[str], str]] = None,
                              timeout: Optional[float] = None,
                              show_timeout_hints: bool = True) -> Tuple[str, str, int]:
        """Execute a shell command with full interactive support and environment consistency."""
        
        # Set reasonable default timeout for shell commands
        if timeout is None:
            timeout = 60  # 1 minute default for shell commands (reduced from 5 minutes)
        
        # For simple commands, try subprocess first (faster and more reliable)
        simple_commands = ['which', 'ls', 'pwd', 'echo', 'cat', 'grep', 'find', 'ps', 'whoami', 'id', 'date', 'uname']
        command_parts = command.split()
        is_simple_command = (
            len(command_parts) > 0 and 
            command_parts[0] in simple_commands and
            'sudo' not in command and
            '|' not in command and
            '&&' not in command and
            '||' not in command and
            input_callback is None  # No input callback needed
        )
        
        if is_simple_command:
            try:
                # Use simple subprocess for basic commands
                env = self._prepare_shell_environment()
                activation_cmd = self._get_shell_activation_command()
                
                if activation_cmd:
                    full_command = f"bash -c '{activation_cmd} && {command}'"
                else:
                    full_command = command
                
                logger.debug(f"Executing simple shell command: {full_command}")
                
                process = subprocess.run(
                    full_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.cwd,
                    env=env,
                    timeout=timeout
                )
                
                stdout_output = process.stdout or ""
                stderr_output = process.stderr or ""
                
                # Call callbacks if provided
                if stdout_callback and stdout_output:
                    stdout_callback(stdout_output)
                if stderr_callback and stderr_output:
                    stderr_callback(stderr_output)
                
                return stdout_output, stderr_output, process.returncode
                
            except subprocess.TimeoutExpired:
                error_msg = f"Simple command timed out after {timeout} seconds"
                if stderr_callback:
                    stderr_callback(error_msg)
                return "", error_msg, 124
            except Exception as e:
                error_msg = f"Error executing simple command: {str(e)}"
                logger.warning(error_msg)
                # Fall through to pexpect method
        
        # For complex/interactive commands, use pexpect
        if not PEXPECT_AVAILABLE:
            logger.error("pexpect is not available. Cannot execute interactive shell commands.")
            return "", "Error: pexpect not available. Install with: pip install pexpect", 1
        
        try:
            # Prepare environment to match Python kernel
            env = self._prepare_shell_environment()
            
            # Prepare the command with environment activation
            activation_cmd = self._get_shell_activation_command()
            if activation_cmd:
                full_command = f"bash -c '{activation_cmd} && {command}'"
            else:
                full_command = command
            
            logger.debug(f"Executing interactive shell command: {full_command}")
            logger.debug(f"Using environment type: {self.env_info.get('env_type', 'unknown')}")
            
            # Use pexpect to handle interactive commands
            child = pexpect.spawn(full_command, timeout=timeout, cwd=self.cwd, env=env)
            stdout_content = []
            stderr_content = []
            
            start_time = time.time()
            last_output_time = time.time()
            timeout_hint_shown = False
            
            try:
                while True:
                    try:
                        # Wait for output or EOF with a shorter timeout for responsiveness
                        index = child.expect([pexpect.TIMEOUT, pexpect.EOF], timeout=0.5)
                        
                        if index == 0:  # Timeout - check if process is waiting for input or still running
                            # Get any output that's available
                            if child.before:
                                output = child.before.decode('utf-8', errors='ignore')
                                stdout_content.append(output)
                                if stdout_callback:
                                    stdout_callback(output)
                                last_output_time = time.time()  # Reset timer on output
                            
                            # Check if we should show timeout hint
                            elapsed_since_last_output = time.time() - last_output_time
                            if show_timeout_hints and elapsed_since_last_output > 30 and not timeout_hint_shown:
                                hint_msg = (
                                    f"\n⏱️  Shell command has been running for {elapsed_since_last_output:.0f} seconds with no output.\n"
                                    f"💡 You can:\n"
                                    f"   • Wait for the command to complete\n"
                                    f"   • Press Ctrl+C to abort the command\n"
                                    f"   • The command will timeout after {timeout} seconds total\n"
                                    f"🔄 Continuing to wait...\n"
                                )
                                if stdout_callback:
                                    stdout_callback(hint_msg)
                                else:
                                    print(hint_msg, flush=True)
                                timeout_hint_shown = True
                            
                            # Check if this looks like an input prompt
                            if child.isalive():
                                recent_output = ''.join(stdout_content[-3:])  # Last few outputs
                                if any(indicator in recent_output.lower() for indicator in 
                                       ['?', ':', 'password', 'overwrite', '(y/n)', '(y/N)', '(Y/n)']):
                                    if input_callback:
                                        user_input = input_callback(recent_output)
                                        child.sendline(user_input)
                                        logger.debug(f"Sent input to shell command: {user_input}")
                                        last_output_time = time.time()  # Reset timer after input
                                    else:
                                        # Default behavior for common prompts
                                        if any(yes_prompt in recent_output for yes_prompt in 
                                               ['(y/N)', '(Y/n)', 'Overwrite?']):
                                            child.sendline('y')
                                            logger.debug("Sent default 'y' response to shell command")
                                        else:
                                            child.sendline('')
                                            logger.debug("Sent empty response to shell command")
                                        last_output_time = time.time()  # Reset timer after input
                            
                            # Check total elapsed time
                            total_elapsed = time.time() - start_time
                            if total_elapsed > timeout:
                                timeout_msg = f"\n⏰ Shell command timed out after {timeout} seconds. Terminating...\n"
                                if stderr_callback:
                                    stderr_callback(timeout_msg)
                                else:
                                    print(timeout_msg, flush=True)
                                child.terminate()
                                break
                            continue
                            
                        elif index == 1:  # EOF - process finished
                            # Get any remaining output
                            if child.before:
                                output = child.before.decode('utf-8', errors='ignore')
                                stdout_content.append(output)
                                if stdout_callback:
                                    stdout_callback(output)
                            break
                    
                    except pexpect.exceptions.TIMEOUT:
                        if not child.isalive():
                            break
                        continue
                    except pexpect.exceptions.EOF:
                        break
                        
            except KeyboardInterrupt:
                interrupt_msg = (
                    f"\n🛑 KeyboardInterrupt received (Ctrl+C).\n"
                    f"🔄 Terminating shell command: {command[:50]}{'...' if len(command) > 50 else ''}\n"
                )
                if stderr_callback:
                    stderr_callback(interrupt_msg)
                else:
                    print(interrupt_msg, flush=True)
                
                try:
                    child.terminate()
                    child.wait()
                except:
                    pass
                
                return ''.join(stdout_content), "Shell command interrupted by user (Ctrl+C)", 130  # Standard exit code for Ctrl+C
            
            # Get final output and exit code
            try:
                child.close()
            except:
                pass
            exit_code = child.exitstatus if child.exitstatus is not None else 0
            
            # Show completion message for long commands
            total_time = time.time() - start_time
            if show_timeout_hints and total_time > 10:
                completion_msg = f"\n✅ Shell command completed in {total_time:.1f} seconds with exit code: {exit_code}\n"
                logger.debug(completion_msg.strip())
            
            return ''.join(stdout_content), ''.join(stderr_content), exit_code
            
        except Exception as e:
            error_msg = f"Error executing shell command: {str(e)}"
            logger.error(error_msg)
            return "", error_msg, 1
    
    def _execute_python_code(self, code: str, 
                            stdout_callback: Optional[Callable[[str], None]] = None,
                            stderr_callback: Optional[Callable[[str], None]] = None,
                            input_callback: Optional[Callable[[str], str]] = None,
                            timeout: Optional[float] = None,
                            show_timeout_hints: bool = True) -> Tuple[str, str, Any]:
        """Execute Python code in the Jupyter kernel with input and timeout support."""
        # Synchronize all globals from the main runtime to the sandbox runtime
        # Skip during initialization to avoid circular dependencies
        if not hasattr(self, '_initializing') or not self._initializing:
            self.sync_globals()
        
        try:
            # keep a history of executed code
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_dir = f"{g.AGENTS_SANDBOX_DIR}/executed_code"
            if not os.path.exists(history_dir):
                os.makedirs(history_dir)
            code_file = f"{history_dir}/{timestamp}.py"
            with open(code_file, "w") as f:
                f.write(code)
            
            # Send code to the kernel
            msg_id = self.kc.execute(code)
            
            # Get the reply (contains execution count and execution status)
            try:
                # Use a reasonable timeout for getting the initial reply,
                # but this doesn't affect the main execution
                reply = self.kc.get_shell_msg(timeout=30)
                if reply['content']['status'] == 'error':
                    error_msg = f"Error executing code: {reply['content']['ename']}: {reply['content']['evalue']}"
                    logging.error(error_msg)
                    if stderr_callback:
                        stderr_callback(f"\n{error_msg}\n")
            except Exception as e:
                if stderr_callback:
                    stderr_callback(str(e))
            
            # For Python code, only show timeout hints if timeout is explicitly set
            effective_show_hints = show_timeout_hints and timeout is not None
            
            # Collect and return all output with timeout handling
            return self._collect_output(timeout=timeout, stdout_callback=stdout_callback, 
                                      stderr_callback=stderr_callback, input_callback=input_callback,
                                      show_timeout_hints=effective_show_hints)
        except KeyboardInterrupt:
            interrupt_msg = (
                f"\n🛑 KeyboardInterrupt received (Ctrl+C).\n"
                f"🔄 Interrupting Python code execution...\n"
            )
            if stderr_callback:
                stderr_callback(interrupt_msg)
            else:
                print(interrupt_msg, flush=True)
            
            # Try to interrupt the kernel
            try:
                self.km.interrupt_kernel()
            except Exception as e:
                logger.warning(f"Failed to interrupt kernel: {e}")
            
            return "", "Python execution interrupted by user (Ctrl+C)", None
        finally:
            pass
    
    def execute_with_input_queue(self, code: str, input_queue: Queue[str], 
                                stdout_callback: Optional[Callable[[str], None]] = None,
                                stderr_callback: Optional[Callable[[str], None]] = None,
                                timeout: Optional[float] = None,
                                show_timeout_hints: bool = True) -> Tuple[str, str, Any]:
        """
        Execute code with a pre-populated queue of inputs.
        
        Args:
            code: Python code to execute
            input_queue: Queue containing input strings to provide when requested
            stdout_callback: Optional callback function to receive stdout in real-time
            stderr_callback: Optional callback function to receive stderr in real-time
            timeout: Maximum time in seconds for the entire execution (None for no limit, which is default)
            show_timeout_hints: Whether to show timeout hints after 30 seconds (default: True, only if timeout is set)
            
        Returns:
            Tuple of (stdout, stderr, result)
        """
        def input_handler(prompt: str) -> str:
            try:
                return input_queue.get_nowait()
            except Empty:
                return ""  # Return empty string if no input available
        
        return self.execute(code, stdout_callback, stderr_callback, input_handler, timeout, show_timeout_hints)
    
    def send_input(self, user_input: str) -> None:
        """
        Send input to the currently running code.
        This can be called from another thread while code is executing.
        
        Args:
            user_input: The input string to send
        """
        try:
            self.kc.input(user_input)
        except Exception as e:
            logger.error(f"Failed to send input: {str(e)}")
    
    def restart(self) -> None:
        """Restart the kernel."""
        # Set initializing flag to prevent sync_globals during restart
        self._initializing = True
        
        self.kc.stop_channels()
        self.km.restart_kernel()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=30)
        
        # Re-execute initialization code
        self._execute_initialization_code()
        
        # Now that initialization is complete, sync the globals
        self._initializing = False
        self.sync_globals()
        
    def shutdown(self) -> None:
        """Shutdown the kernel."""
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel()
            logger.info("Python sandbox kernel shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down kernel: {str(e)}")
            
    def __del__(self) -> None:
        """Ensure the kernel is shut down when the object is garbage collected."""
        try:
            self.shutdown()
        except:
            pass