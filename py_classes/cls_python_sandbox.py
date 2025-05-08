#!/usr/bin/env python3

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
from py_classes.globals import Globals, g
logger = logging.getLogger(__name__)

class PythonSandbox:
    """
    A class to manage a persistent Python sandbox for code execution using Jupyter kernels.
    This provides process isolation and maintains state across multiple executions.
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
        self._raw_execute("import sys, os, json, io, traceback")
        self._raw_execute(f"os.chdir('{self.cwd}')")
        self._raw_execute("print(f'Python sandbox initialized with working directory: {os.getcwd()}')")
        
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
            return self._collect_output(timeout=None)
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
                       stderr_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, str, Any]:
        """
        Collect all output from the kernel execution.
        
        Args:
            timeout: Maximum time to wait for more output (seconds), None means wait indefinitely
            stdout_callback: Optional callback function to handle stdout in real-time
            stderr_callback: Optional callback function to handle stderr in real-time
            
        Returns:
            Tuple of (stdout, stderr, result)
        """
        stdout_content = []
        stderr_content = []
        result = None
        error = None
        execution_completed = False
        
        # Continue getting messages until idle message is received
        start_time = time.time()
        last_message_time = time.time()
        
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=0.1)
                last_message_time = time.time()  # Update last message time
                msg_type = msg['header']['msg_type']
                content = msg['content']
                
                if msg_type == 'stream':
                    if content['name'] == 'stdout':
                        output_text = content['text']
                        stdout_content.append(output_text)
                        if stdout_callback:
                            stdout_callback(output_text)
                    elif content['name'] == 'stderr':
                        error_text = content['text']
                        stderr_content.append(error_text)
                        if stderr_callback:
                            stderr_callback(error_text)
                        
                elif msg_type == 'execute_result':
                    result = content['data'].get('text/plain', '')
                    
                elif msg_type == 'display_data':
                    result = content['data'].get('text/plain', '')
                    
                elif msg_type == 'error':
                    error_name = content['ename']
                    error_value = content['evalue']
                    error_traceback = '\n'.join(content['traceback'])
                    error = f"{error_name}: {error_value}\n{error_traceback}"
                    if stderr_callback:
                        stderr_callback(error)
                    
                elif msg_type == 'status' and content['execution_state'] == 'idle':
                    # Kernel has finished processing
                    execution_completed = True
                    break
                    
            except Empty:
                # No more messages in the queue
                current_time = time.time()
                # Check if we've been waiting too long for any message, but only if timeout is set
                if timeout is not None and (current_time - last_message_time > timeout):
                    # No messages for a while, assume execution is complete or stuck
                    break
                continue
        
        # If we're exiting without seeing an idle state, we won't show any warnings
        # as these are causing unintended interruptions. We'll just silently continue
        # and let the execution proceed normally, trusting the user to interrupt if needed.
        
        return ''.join(stdout_content), ''.join(stderr_content), result if not error else error
    
    def execute(self, code: str, stdout_callback: Optional[Callable[[str], None]] = None,
               stderr_callback: Optional[Callable[[str], None]] = None, timeout: Optional[float] = None) -> Tuple[str, str, Any]:
        """
        Execute the given code in the kernel and return stdout, stderr, and result.
        
        Args:
            code: Python code to execute
            stdout_callback: Optional callback function to receive stdout in real-time
            stderr_callback: Optional callback function to receive stderr in real-time
            timeout: Maximum time in seconds without receiving new output before assuming completion (None for no limit)
            
        Returns:
            Tuple of (stdout, stderr, result)
        """
        # Synchronize all globals from the main runtime to the sandbox runtime
        # Skip during initialization to avoid circular dependencies
        if not hasattr(self, '_initializing') or not self._initializing:
            self.sync_globals()
        
        try:
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
            
            # Collect and return all output, using the same timeout value
            # Always use None to disable timeouts
            return self._collect_output(timeout=None, stdout_callback=stdout_callback, stderr_callback=stderr_callback)
        finally:
            pass
    
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