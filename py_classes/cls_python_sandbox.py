#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import time
from typing import Tuple, Any, Optional, Dict, Callable
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
        self.ensure_kernel_available()
        if g.USE_SANDBOX:
            self.cwd = g.AGENTS_SANDBOX_DIR
        else:
            self.cwd = os.getcwd()
        self._start_kernel()
        self._execution_timeout = False
        self._last_output_time = None
        
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
        
        # Execute initialization code and set working directory
        self.execute("import sys, os, json, io, traceback")
        self.execute(f"os.chdir('{self.cwd}')")
        self.execute("print(f'Python sandbox initialized with working directory: {os.getcwd()}')")
        
        # Suppress tqdm warnings about ipywidgets
        self.execute("import warnings")
        self.execute("warnings.filterwarnings('ignore', category=UserWarning, module='tqdm')")
        
        # Add project directories to Python path to access project modules
        self.execute(f"sys.path.append('{g.PROJ_DIR_PATH}')")  # Add main project dir
        self.execute(f"sys.path.append('{os.path.dirname(g.PROJ_DIR_PATH)}')")  # Add parent dir
        
        # Import common modules
        self.execute("import os, pathlib")
        self.execute("from utils import *")
        self.execute("home_dir = str(pathlib.Path.home())")  # Make home dir available
        
        self.execute(f"from py_classes.globals import g")
        self.execute(f"g.FORCE_LOCAL = {g.FORCE_LOCAL}")
        self.execute(f"g.LLM = {g.LLM}")
        self.execute(f"g.LLM_STRENGTHS = {g.LLM_STRENGTHS}")
        
        # Import utils
        self.execute("from utils import *")
        
        # Print current path for debugging
        self.execute("print(f'Python path includes: {sys.path}')")
    
    def _timeout_handler(self) -> None:
        """Handle execution timeout by interrupting the kernel when no new output is received."""
        current_time = time.time()
        if self._last_output_time and (current_time - self._last_output_time) > self._max_idle_time:
            self._execution_timeout = True
            logger.warning("No output received for specified idle time, interrupting kernel")
            try:
                self.km.interrupt_kernel()
            except Exception as e:
                logger.error(f"Failed to interrupt kernel: {str(e)}")
                # As a last resort, restart the kernel
                self.restart()
        
    def _collect_output(self, timeout: float, stdout_callback: Optional[Callable[[str], None]] = None, 
                       stderr_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, str, Any]:
        """
        Collect all output from the kernel execution.
        
        Args:
            timeout: Maximum time to wait for more output (seconds)
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
        
        # Initialize last output time
        self._last_output_time = time.time()
        
        # Continue getting messages until idle message is received or timeout
        start_time = time.time()
        last_message_time = time.time()
        
        while True:
            if self._execution_timeout:
                timeout_msg = "\nExecution timed out due to no output received.\n"
                if stderr_callback:
                    stderr_callback(timeout_msg)
                stderr_content.append(timeout_msg)
                error = "TimeoutError: No output received for specified idle time."
                break
                
            try:
                msg = self.kc.get_iopub_msg(timeout=0.1)
                last_message_time = time.time()  # Update last message time
                msg_type = msg['header']['msg_type']
                content = msg['content']
                
                # Update last output time whenever we receive any output
                if msg_type in ['stream', 'execute_result', 'display_data', 'error']:
                    self._last_output_time = time.time()
                
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
                # Check if we've been waiting too long for any message
                if current_time - last_message_time > timeout:
                    # No messages for a while, assume execution is complete or stuck
                    break
                continue
        
        # If we're exiting without seeing an idle state, log a warning
        if not execution_completed and not self._execution_timeout:
            warning_msg = "\n⚠️ Warning: Execution may not have completed properly, no idle state detected.\n"
            logger.warning("⚠️ Execution may not have completed properly, no idle state detected")
            if stderr_callback:
                stderr_callback(warning_msg)
            stderr_content.append(warning_msg)
                
        return ''.join(stdout_content), ''.join(stderr_content), result if not error else error
    
    def execute(self, code: str, stdout_callback: Optional[Callable[[str], None]] = None,
               stderr_callback: Optional[Callable[[str], None]] = None, max_idle_time: Optional[int] = None) -> Tuple[str, str, Any]:
        """
        Execute the given code in the kernel and return stdout, stderr, and result.
        
        Args:
            code: Python code to execute
            stdout_callback: Optional callback function to receive stdout in real-time
            stderr_callback: Optional callback function to receive stderr in real-time
            max_idle_time: Maximum time in seconds without receiving new output before interrupting the kernel (None for no limit)
            
        Returns:
            Tuple of (stdout, stderr, result)
        """
        # Reset timeout flag and last output time
        self._execution_timeout = False
        self._last_output_time = None
        
        # Store max_idle_time as instance variable
        self._max_idle_time = max_idle_time
        
        # Set up timeout if specified
        self._timeout_timer = None
        if max_idle_time is not None and max_idle_time > 0:
            self._timeout_timer = threading.Timer(max_idle_time, self._timeout_handler)
            self._timeout_timer.daemon = True
            self._timeout_timer.start()
        
        try:
            # Send code to the kernel
            msg_id = self.kc.execute(code)
            
            # Get the reply (contains execution count and execution status)
            try:
                reply = self.kc.get_shell_msg(timeout=30)
                if reply['content']['status'] == 'error':
                    error_msg = f"Error executing code: {reply['content']['ename']}: {reply['content']['evalue']}"
                    logging.error(error_msg)
                    if stderr_callback:
                        stderr_callback(f"\n{error_msg}\n")
            except Exception as e:
                if stderr_callback:
                    stderr_callback(str(e))
            
            # Collect and return all output
            return self._collect_output(timeout=30, stdout_callback=stdout_callback, stderr_callback=stderr_callback)
        finally:
            # Cancel timeout timer if it exists and is still running
            if self._timeout_timer and self._timeout_timer.is_alive():
                self._timeout_timer.cancel()
    
    def restart(self) -> None:
        """Restart the kernel."""
        self.kc.stop_channels()
        self.km.restart_kernel()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=30)
        
        # Re-execute initialization code and set working directory
        self.execute("import sys, os, json, io, traceback")
        self.execute(f"os.chdir('{self.cwd}')")
        
        # Add project directories to Python path to access project modules
        self.execute(f"sys.path.append('{g.PROJ_DIR_PATH}')")  # Add main project dir
        self.execute(f"sys.path.append('{os.path.dirname(g.PROJ_DIR_PATH)}')")  # Add parent dir
        
        # Import common modules
        self.execute("import os, pathlib")
        self.execute("from utils import *")
        self.execute("home_dir = str(pathlib.Path.home())")  # Make home dir available
        
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