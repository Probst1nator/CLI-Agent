# tools/bash.py
"""
This file implements the 'bash' tool. It allows the agent to execute
shell commands within a persistent pseudo-terminal, with live
streaming of output to the console and robust timeout handling.
"""
import pexpect
import io
import uuid
import logging
import re
import asyncio
import threading
import sys
import warnings
import contextlib
import os
from typing import Callable, Optional, Dict, Any
from termcolor import colored

# Assumes g is initialized and available from the project's core
from core.globals import g

# Add warning suppression for bash tool specifically
@contextlib.contextmanager
def suppress_bash_warnings():
    """
    Suppress warnings specifically for bash tool operations.
    CRITICAL FIX: Only suppress non-critical warnings, preserve bash startup errors.
    """
    # Always allow critical errors through when --debug is enabled or when bash fails
    debug_mode = '--debug' in sys.argv
    
    if not debug_mode:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # CRITICAL: Don't redirect stderr during bash initialization
            # Bash startup errors need to be visible for debugging EOF issues
            try:
                yield
            except Exception as e:
                # If bash fails, re-raise with better context
                logging.error(f"Bash session creation failed: {e}")
                raise
    else:
        # In debug mode, show everything for diagnosis
        logging.debug("DEBUG mode: Bash warnings and errors will be shown")
        yield

class BashSession:
    """
    Manages a persistent bash session in a pseudo-terminal with async execution
    and robust, global timeout enforcement.
    """
    def __init__(self,
                 stdout_callback: Optional[Callable[[str], None]] = None,
                 stderr_callback: Optional[Callable[[str], None]] = None,
                 fallback_mode: bool = False):

        self.stdout_callback = stdout_callback or (lambda text: print(text, end=''))
        self.stderr_callback = stderr_callback or (lambda text: print(colored(f"STDERR: {text}", "red"), end=''))

        self.bash_prompt_regex = f"__PROMPT_{uuid.uuid4().hex}__"
        self.child = None
        
        # STREAMING FIX: Add state tracking to prevent duplicate output
        self.displayed_lines: set = set()  # Track what we've already shown
        self.last_buffer_position: int = 0  # Track where we last read from buffer

        # Only log session start in debug mode to reduce noise
        debug_mode = '--debug' in sys.argv
        if debug_mode:
            logging.info(colored("  - Starting bash session...", "cyan"))
        
        # ENHANCED DIAGNOSTICS: Check shell availability and environment
        import shutil
        bash_path = shutil.which('bash') or '/usr/bin/bash' or '/bin/bash'
        if debug_mode:
            logging.debug(f"  - Detected bash path: {bash_path}")
        
        # Check if bash is executable
        if not os.path.isfile(bash_path) or not os.access(bash_path, os.X_OK):
            raise RuntimeError(f"Bash not found or not executable at {bash_path}")
        
        # FALLBACK MODE: Use different parameters for problematic environments
        if fallback_mode:
            if debug_mode:
                logging.warning("  - Using fallback mode for bash session creation")
            spawn_command = f'{bash_path} -i'  # Interactive mode without profile restrictions
            spawn_timeout = 30  # Shorter timeout for fallback
            spawn_dimensions = (80, 24)  # Smaller dimensions
        else:
            spawn_command = f'{bash_path} --noprofile --norc'
            spawn_timeout = 60
            spawn_dimensions = (24, 120)
        
        try:
            if debug_mode:
                logging.debug(f"  - Spawning bash with command: {spawn_command}")
                logging.debug(f"  - Using prompt: {self.bash_prompt_regex}")
            
            # ENHANCED ERROR CAPTURE: Use selected parameters
            self.child = pexpect.spawn(spawn_command, 
                                       encoding='utf-8', 
                                       timeout=spawn_timeout, 
                                       dimensions=spawn_dimensions, 
                                       echo=False)
            
            # CRITICAL: Log bash process state immediately after spawn
            if self.child.isalive():
                if debug_mode:
                    logging.debug(f"  - Bash process started successfully (PID: {self.child.pid})")
            else:
                raise RuntimeError(f"Bash process died immediately after spawn (exit code: {self.child.exitstatus})")
            
            # Set the custom prompt and wait for it to appear
            if debug_mode:
                logging.debug(f"  - Setting custom prompt: {self.bash_prompt_regex}")
            self.child.sendline(f'export PS1="{self.bash_prompt_regex}"')
            
            try:
                self.child.expect_exact(self.bash_prompt_regex, timeout=10)
                if debug_mode:
                    logging.debug("  - Custom prompt established successfully")
            except pexpect.TIMEOUT:
                logging.error(f"  - Timeout waiting for prompt. Buffer content: '{self.child.before}'")
                raise
            except pexpect.EOF:
                logging.error(f"  - EOF during prompt setup. Before: '{self.child.before}', Exit status: {self.child.exitstatus}")
                raise
            
            # The problematic 'stty' and 'set +H' commands are now removed.
            
            # Send a final command to flush any startup messages and confirm the session is ready
            dummy_boundary = f"__INIT_BOUNDARY_{uuid.uuid4().hex}__"
            if debug_mode:
                logging.debug(f"  - Testing session with boundary: {dummy_boundary}")
            self.child.sendline(f'echo "{dummy_boundary}"')
            
            try:
                self.child.expect_exact(dummy_boundary, timeout=10) # Expect the echo'd boundary
                if debug_mode:
                    logging.debug("  - Boundary echo successful")
                self.child.expect_exact(self.bash_prompt_regex, timeout=10) # Expect the prompt that follows
                if debug_mode:
                    logging.debug("  - Final prompt confirmation successful")
            except pexpect.TIMEOUT:
                logging.error(f"  - Timeout during session validation. Buffer: '{self.child.before}'")
                raise
            except pexpect.EOF:
                logging.error(f"  - EOF during session validation. Before: '{self.child.before}', Exit: {self.child.exitstatus}")
                raise
            
            # Clear buffers to start clean
            self.child.before = self.child.after = ""
            if debug_mode:
                logging.info(colored("  - Bash session ready.", "cyan"))

        except pexpect.exceptions.ExceptionPexpect as e:
            # --- Enhanced Error Logging ---
            logging.error(colored(f"  - Failed to spawn bash: {e}", "red"))
            if self.child:
                logging.error(f"    Pexpect 'before' buffer: {self.child.before}")
                logging.error(f"    Pexpect 'after' buffer: {self.child.after}")
            self.child = None
            raise

    def _stream_output(self, line: str):
        # Filter out the internal prompt from the streamed output
        if self.bash_prompt_regex not in line:
            self.stdout_callback(f"⚙️  {line}\n")

    async def execute(self, command: str, timeout: int) -> str:
        """
        Executes a command asynchronously with a global timeout.
        """
        if not self.child or not self.child.isalive():
            raise ConnectionError("Bash session is not active.")

        command_lines = [line.strip() for line in command.strip().split('\n') if line.strip()]
        clean_command = '; '.join(command_lines)

        if not clean_command:
            return "The bash call was passed without a command."

        boundary = f"__BOUNDARY_{uuid.uuid4().hex}__"
        full_command_with_boundary = f'{clean_command}; echo "{boundary}"'

        async def _read_loop():
            output_capture = io.StringIO()
            
            # COMMAND HEADER: Show execution context before starting
            import getpass
            import socket
            import os
            current_user = getpass.getuser()
            hostname = socket.gethostname()
            current_dir = os.getcwd()
            
            # Display command execution header with colors
            from termcolor import colored
            
            # Create colorized header components
            user_colored = colored(current_user, 'green', attrs=['bold'])
            host_colored = colored(hostname, 'blue', attrs=['bold'])
            dir_colored = colored(current_dir, 'yellow')
            prompt_colored = colored('$', 'white', attrs=['bold'])
            command_colored = colored(clean_command, 'white', attrs=['bold'])
            
            # Assemble the colorized header
            header = f"[{user_colored}@{host_colored}:{dir_colored}]{prompt_colored} {command_colored}"
            self.stdout_callback(f"💻 {header}")
            
            self.child.sendline(full_command_with_boundary)
            
            loop = asyncio.get_running_loop()
            
            # STREAMING FIX: Reset state tracking for this command
            self.displayed_lines.clear()
            self.last_buffer_position = 0
            
            # Skip the initial command echo
            try:
                await loop.run_in_executor(None, self.child.expect, pexpect.TIMEOUT, 0.1)
                # STREAMING FIX: Don't skip actual output, only filter out command echo later
                # Reset position to 0 so we capture all output including the first line
                self.last_buffer_position = 0
            except:
                pass
            
            while True:
                
                try:
                    # Use shorter timeout for more responsive streaming
                    index = await loop.run_in_executor(
                        None, lambda: self.child.expect([boundary, self.bash_prompt_regex, pexpect.TIMEOUT], timeout=0.5)
                    )
                    
                    # STREAMING FIX: Read only NEW content since last position
                    current_buffer = self.child.before or ""
                    new_content = current_buffer[self.last_buffer_position:]
                    self.last_buffer_position = len(current_buffer)
                    
                    # Process only the new content to prevent duplication
                    if new_content:
                        # STREAMING FIX: More precise command echo filtering
                        # Remove command echo if it appears at the start, but preserve actual output
                        lines_to_filter = [
                            full_command_with_boundary.strip(),
                            clean_command.strip()
                        ]
                        
                        filtered_lines = []
                        for line in new_content.split('\n'):
                            # Skip lines that are exactly the command echo
                            if line.strip() not in lines_to_filter:
                                filtered_lines.append(line)
                        
                        new_content = '\n'.join(filtered_lines)
                        
                        # STREAMING FIX: Process only new lines with duplicate detection
                        new_lines = new_content.split('\n')
                        for line in new_lines:
                            clean_line = line.strip()
                            if (clean_line and 
                                self.bash_prompt_regex not in clean_line and 
                                clean_line != boundary and
                                clean_line not in self.displayed_lines):  # Duplicate check
                                
                                self.displayed_lines.add(clean_line)  # Track this line
                                self._stream_output(clean_line)
                                output_capture.write(clean_line + '\n')
                    
                    if index == 0:  # Boundary found - command completed
                        break
                    elif index == 1:  # Prompt found - command completed
                        break
                    # index == 2 is TIMEOUT - continue reading for more output
                    
                except Exception:
                    # If we get an exception, try to read what's available
                    if hasattr(self.child, 'before') and self.child.before:
                        remaining = self.child.before.strip()
                        if remaining and self.bash_prompt_regex not in remaining:
                            self._stream_output(remaining)
                            output_capture.write(remaining + '\n')
                    break
            
            await loop.run_in_executor(None, self.child.expect_exact, self.bash_prompt_regex, 5)
            return output_capture.getvalue().strip()

        try:
            return await asyncio.wait_for(_read_loop(), timeout=timeout)
        except asyncio.TimeoutError:
            return self._convert_to_background_process(command, timeout)
        except pexpect.exceptions.EOF:
            error_string = "\n[Error: EOF while executing command. The session closed unexpectedly.]\n"
            self.stderr_callback(error_string)
            self.child = None
            return error_string

    def _convert_to_background_process(self, command: str, timeout: int) -> str:
        """
        Convert a timed-out command to a background process instead of terminating it.
        """
        try:
            partial_output = ""
            if self.child and self.child.before:
                partial_output = self.child.before.strip()
            
            bg_child = pexpect.spawn('bash', encoding='utf-8', timeout=None)
            bg_child.sendline(command)
            
            from tools.bash import bash
            process_id = bash._create_background_process(command, bg_child)
            
            try:
                self.child.sendintr()
                self.child.expect_exact(self.bash_prompt_regex, timeout=2)
                self.child.before = self.child.after = ""
            except:
                pass
            
            response = f"⏳ Command exceeded {timeout}s timeout and was moved to background.\n"
            response += f"📋 Process ID: {process_id}\n"
            response += f"🔧 Command: {command}\n"
            if partial_output:
                response += f"📄 Partial output: {partial_output[:200]}{'...' if len(partial_output) > 200 else ''}\n"
            response += "\n💡 Use <taskmanager><list/></taskmanager> to see all background processes."
            response += f"\n💡 Use <taskmanager><stdout>{process_id}</stdout></taskmanager> to view output."
            response += f"\n💡 Use <taskmanager><input><pid>{process_id}</pid><data>y\n</data></input></taskmanager> to send input."
            
            return response
            
        except Exception as e:
            logging.error(f"Error converting to background process: {e}")
            error_string = f"\n[Error: TIMEOUT after {timeout} seconds. Failed to convert to background process: {e}]\n"
            self._try_recover_session()
            return error_string

    def _try_recover_session(self):
        if not self.child or not self.child.isalive(): return
        try:
            self.child.sendintr()
            self.child.expect_exact(self.bash_prompt_regex, timeout=2)
            self.child.before = self.child.after = ""
            self.stdout_callback("\n[Session recovered by sending SIGINT (Ctrl+C)]\n")
        except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF):
            self.stderr_callback("\n[Failed to recover session. Terminating process.]\n")
            if self.child: self.child.close(force=True)
            self.child = None

    def close(self):
        if self.child and self.child.isalive():
            self.child.close(force=True)
        logging.info(colored("  - Bash session closed.", "cyan"))
        self.child = None

class bash:
    _session: Optional[BashSession] = None
    _background_processes: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def get_delim() -> str:
        return 'bash'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "bash",
            "description": "Executes shell commands in a persistent terminal session, streaming output live. Requires a <timeout> tag specifying the maximum execution time in seconds. Essential for file system operations, running scripts, and system interaction.",
            "example": "<bash><timeout>30</timeout>\nls -l /app/src\n</bash>"
        }

    @staticmethod
    def _get_session() -> BashSession:
        session_dead = (bash._session is None or 
                        bash._session.child is None or 
                        not bash._session.child.isalive())
        
        if session_dead:
            if bash._session:
                logging.warning(colored("  - Bash session was found dead. Restarting.", "yellow"))
            
            # SILENT SESSION CREATION: Try multiple approaches with minimal output
            max_attempts = 3
            session_created = False
            
            for attempt in range(max_attempts):
                try:
                    with suppress_bash_warnings():
                        bash._session = BashSession(
                            stdout_callback=getattr(g, 'stdout_callback', None),
                            stderr_callback=getattr(g, 'stderr_callback', None)
                        )
                    session_created = True
                    break
                    
                except (pexpect.EOF, pexpect.TIMEOUT, RuntimeError):
                    if attempt < max_attempts - 1:
                        # Silent retry with exponential backoff
                        import time
                        time.sleep(0.5 * (2 ** attempt))
                    
            # If standard attempts failed, try fallback silently
            if not session_created:
                try:
                    with suppress_bash_warnings():
                        bash._session = BashSession(
                            stdout_callback=getattr(g, 'stdout_callback', None),
                            stderr_callback=getattr(g, 'stderr_callback', None),
                            fallback_mode=True
                        )
                    session_created = True
                except Exception:
                    pass
            
            # Only show error if everything failed
            if not session_created:
                logging.error(colored("  - Bash session creation failed completely", "red"))
                raise RuntimeError("Unable to create bash session")
                        
        return bash._session
    
    @staticmethod
    def _create_background_process(command: str, child_process: pexpect.spawn) -> str:
        """
        Create a background process entry for tracking by taskmanager.
        """
        process_id = f"bg_{uuid.uuid4().hex[:8]}"
        output_buffer = io.StringIO()
        
        bash._background_processes[process_id] = {
            'source': 'bash',
            'command': command,
            'child': child_process,
            'status': 'running',
            'created_at': asyncio.get_event_loop().time(),
            'output_buffer': output_buffer
        }
        
        def capture_output():
            try:
                while child_process.isalive():
                    try:
                        line = child_process.readline()
                        if line:
                            output_buffer.write(line)
                            output_buffer.flush()
                    except pexpect.TIMEOUT:
                        continue
                    except (pexpect.EOF, OSError):
                        break
                
                try:
                    remaining = child_process.read()
                    if remaining:
                        output_buffer.write(remaining)
                        output_buffer.flush()
                except:
                    pass
                
                bash._background_processes[process_id]['status'] = 'finished'
            except Exception as e:
                logging.error(f"Error capturing background process output: {e}")
                if process_id in bash._background_processes:
                    bash._background_processes[process_id]['status'] = 'error'
        
        capture_thread = threading.Thread(target=capture_output, daemon=True)
        capture_thread.start()
        
        return process_id

    @staticmethod
    async def run(content: str) -> str:
        """
        Parses XML-like input, extracts a command and timeout, and executes it.
        """
        try:
            timeout_match = re.search(r'<timeout>(\d+)</timeout>', content)
            
            if timeout_match:
                timeout = int(timeout_match.group(1))
                if timeout <= 0: timeout = 5
                command_content = content.replace(timeout_match.group(0), '').strip()
            else:
                timeout = 5
                command_content = content.strip()

            if not command_content:
                return "Error: No command provided to execute in the <bash> tool."

            session = bash._get_session()
            bash_return = await session.execute(command_content, timeout=timeout)
            
            if not bash_return.strip():
                bash_return = "The bash tool exited successfully with no output."
            return bash_return
        except Exception as e:
            logging.error(f"Fatal error executing bash command: {e}", exc_info=True)
            return f"Fatal error executing bash command: {e}"