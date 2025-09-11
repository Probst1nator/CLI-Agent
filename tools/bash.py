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
import os
import warnings
import contextlib
from typing import Callable, Optional, Dict, Any
from termcolor import colored

# Assumes g is initialized and available from the project's core
from core.globals import g

# Add warning suppression for bash tool specifically
@contextlib.contextmanager
def suppress_bash_warnings():
    """Suppress warnings specifically for bash tool operations"""
    if '--debug' not in sys.argv:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Temporarily redirect stderr for noisy subprocess operations
            old_stderr = sys.stderr
            try:
                yield
            finally:
                sys.stderr = old_stderr
    else:
        yield

class BashSession:
    """
    Manages a persistent bash session in a pseudo-terminal with async execution
    and robust, global timeout enforcement.
    """
    def __init__(self,
                 stdout_callback: Optional[Callable[[str], None]] = None,
                 stderr_callback: Optional[Callable[[str], None]] = None):

        self.stdout_callback = stdout_callback or (lambda text: print(text, end=''))
        self.stderr_callback = stderr_callback or (lambda text: print(colored(f"STDERR: {text}", "red"), end=''))

        self.bash_prompt_regex = f"__PROMPT_{uuid.uuid4().hex}__"
        self.child = None

        logging.info(colored("  - Starting bash session...", "cyan"))
        
        try:
            # --- DEFINITIVE FIX: Use pexpect's echo=False and remove the 'stty' command ---
            self.child = pexpect.spawn('bash --noprofile --norc', 
                                       encoding='utf-8', 
                                       timeout=60, 
                                       dimensions=(24, 120), 
                                       echo=False) # This is the correct, low-level way to disable echo.
            
            # Set the custom prompt and wait for it to appear
            self.child.sendline(f'export PS1="{self.bash_prompt_regex}"')
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)
            
            # The problematic 'stty' and 'set +H' commands are now removed.
            
            # Send a final command to flush any startup messages and confirm the session is ready
            dummy_boundary = f"__INIT_BOUNDARY_{uuid.uuid4().hex}__"
            self.child.sendline(f'echo "{dummy_boundary}"')
            self.child.expect_exact(dummy_boundary, timeout=10) # Expect the echo'd boundary
            self.child.expect_exact(self.bash_prompt_regex, timeout=10) # Expect the prompt that follows
            
            # Clear buffers to start clean
            self.child.before = self.child.after = ""
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
            self.child.sendline(full_command_with_boundary)
            
            while True:
                loop = asyncio.get_running_loop()
                patterns = self.child.compile_pattern_list([boundary, self.bash_prompt_regex, r'\r\n'])
                index = await loop.run_in_executor(
                    None, lambda: self.child.expect_list(patterns, timeout=timeout + 5)
                )
                
                output_fragment = self.child.before.replace(full_command_with_boundary, "").lstrip('\r\n')

                if output_fragment:
                    clean_line = output_fragment.strip()
                    if clean_line and self.bash_prompt_regex not in clean_line:
                        self._stream_output(clean_line)
                        output_capture.write(clean_line + '\n')
                
                if index == 0:  # Boundary found
                    break
                if index == 1: # Intermediate prompt found, ignore it
                    continue
            
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
            with suppress_bash_warnings():
                bash._session = BashSession(
                    stdout_callback=getattr(g, 'stdout_callback', None),
                    stderr_callback=getattr(g, 'stderr_callback', None)
                )
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