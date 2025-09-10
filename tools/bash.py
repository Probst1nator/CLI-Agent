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
from typing import Callable, Optional
from termcolor import colored

# Assumes g is initialized and available from the project's core
from core.globals import g

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
            self.child = pexpect.spawn('bash --noprofile --norc', encoding='utf-8', timeout=60, dimensions=(24, 120))
            self.child.sendline(f'export PS1="{self.bash_prompt_regex}"')
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)
            self.child.sendline('stty -echo; set +H')
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)
            
            dummy_boundary = f"__INIT_BOUNDARY_{uuid.uuid4().hex}__"
            self.child.sendline(f'echo "{dummy_boundary}"')
            self.child.expect_exact(dummy_boundary, timeout=10)
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)
            
            self.child.before = self.child.after = ""
            logging.info(colored("  - Bash session ready.", "cyan"))

        except pexpect.exceptions.ExceptionPexpect as e:
            logging.error(colored(f"  - Failed to spawn bash: {e}", "red"))
            self.child = None
            raise

    def _stream_output(self, line: str):
        self.stdout_callback(f"⚙️  {line}\n")

    async def execute(self, command: str, timeout: int) -> str:
        """
        Executes a command asynchronously with a global timeout.
        """
        if not self.child or not self.child.isalive():
            raise ConnectionError("Bash session is not active.")

        clean_command = command.strip()
        if not clean_command:
            return "The bash call was passed without a command."

        boundary = f"__BOUNDARY_{uuid.uuid4().hex}__"
        full_command_with_boundary = f'{clean_command}; echo "{boundary}"'

        async def _read_loop():
            output_capture = io.StringIO()
            self.child.sendline(full_command_with_boundary)
            
            while True:
                # Run the blocking pexpect call in a thread to not block asyncio loop
                loop = asyncio.get_running_loop()
                patterns = self.child.compile_pattern_list([boundary, r'\r\n'])
                index = await loop.run_in_executor(
                    None, lambda: self.child.expect_list(patterns, timeout=timeout + 5) # Inner timeout slightly longer
                )
                
                output_fragment = self.child.before.replace(full_command_with_boundary, "").lstrip('\r\n')

                if output_fragment:
                    clean_line = output_fragment.strip()
                    if clean_line:
                        self._stream_output(clean_line)
                        output_capture.write(clean_line + '\n')
                
                if index == 0:  # Boundary found
                    break
            
            await loop.run_in_executor(None, self.child.expect_exact, self.bash_prompt_regex, 10)
            return output_capture.getvalue().strip()

        try:
            return await asyncio.wait_for(_read_loop(), timeout=timeout)
        except asyncio.TimeoutError:
            error_string = f"\n[Error: TIMEOUT while executing command after {timeout} seconds. The process was terminated.]\n"
            if self.child and self.child.before:
                 error_string += f"Partial output: {self.child.before.strip()}"
            self.stderr_callback(error_string)
            self._try_recover_session()
            return error_string
        except pexpect.exceptions.EOF:
            error_string = "\n[Error: EOF while executing command. The session closed unexpectedly.]\n"
            self.stderr_callback(error_string)
            self.child = None # Mark session as dead
            return error_string

    def _try_recover_session(self):
        if not self.child or not self.child.isalive(): return
        try:
            self.child.sendintr() # Send SIGINT (Ctrl+C)
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
            bash._session = BashSession(
                stdout_callback=getattr(g, 'stdout_callback', None),
                stderr_callback=getattr(g, 'stderr_callback', None)
            )
        return bash._session

    @staticmethod
    async def run(content: str) -> str:
        """
        Parses XML-like input, extracts a command and timeout, and executes it
        asynchronously with a global timeout.
        """
        try:
            timeout_match = re.search(r'<timeout>(\d+)</timeout>', content)
            
            if timeout_match:
                timeout = int(timeout_match.group(1))
                if timeout <= 0: timeout = 10
                command_content = content.replace(timeout_match.group(0), '').strip()
            else:
                # Use default timeout of 10 seconds when no timeout is provided
                timeout = 10
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