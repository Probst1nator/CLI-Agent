# tools/bash.py
"""
This file implements the 'bash' tool. It allows the agent to execute
shell commands within a persistent pseudo-terminal, with live
streaming of output to the console.
"""
import pexpect
import io
import uuid
import logging
from typing import Callable, Optional
from termcolor import colored

# Assumes g is initialized and available from the project's core
from core.globals import g

class BashSession:
    """
    Manages a persistent bash session in a pseudo-terminal, allowing for
    command execution with real-time, streaming output.
    """
    def __init__(self,
                 stdout_callback: Optional[Callable[[str], None]] = None,
                 stderr_callback: Optional[Callable[[str], None]] = None):

        self.stdout_callback = stdout_callback or (lambda text: print(text, end=''))
        self.stderr_callback = stderr_callback or (lambda text: print(colored(f"STDERR: {text}", "red"), end=''))

        # A unique, hard-to-guess string to use as the shell prompt. This is the
        # primary mechanism for reliably detecting when a command has finished executing.
        self.bash_prompt_regex = f"__PROMPT_{uuid.uuid4().hex}__"
        self.child = None

        logging.info(colored("  - Starting bash session...", "cyan"))
        
        try:
            # Spawn a new bash process. --noprofile and --norc ensure a clean, predictable environment.
            self.child = pexpect.spawn('bash --noprofile --norc', encoding='utf-8', timeout=60, dimensions=(24, 120))
            
            # Set the unique prompt to reliably detect command completion.
            self.child.sendline(f'export PS1="{self.bash_prompt_regex}"')
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)

            # Configure the shell for automation:
            # stty -echo: Prevents the terminal from echoing back the commands we send.
            # set +H: Disables history expansion (e.g., '!!' or '!$').
            self.child.sendline('stty -echo')
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)
            self.child.sendline('set +H')
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)

            # CRITICAL: Execute a dummy command with a unique boundary to flush all startup messages
            # and leave the session in a pristine, ready state for the first real command.
            dummy_boundary = f"__INIT_BOUNDARY_{uuid.uuid4().hex}__"
            self.child.sendline(f'echo "{dummy_boundary}"')
            self.child.expect_exact(dummy_boundary, timeout=10)
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)
            
            # Clear all buffers to ensure a clean state.
            self.child.before = ""
            self.child.after = ""
            
            logging.info(colored("  - Bash session ready.", "cyan"))

        except pexpect.exceptions.ExceptionPexpect as e:
            logging.error(colored(f"  - Failed to spawn bash: {e}", "red"))
            self.child = None
            raise

    def _stream_output(self, line: str):
        """Processes and streams a single, complete line of output."""
        self.stdout_callback(f"⚙️  {line}\n")

    def execute(self, command: str) -> str:
        """
        Executes a command using an active reading loop to provide true
        live output streaming and capture the full result.
        """
        if not self.child or not self.child.isalive():
            raise ConnectionError("Bash session is not active.")

        try:
            clean_command = command.strip()
            if not clean_command:
                return ""

            # Use another unique boundary to signal the end of the command's output.
            boundary = f"__BOUNDARY_{uuid.uuid4().hex}__"
            full_command_with_boundary = f'{clean_command}; echo "{boundary}"'
            
            self.child.sendline(full_command_with_boundary)
            
            output_capture = io.StringIO()
            
            while True:
                # Expect either the boundary or a newline. This allows for line-by-line streaming.
                patterns = self.child.compile_pattern_list([boundary, r'\r\n'])
                index = self.child.expect_list(patterns, timeout=300)

                # The output since the last match is in `child.before`.
                output_fragment = self.child.before
                
                # Filter out the echoed command itself from the output.
                if full_command_with_boundary in output_fragment:
                    output_fragment = output_fragment.replace(full_command_with_boundary, "").lstrip('\r\n')

                if output_fragment:
                    # Stream and capture the clean fragment.
                    clean_line = output_fragment.strip()
                    if clean_line:
                        self._stream_output(clean_line)
                        output_capture.write(clean_line + '\n')
                
                if index == 0:  # Boundary found, command is finished.
                    break
                # if index == 1, it was just a newline, so we loop again for more output.

            # Consume the final shell prompt to leave the session clean for the next command.
            self.child.expect_exact(self.bash_prompt_regex, timeout=10)

            return output_capture.getvalue().strip()

        except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
            error_type = type(e).__name__
            error_string = f"\n[Error: {error_type} while executing command. The session may have timed out or closed.]\n"
            if self.child and self.child.before:
                 error_string += f"Partial output: {self.child.before.strip()}"
            self.stderr_callback(error_string)
            self._try_recover_session()
            if isinstance(e, pexpect.exceptions.EOF):
                self.child = None # Mark session as dead
            return error_string

    def _try_recover_session(self):
        """Attempts to recover a stalled session by sending Ctrl+C (SIGINT)."""
        if not self.child or not self.child.isalive(): return
        try:
            self.child.sendintr() # Send SIGINT
            self.child.expect_exact(self.bash_prompt_regex, timeout=5)
            self.child.before, self.child.after = "", "" # Clean up buffers after recovery
            self.stdout_callback("[Session recovered by sending SIGINT (Ctrl+C)]\n")
        except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF):
            self.stderr_callback("[Failed to recover session. It may be unstable or dead.]\n")
            if self.child: self.child.close(force=True)
            self.child = None

    def close(self):
        """Closes the bash session."""
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
            "description": "Executes shell commands in a persistent terminal session, streaming output live. Essential for file system operations, running scripts, and system interaction.",
            "example": "<bash>\nls -l /app/src\n</bash>"
        }

    @staticmethod
    def _get_session() -> BashSession:
        """
        Initializes and returns the singleton BashSession instance.
        If the session has died, it is automatically restarted.
        """
        session_dead = (bash._session is None or 
                        bash._session.child is None or 
                        not bash._session.child.isalive())
        
        if session_dead:
            if bash._session:
                logging.warning(colored("  - Bash session was found dead. Restarting.", "yellow"))
            # This assumes that the stdout/stderr callbacks are available via `g`
            # which is set up in the main application entry point.
            bash._session = BashSession(
                stdout_callback=getattr(g, 'stdout_callback', None),
                stderr_callback=getattr(g, 'stderr_callback', None)
            )
        return bash._session

    @staticmethod
    def run(content: str) -> str:
        """
        Executes a shell command. Live output is streamed to the console,
        and the full, clean output is returned as a string upon completion.
        """
        try:
            session = bash._get_session()
            return session.execute(content)
        except Exception as e:
            # This catch is mainly for fatal errors during session creation.
            logging.error(f"Fatal error executing bash command: {e}", exc_info=True)
            return f"Fatal error executing bash command: {e}"