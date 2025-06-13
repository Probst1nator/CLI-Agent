import datetime
import pexpect
import tempfile
import os
import time
import re
from typing import Callable, List, Tuple, Optional
from py_classes.globals import g

class ComputationalNotebook:
    def __init__(self,
                 stdout_callback: Optional[Callable[[str], None]] = None,
                 stderr_callback: Optional[Callable[[str], None]] = None,
                 input_prompt_handler: Optional[Callable[[str, str], str]] = None):

        self.stdout_callback = stdout_callback or (lambda text: print(text, end=''))
        self.stderr_callback = stderr_callback or (lambda text: print(f"STDERR: {text}", end=''))
        self.input_prompt_handler = input_prompt_handler

        self.bash_prompt_regex = r"(\([^)]+\)\s+)?[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+:[^#$]*[#$]\s*$"
        # Alternative simpler regex for just detecting the prompt end
        self.bash_prompt_end_regex = r"^\s*[#$]\s*$"
        
        # Track cumulative output to avoid re-displaying
        self.cumulative_output = ""
        
        # Track when we're executing Python code to add emojis
        self.is_executing_python = False
        
        # Track when we're executing shell commands to add emojis
        self.is_executing_shell = False

        # Clean up old temporary files from previous runs
        g.cleanup_temp_py_files()

        self.child = pexpect.spawn('bash', encoding='utf-8', timeout=30)
        self._expect_bash_prompt()

    def _expect_bash_prompt(self, timeout=30):
        self.child.expect(self.bash_prompt_regex, timeout=timeout)
        output = str(self.child.before) + str(self.child.after)
        self.stdout_callback(output)

    def _stream_output_until_prompt(self, timeout=30):
        """Stream output using pexpect.expect for robust prompt detection."""
        start_time = time.time()
        last_output_time = time.time()
        stall_threshold = 30
        all_output_for_context = ""

        # Before starting, clear any lingering output in pexpect's buffer
        try:
            self.child.read_nonblocking(size=100000, timeout=0.1)
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass
        self.child.before = ""

        while time.time() - start_time < timeout:
            try:
                # Expect either the prompt or a timeout
                index = self.child.expect([self.bash_prompt_regex, pexpect.TIMEOUT], timeout=1)

                if index == 0:
                    # Matched the bash prompt. The command is finished.
                    # The complete output is everything before the prompt, plus the prompt itself.
                    final_output = self.child.before + self.child.after
                    if final_output:
                        processed_output = self._process_output_with_emoji(final_output)
                        self.stdout_callback(processed_output)
                    break
                elif index == 1:
                    # A timeout occurred. This means the command is still running and has produced
                    # some output, but not the final prompt yet.
                    new_output = self.child.before
                    if new_output:
                        processed_output = self._process_output_with_emoji(new_output)
                        self.stdout_callback(processed_output)
                        all_output_for_context += new_output
                        last_output_time = time.time()
                    
                    # Check for a stall.
                    time_since_last_output = time.time() - last_output_time
                    if time_since_last_output > stall_threshold:
                        if self.input_prompt_handler:
                            decision = self.input_prompt_handler(all_output_for_context)
                            if decision is True: # Wait longer
                                last_output_time = time.time()
                                stall_threshold = min(stall_threshold * 1.2, 120)
                            elif decision is False: # Interrupt
                                self.child.sendcontrol('c')
                                self.stdout_callback("\n[Process interrupted by user]\n")
                                break
                            elif isinstance(decision, str): # Send input
                                self.child.sendline(decision)
                                all_output_for_context = "" # Reset context after input
                                last_output_time = time.time()
                        else:
                            # No handler, just wait
                            last_output_time = time.time()

            except pexpect.EOF:
                self.stdout_callback("\n[Process finished unexpectedly]\n")
                break
            except pexpect.TIMEOUT:
                # This can happen if the outer loop timeout is reached
                self.stdout_callback("\n[Command timed out]\n")
                break

    def get_initialization_code(self) -> str:
        """Return initialization code for the sandbox as a single string."""
        return f"""
import sys, os, json, io, datetime
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.chdir('{os.getcwd()}')
sys.path.append('{g.CLIAGENT_ROOT_PATH}')
from py_classes.globals import g
from utils import *
"""

    def execute(self, command: str, is_python_code: bool = False, persist_python_state: bool = True):
        """Execute a command with real-time output streaming."""
        
        if is_python_code:
            py_script_path = os.path.join(g.CLIAGENT_TEMP_STORAGE_PATH, f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.py')
            with open(py_script_path, 'w') as py_file:
                py_file.write(self.get_initialization_code() + "\n" + command)
            
            # Set flag before sending command so the echoed command also gets emoji
            self.is_executing_python = True
            # Debug: confirm flag is set (remove this later)
            # Always use non-persistent execution for simplicity
            self.child.sendline(f"python3 -u {py_script_path}")
            self._stream_output_until_prompt(timeout=600)  # Increased to 10 minutes for AI model operations
            self.is_executing_python = False
            # Debug: confirm flag is reset (remove this later)
            # Files are kept until program restart for debugging purposes
        else:
            # Set shell execution flag for bash commands
            self.is_executing_shell = True
            self.child.sendline(command)
            self._stream_output_until_prompt(timeout=300)  # Increased to 5 minutes for shell commands
            self.is_executing_shell = False

    def close(self):
        self.child.sendline("exit")
        self.child.close()
        self.stdout_callback("\n[Session closed]\n")

    def _process_output_with_emoji(self, text: str) -> str:
        """Add 🐍 emoji for Python or ⚙️ for shell commands before each line."""
        if not (self.is_executing_python or self.is_executing_shell):
            return text

        emoji = '🐍' if self.is_executing_python else '⚙️'

        # Add emoji to the start and after every newline,
        # ensuring that we don't add an emoji to a blank line if the text ends with one.
        if text.endswith('\n'):
            processed_text = f'{emoji}  ' + text[:-1].replace('\n', f'\n{emoji}  ') + '\n'
        else:
            processed_text = f'{emoji}  ' + text.replace('\n', f'\n{emoji}  ')
        
        return processed_text