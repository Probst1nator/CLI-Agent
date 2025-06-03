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
        """Stream output and detect stalls, consulting LLM when output stops but process hasn't exited."""
        start_time = time.time()
        buffer = ""
        last_output_time = time.time()
        stall_threshold = 10  # seconds without new output before consulting LLM
        command_completed = False
        potential_completion_time = None  # Track when we first see a bash prompt
        
        # CRITICAL FIX: Clear pexpect's internal 'before' buffer to prevent contamination from previous commands
        self.child.before = self.child.string_type()
        
        while time.time() - start_time < timeout:
            try:
                # Try to read any available data
                self.child.expect([pexpect.TIMEOUT], timeout=0.1)
                
                # Get the new output since last read
                new_output = str(self.child.before)
                if new_output and new_output != buffer:
                    # Filter out content we've already displayed
                    if self.cumulative_output and new_output.startswith(self.cumulative_output):
                        # Only show the truly new part that we haven't seen before
                        truly_new = new_output[len(self.cumulative_output):]
                        if truly_new:
                            processed_output = self._process_output_with_emoji(truly_new)
                            self.stdout_callback(processed_output)
                            self.cumulative_output += truly_new
                            last_output_time = time.time()
                            # Reset completion time since we got new output
                            potential_completion_time = None
                    elif not self.cumulative_output or not new_output.startswith(self.cumulative_output):
                        # This is completely new output (first command or unrelated content)
                        processed_output = self._process_output_with_emoji(new_output)
                        self.stdout_callback(processed_output)
                        self.cumulative_output += new_output
                        last_output_time = time.time()
                        # Reset completion time since we got new output
                        potential_completion_time = None
                    
                    buffer = new_output
                    
                    # Clear before for the next iteration to only get new data
                    self.child.before = self.child.string_type()
                
                # Check if we have a bash prompt at the end (potential command completion)
                if buffer:
                    # Look at the end of the buffer, accounting for potential trailing whitespace/newlines
                    buffer_end = buffer.rstrip()
                    if buffer_end:
                        lines = buffer_end.split('\n')
                        if lines:
                            last_line = lines[-1].strip()
                            # Check if last line looks like a bash prompt
                            # Use re.search instead of re.match to be more flexible
                            if re.search(self.bash_prompt_regex, last_line):
                                if potential_completion_time is None:
                                    # First time we see a bash prompt, start waiting
                                    potential_completion_time = time.time()
                                elif time.time() - potential_completion_time >= 3.0:
                                    # We've seen a bash prompt for 3+ seconds with no new output
                                    # This suggests the command is truly complete
                                    command_completed = True
                                    break

                # Check for output stall (only if command hasn't completed)
                if not command_completed:
                    time_since_last_output = time.time() - last_output_time
                    # If we have a potential completion but haven't waited long enough, use a shorter stall threshold
                    current_stall_threshold = 5 if potential_completion_time else stall_threshold
                    
                    if time_since_last_output >= current_stall_threshold:
                        # No output for current_stall_threshold seconds and process is still running
                        # Ask LLM what to do
                        if self.input_prompt_handler:
                            decision = self.input_prompt_handler(buffer)
                            
                            # Handle the new return types: bool | str
                            if decision is True:
                                # True means continue without providing input (wait longer)
                                last_output_time = time.time()  # Reset timer to wait longer
                                stall_threshold = min(stall_threshold * 1.5, 60)  # Increase threshold, max 60s
                                potential_completion_time = None  # Reset completion detection
                            elif decision is False:
                                # False means interrupt execution
                                self.child.sendcontrol('c')
                                self.stdout_callback("\n[Process interrupted by user]\n")
                                break
                            elif isinstance(decision, str):
                                # String means send it as input to the process
                                self.child.sendline(decision)
                                self.stdout_callback(f"\n[Sent input: {decision}]\n")
                                last_output_time = time.time()  # Reset timer
                                potential_completion_time = None  # Reset completion detection
                            else:
                                # Fallback: treat any other response as "wait"
                                last_output_time = time.time()
                                potential_completion_time = None  # Reset completion detection
                        else:
                            # No input handler available, just continue waiting
                            last_output_time = time.time()
                        
            except pexpect.TIMEOUT:
                continue
            except pexpect.EOF:
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
        """Add ⚙️ emoji before newlines during Python or shell execution."""
        if not (self.is_executing_python or self.is_executing_shell):
            return text
        
        # Split into lines and add emoji before each line
        lines = text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            # Hide everything before $ sign in bash prompts
            if '$' in line:
                # Find the last occurrence of $ and keep from there
                dollar_index = line.rfind('$')
                if dollar_index != -1:
                    line = '$' + line[dollar_index + 1:]
            
            if i == 0:
                # First line - add emoji at the start
                processed_lines.append('⚙️  ' + line)
            else:
                # Subsequent lines - add emoji after newline
                processed_lines.append('\n⚙️  ' + line)
        
        # Join and handle final newline correctly
        result = ''.join(processed_lines)
        if text.endswith('\n') and not result.endswith('\n'):
            result += '\n'
        
        return result