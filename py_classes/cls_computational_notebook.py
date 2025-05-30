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

        self.bash_prompt_regex = r"([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+:[^#$]*[#$]\s*)"

        self.child = pexpect.spawn('bash', encoding='utf-8', timeout=30)
        self._expect_bash_prompt()

    def _expect_bash_prompt(self, timeout=30):
        self.child.expect(self.bash_prompt_regex, timeout=timeout)
        output = str(self.child.before) + str(self.child.after)
        self.stdout_callback(output)

    def _stream_output_until_prompt(self, timeout=30):
        """Stream output until we get back to bash prompt."""
        start_time = time.time()
        buffer = ""
        
        while time.time() - start_time < timeout:
            try:
                # Try to read any available data
                self.child.expect([pexpect.TIMEOUT], timeout=0.1)
                
                # Get the new output since last read
                new_output = str(self.child.before)
                if new_output and new_output != buffer:
                    # Only show the new part
                    if buffer and new_output.startswith(buffer):
                        diff = new_output[len(buffer):]
                        if diff:
                            self.stdout_callback(diff)
                    else:
                        self.stdout_callback(new_output)
                    buffer = new_output
                
                # Check if we're back at bash prompt
                if self.bash_prompt_regex and buffer:
                    if re.search(self.bash_prompt_regex, buffer):
                        break
                        
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
print(f'> Python sandbox initialized successfully with working directory: {{os.getcwd()}}')
"""

    def execute(self, command: str, is_python_code: bool = False, persist_python_state: bool = True):
        """Execute a command with real-time output streaming."""
        if is_python_code:
            py_script_path = os.path.join(g.CLIAGENT_TEMP_STORAGE_PATH, f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.py')
            with open(py_script_path, 'w') as py_file:
                py_file.write(self.get_initialization_code() + "\n" + command)
            
            # Always use non-persistent execution for simplicity
            self.child.sendline(f"python3 -u {py_script_path}")
            self._stream_output_until_prompt(timeout=120)
            try:
                os.remove(py_script_path)
            except:
                pass
        else:
            self.child.sendline(command)
            self._stream_output_until_prompt(timeout=120)

    def close(self):
        self.child.sendline("exit")
        self.child.close()
        self.stdout_callback("\n[Session closed]\n")