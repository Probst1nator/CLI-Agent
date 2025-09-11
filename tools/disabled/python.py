# tools/disabled/python.py
"""
This file implements the 'python' tool. It allows the agent to execute
Python code within a persistent, interactive Python REPL.

NOTE: This is a highly complex tool that relies on precise interaction with a
pexpect-managed Python session. Its complexity can make it fragile, which might
be why it is in the 'disabled' directory. This version is simplified for robustness.
"""
import pexpect
import os
import base64
import ast
import logging
from typing import Callable, Optional
from termcolor import colored

# Assumes g is initialized and available from the project's core
from core.globals import g

class PythonSession:
    """Manages a persistent, interactive Python session for code execution."""
    def __init__(self,
                 stdout_callback: Optional[Callable[[str], None]] = None,
                 stderr_callback: Optional[Callable[[str], None]] = None):

        self.stdout_callback = stdout_callback or (lambda text: print(text, end=''))
        self.stderr_callback = stderr_callback or (lambda text: print(colored(f"STDERR: {text}", "red"), end=''))
        
        g.cleanup_temp_py_files()
        logging.info(colored("  - Setting up persistent Python environment...", "cyan"))
        self.python_child = None
        self._initialize_persistent_python()
        logging.info(colored("  - Python session ready.", "cyan"))

    def _initialize_persistent_python(self):
        """Spawns a new Python REPL and injects initialization code."""
        try:
            if self.python_child and self.python_child.isalive():
                self.python_child.close()
            
            self.python_child = pexpect.spawn('python3 -i -u', encoding='utf-8', timeout=60)
            self.python_child.expect('>>> ', timeout=10)
            
            init_code = self.get_initialization_code()
            for line in init_code.strip().split('\n'):
                if line.strip():
                    self.python_child.sendline(line)
                    self.python_child.expect(['>>> ', r'\.\.\. '], timeout=10)
        except Exception as e:
            self.stderr_callback(f"\n[Failed to initialize persistent Python session: {e}]\n")
            self.python_child = None

    def _fix_python_indentation(self, code: str) -> str:
        """Attempts to fix common indentation errors in LLM-generated Python code."""
        try:
            ast.parse(code)
            return code # Code is already valid
        except (IndentationError, SyntaxError):
            try:
                # A simple but effective strategy: re-indent based on colons
                lines = code.split('\n')
                fixed_lines = []
                indent_level = 0
                for line in lines:
                    stripped = line.lstrip()
                    # Basic dedent logic for common keywords
                    if stripped.startswith(('elif', 'else', 'except', 'finally')):
                        indent_level = max(0, indent_level - 1)
                    
                    fixed_lines.append('    ' * indent_level + stripped)
                    
                    # Basic indent logic
                    if stripped.endswith(':'):
                        indent_level += 1
                
                fixed_code = '\n'.join(fixed_lines)
                ast.parse(fixed_code)
                return fixed_code
            except Exception:
                return code # Fallback to original code if fixing fails

    def execute(self, command: str):
        """Executes Python code in the persistent session."""
        if not self.python_child or not self.python_child.isalive():
            self._initialize_persistent_python()
        
        if not self.python_child:
            self.stderr_callback("[Error: Python session is not available.]\n")
            return

        try:
            cleaned_command = self._fix_python_indentation(command)
            
            # For multi-line code, wrap it to ensure it executes as a single block.
            # Using base64 encoding avoids issues with quotes and special characters.
            encoded_command = base64.b64encode(cleaned_command.encode('utf-8')).decode('ascii')
            exec_command = f"import base64; exec(base64.b64decode('{encoded_command}').decode('utf-8'))"
            
            self.python_child.sendline(exec_command)
            
            # Expect the prompt, which signals the end of output for this command.
            self.python_child.expect('>>> ', timeout=300)
            output = self.python_child.before.strip()
            
            # Clean up the output by removing the exec command echo and prompt artifacts.
            if exec_command in output:
                output = output.replace(exec_command, '').lstrip('\r\n')
            
            if output:
                self.stdout_callback("🐍  " + output.replace('\n', '\n🐍  ') + '\n')

        except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
            self.stderr_callback(f"\n[Error during Python execution: {type(e).__name__}. Session may be unstable.]\n")
            self._initialize_persistent_python() # Attempt to restart the session
        except Exception as e:
            self.stderr_callback(f"\n[An unexpected error occurred in Python execution: {e}]\n")
            self._initialize_persistent_python()

    def get_initialization_code(self) -> str:
        """Returns standard imports and setup code to inject into the Python session."""
        return f"""
import sys, os, json, io, datetime
os.chdir('{os.getcwd()}')
sys.path.append('{g.CLIAGENT_ROOT_PATH}')
from core.globals import g
# The agent is expected to generate any other necessary imports.
"""

    def close(self):
        """Closes the Python session."""
        if self.python_child and self.python_child.isalive():
            try:
                self.python_child.sendline("exit()")
                self.python_child.expect(pexpect.EOF, timeout=5)
            except (pexpect.TIMEOUT, pexpect.EOF):
                pass
            finally:
                self.python_child.close()
        self.stdout_callback("\n[Python session closed]\n")


class python:
    _session: Optional[PythonSession] = None
    
    @staticmethod
    def get_delim() -> str:
        return 'python'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "python",
            "description": "Executes Python code in a persistent interactive session. Variables, functions, and imports are maintained across calls.",
            "example": "<python>\nimport os\nprint(f'Current directory: {os.getcwd()}')\n</python>"
        }

    @staticmethod
    def _get_session() -> PythonSession:
        """Lazy initialization of the singleton PythonSession instance."""
        if python._session is None:
            python._session = PythonSession(
                stdout_callback=getattr(g, 'stdout_callback', None),
                stderr_callback=getattr(g, 'stderr_callback', None)
            )
        return python._session

    @staticmethod
    def run(content: str) -> str:
        """Executes the Python code using the persistent session."""
        try:
            session = python._get_session()
            session.execute(content)
            # The return value for the agent's context is minimal, as the
            # actual output is streamed live to the user via callbacks.
            return "Python code execution initiated. Output was streamed to the console."
        except Exception as e:
            logging.error(f"Error executing Python code: {e}", exc_info=True)
            return f"Error executing Python code: {e}"