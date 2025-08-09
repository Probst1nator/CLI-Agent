import datetime
import pexpect
import os
import time
from typing import Callable, Optional
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

        # Initialize bash session
        self.child = pexpect.spawn('bash', encoding='utf-8', timeout=30)
        self._expect_bash_prompt(suppress_output=True)  # Suppress initial bash prompt
        
        # Initialize persistent Python session
        self.python_child = None
        self._initialize_persistent_python()

    def _initialize_persistent_python(self):
        """Initialize persistent Python session."""
        try:
            if self.python_child:
                self.python_child.close()
            
            # Spawn persistent Python session with interactive mode
            self.python_child = pexpect.spawn('python3 -i -u', encoding='utf-8', timeout=30)
            
            # Wait for initial Python prompt
            self.python_child.expect('>>> ', timeout=10)
            
            # Send initialization code line by line
            init_lines = self.get_initialization_code().strip().split('\n')
            for line in init_lines:
                if line.strip():  # Skip empty lines
                    self.python_child.sendline(line)
                    # Wait for prompt or continuation
                    try:
                        index = self.python_child.expect(['>>> ', '\.\.\. '], timeout=10)
                        if index == 1:  # Continuation prompt, send empty line to complete
                            self.python_child.sendline('')
                            self.python_child.expect('>>> ', timeout=10)
                    except pexpect.TIMEOUT:
                        # If timeout, try to recover
                        self.python_child.sendline('')
                        try:
                            self.python_child.expect('>>> ', timeout=5)
                        except pexpect.TIMEOUT:
                            pass  # Continue anyway
            
        except Exception as e:
            self.stdout_callback(f"\n[Failed to initialize persistent Python session: {e}]\n")
            self.python_child = None
    
    def _suppress_utility_return_values(self, code: str) -> str:
        """Modify code to suppress return values from utility function calls."""
        import re
        
        lines = code.strip().split('\n')
        processed_lines = []
        
        # List of utility modules that return JSON but also print
        utility_modules = [
            'viewfiles', 'editfile', 'web_fetch', 'searchweb', 'todos', 'showuser',
            'removefile', 'takescreenshot', 'generateimage', 'viewimage', 
            'homeassistant', 'tts', 'architectnewutil'
        ]
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check if this line calls a utility function directly (not assigned)
            if stripped_line and not stripped_line.startswith('#'):
                # Check if it's a standalone utility call (not already assigned)
                for util_module in utility_modules:
                    pattern = fr'^{util_module}\.run\s*\('
                    if re.match(pattern, stripped_line) and '=' not in stripped_line.split(util_module)[0]:
                        # This is a standalone utility call - suppress its return value
                        processed_lines.append(f"_ = {line}")
                        break
                else:
                    # Not a utility call or already assigned
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)

    def _execute_python_in_persistent_session(self, command: str):
        """Execute Python code in the persistent interactive Python session."""
        try:
            if not self.python_child or not self.python_child.isalive():
                self._initialize_persistent_python()
            
            if not self.python_child:
                raise Exception("Failed to initialize Python session")
            
            # Suppress utility return values to prevent REPL output clutter
            processed_command = self._suppress_utility_return_values(command)
            
            # Execute multi-line code properly using exec()
            collected_output = []
            
            # For multi-line code, wrap it in exec() to execute as a complete block
            if '\n' in processed_command.strip() and len(processed_command.strip().split('\n')) > 1:
                # Multi-line code - use compile and exec for better error handling
                try:
                    # Test if code can be compiled first
                    compile(processed_command, '<string>', 'exec')
                    # Use base64 encoding to avoid any string escaping issues
                    import base64
                    encoded_command = base64.b64encode(processed_command.encode('utf-8')).decode('ascii')
                    exec_command = f"import base64; exec(base64.b64decode('{encoded_command}').decode('utf-8'))"
                    self.python_child.sendline(exec_command)
                except SyntaxError as e:
                    self.stdout_callback(f"[Syntax Error: {e}]\\n")
                    return
            else:
                # Single line code - send directly
                self.python_child.sendline(processed_command)
            
            # Brief pause to let Python process the command
            time.sleep(0.1)
            
            try:
                # Wait for completion with interactive input support
                start_time = time.time()
                last_output_time = time.time()
                stall_threshold = 10
                all_python_output = ""
                
                while time.time() - start_time < 10:
                    try:
                        index = self.python_child.expect(['>>> ', pexpect.TIMEOUT], timeout=1)
                        
                        if index == 0:  # Got prompt - done
                            if self.python_child.before:
                                output = str(self.python_child.before).strip()
                                # Filter out the command echo and exec artifacts
                                if output:
                                    output_lines = output.split('\n')
                                    filtered_lines = []
                                    command_lines = processed_command.split('\n')
                                    
                                    for out_line in output_lines:
                                        out_line_stripped = out_line.strip()
                                        # Skip if it's exactly one of the command lines we sent
                                        if out_line_stripped and out_line_stripped not in command_lines:
                                            # Also filter out exec wrapper artifacts and continuation prompts
                                            if not (out_line_stripped.startswith("exec(") or 
                                                   out_line_stripped == "..." or
                                                   out_line_stripped.endswith(")") and "exec(" in out_line_stripped or
                                                   out_line_stripped.startswith("... ") or
                                                   (out_line_stripped.startswith("...") and len(out_line_stripped.strip("... ")) <= 5)):
                                                filtered_lines.append(out_line)
                                    
                                    if filtered_lines:
                                        collected_output.extend(filtered_lines)
                            break
                            
                        elif index == 1:  # TIMEOUT - still running
                            new_output = self.python_child.before if self.python_child.before else ""
                            if new_output:
                                new_text = new_output.replace(all_python_output, "")
                                if new_text.strip():
                                    all_python_output += new_text
                                    last_output_time = time.time()
                            
                            # Check for stall and handle input
                            time_since_last_output = time.time() - last_output_time
                            if time_since_last_output > stall_threshold:
                                if self.input_prompt_handler:
                                    decision = self.input_prompt_handler(all_python_output, "python")
                                    if decision is True:  # Wait longer
                                        last_output_time = time.time()
                                        stall_threshold = min(stall_threshold * 1.2, 60)
                                    elif decision is False:  # Interrupt
                                        self.python_child.sendcontrol('c')
                                        break
                                    elif isinstance(decision, str):  # Send input
                                        try:
                                            self.python_child.sendline(decision)
                                            all_python_output = ""  # Reset context
                                            last_output_time = time.time()
                                        except Exception:
                                            break
                                else:
                                    last_output_time = time.time()
                    
                    except pexpect.TIMEOUT:
                        break
                
                # Final collection of any remaining output
                if all_python_output and all_python_output not in collected_output:
                    collected_output.append(all_python_output)
                            
            except pexpect.TIMEOUT:
                # Handle major timeout
                self.stdout_callback("[Python execution timed out]\n")
                # Try to interrupt and recover
                self.python_child.sendcontrol('c')
                try:
                    self.python_child.expect('>>> ', timeout=2)
                except pexpect.TIMEOUT:
                    # Reinitialize session if we can't recover
                    self._initialize_persistent_python()
                return
            
            # Output all collected results
            if collected_output:
                final_output = '\n'.join(collected_output)
                if final_output.strip():
                    self.stdout_callback(final_output + '\n')
                    
        except Exception as e:
            self.stdout_callback(f"\n[Error in persistent Python execution: {str(e)}]\n")
            # Try to reinitialize session on error
            self._initialize_persistent_python()


    def _expect_bash_prompt(self, timeout=30, suppress_output=False):
        self.child.expect(self.bash_prompt_regex, timeout=timeout)
        # Only include command output, not the prompt itself
        output = str(self.child.before).strip()
        if not suppress_output and output:
            self.stdout_callback(output + '\n')

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
        except Exception:
            pass  # Ignore any other exceptions during cleanup
        self.child.before = ""

        while time.time() - start_time < timeout:
            try:
                # Expect either the prompt or a timeout
                index = self.child.expect([self.bash_prompt_regex, pexpect.TIMEOUT], timeout=1)

                if index == 0:
                    # Matched the bash prompt. The command is finished.
                    # Only include the command output, not the prompt itself
                    final_output = self.child.before.strip()
                    if final_output:
                        # Process the complete output with emojis
                        processed_output = self._process_output_with_emoji(final_output)
                        self.stdout_callback(processed_output)
                    break
                elif index == 1:
                    # A timeout occurred. This means the command is still running and has produced
                    # some output, but not the final prompt yet.
                    new_output = self.child.before.replace(all_output_for_context, "")
                    if new_output:
                        processed_output = self._process_output_with_emoji(new_output)
                        self.stdout_callback(processed_output)
                        all_output_for_context += new_output
                        last_output_time = time.time()
                    
                    # Check for a stall.
                    time_since_last_output = time.time() - last_output_time
                    if time_since_last_output > stall_threshold:
                        if self.input_prompt_handler:
                            decision = self.input_prompt_handler(all_output_for_context, "shell")
                            if decision is True: # Wait longer
                                last_output_time = time.time()
                                stall_threshold = min(stall_threshold * 1.2, 120)
                            elif decision is False: # Interrupt
                                self._safe_interrupt()
                                break
                            elif isinstance(decision, str): # Send input
                                try:
                                    self.child.sendline(decision)
                                    all_output_for_context = "" # Reset context after input
                                    last_output_time = time.time()
                                except Exception:
                                    break  # If we can't send input, exit gracefully
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
            except KeyboardInterrupt:
                # Handle Ctrl+C during streaming
                self._safe_interrupt()
                break
            except Exception as e:
                # Catch any other exceptions and exit gracefully
                self.stdout_callback(f"\n[Error during execution: {str(e)}]\n")
                break

    def _safe_interrupt(self):
        """Safely interrupt the current process and restore a clean prompt."""
        try:
            # Send Ctrl+C to interrupt the process
            self.child.sendcontrol('c')
            self.stdout_callback("\n[Process interrupted]\n")
            
            # Give the process a moment to respond to the interrupt
            time.sleep(0.5)
            
            # Try to get back to a clean prompt
            try:
                # Send a newline and wait for prompt with a short timeout
                self.child.sendline('')
                self.child.expect(self.bash_prompt_regex, timeout=3)
            except (pexpect.TIMEOUT, pexpect.EOF):
                # If we can't get a clean prompt, try sending another interrupt
                try:
                    self.child.sendcontrol('c')
                    self.child.sendline('')
                    self.child.expect(self.bash_prompt_regex, timeout=2)
                except Exception:
                    # If all else fails, just continue - the next command will handle it
                    pass
                    
        except Exception:
            # If interrupt fails, just continue - we'll handle it in the next command
            pass

    def get_initialization_code(self) -> str:
        """Return initialization code for the sandbox as a single string."""
        # Dynamically import all available utilities
        utils_imports = self._get_utils_imports()
        
        return f"""
import sys, os, json, io, datetime
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.chdir('{os.getcwd()}')
sys.path.append('{g.CLIAGENT_ROOT_PATH}')
from py_classes.globals import g
from utils import *

# Import all utility modules for easy access
{utils_imports}
"""

    def _get_utils_imports(self) -> str:
        """Generate import statements for all available utility modules."""
        try:
            import glob
            import os
            
            utils_path = os.path.join(g.CLIAGENT_ROOT_PATH, 'utils')
            imports = []
            
            # Find all Python files in utils directory
            for py_file in glob.glob(os.path.join(utils_path, '*.py')):
                filename = os.path.basename(py_file)
                
                # Skip __init__.py and private files
                if filename.startswith('_') or filename == '__init__.py':
                    continue
                    
                module_name = filename[:-3]  # Remove .py extension
                
                # Check if the module has a run function
                try:
                    module_path = f"utils.{module_name}"
                    import importlib
                    module = importlib.import_module(module_path)
                    if hasattr(module, 'run'):
                        imports.append(f"import utils.{module_name} as {module_name}")
                except (ImportError, AttributeError):
                    continue
            
            return '\n'.join(imports)
            
        except Exception:
            # Fallback to manual list if dynamic discovery fails
            return """
import utils.viewfiles as viewfiles
import utils.editfile as editfile  
import utils.web_fetch as web_fetch
import utils.searchweb as searchweb
import utils.todos as todos
import utils.showuser as showuser
import utils.removefile as removefile
import utils.takescreenshot as takescreenshot
import utils.generateimage as generateimage
import utils.viewimage as viewimage
import utils.homeassistant as homeassistant
import utils.tts as tts
import utils.architectnewutil as architectnewutil
"""

    def _fix_python_indentation(self, code: str) -> str:
        """Fix common Python indentation issues in LLM-generated code."""
        try:
            import ast
            import re
            
            # First: try original code
            try:
                ast.parse(code)
                return code
            except:
                pass
            
            # Second: simple approach - use autopep8 style indentation normalization
            # This handles the common case where LLMs mix 2/3/4 space indentation
            lines = code.strip().split('\n')
            
            # Pass 1: convert tabs to spaces
            lines = [line.expandtabs(4) for line in lines]
            
            # Pass 2: detect and normalize indentation levels
            normalized = []
            indent_level = 0
            
            for line in lines:
                if not line.strip():  # Empty line
                    normalized.append('')
                    continue
                
                content = line.lstrip()
                
                # Determine if we should increase, maintain, or decrease indentation
                if any(content.startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', '@']):
                    # Top-level - reset to 0
                    indent_level = 0
                elif content.startswith(('else:', 'elif ', 'except', 'finally:')):
                    # These dedent back to their matching block level
                    indent_level = max(0, indent_level - 4)
                elif re.match(r'^(return|break|continue|pass|raise)\b', content):
                    # These should be at least indented (inside a function/loop)
                    if indent_level == 0:
                        indent_level = 4
                
                # Apply current indentation
                normalized.append(' ' * indent_level + content)
                
                # If this line ends with ':', next line should be indented more
                if content.rstrip().endswith(':'):
                    indent_level += 4
            
            result = '\n'.join(normalized)
            
            # Test the result
            try:
                ast.parse(result)
                return result
            except:
                pass
            
            # Final fallback: try the original autopep8 approach of using only even indentation levels
            lines = code.strip().split('\n')
            even_indented = []
            
            for line in lines:
                if not line.strip():
                    even_indented.append('')
                    continue
                    
                # Get current indentation
                indent_chars = len(line) - len(line.lstrip())
                
                # Round to nearest multiple of 4
                new_indent = ((indent_chars + 2) // 4) * 4
                
                even_indented.append(' ' * new_indent + line.lstrip())
            
            final_result = '\n'.join(even_indented)
            
            try:
                ast.parse(final_result)
                return final_result
            except:
                # Ultimate fallback
                return code
                
        except Exception:
            return code

    def execute(self, command: str, is_python_code: bool = False, persist_python_state: bool = True):
        """Execute a command with real-time output streaming. Never raises exceptions."""
        try:
            if is_python_code:
                try:
                    # Fix indentation issues in Python code before execution
                    cleaned_command = self._fix_python_indentation(command)
                    
                    # Set flag before sending command so the echoed command also gets emoji
                    self.is_executing_python = True
                    
                    if persist_python_state:
                        # Use persistent interactive Python session
                        self._execute_python_in_persistent_session(cleaned_command)
                    else:
                        # Non-persistent execution - use temporary script
                        py_script_path = os.path.join(g.CLIAGENT_TEMP_STORAGE_PATH, f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.py')
                        with open(py_script_path, 'w') as py_file:
                            py_file.write(self.get_initialization_code() + "\n" + cleaned_command)
                        self.child.sendline(f"python3 -u {py_script_path}")
                        self._stream_output_until_prompt(timeout=600)  # Increased to 10 minutes for AI model operations
                except Exception as e:
                    self.stdout_callback(f"\n[Error executing Python code: {str(e)}]\n")
                finally:
                    self.is_executing_python = False
                    
            else:
                try:
                    # Set shell execution flag for bash commands
                    self.is_executing_shell = True
                    self.child.sendline(command)
                    self._stream_output_until_prompt(timeout=300)  # Increased to 5 minutes for shell commands
                except Exception as e:
                    self.stdout_callback(f"\n[Error executing shell command: {str(e)}]\n")
                finally:
                    self.is_executing_shell = False
                    
        except KeyboardInterrupt:
            # Handle Ctrl+C at the top level
            self._safe_interrupt()
            # Reset flags to avoid incorrect emoji prefixes on subsequent outputs
            self.is_executing_python = False
            self.is_executing_shell = False
            
        except Exception as e:
            # Catch any other unexpected exceptions
            self.stdout_callback(f"\n[Unexpected error during execution: {str(e)}]\n")
            # Reset flags to avoid incorrect emoji prefixes on subsequent outputs
            self.is_executing_python = False
            self.is_executing_shell = False

    def send_input(self, text: str, session_type: str = "shell"):
        """Send input directly to the pexpect stream.
        
        Args:
            text: The text to send
            session_type: Either "shell" for bash session or "python" for Python session
        """
        try:
            if session_type == "python" and self.python_child and self.python_child.isalive():
                self.python_child.sendline(text)
            elif session_type == "shell" and self.child and self.child.isalive():
                self.child.sendline(text)
            else:
                self.stdout_callback(f"[Error: {session_type} session not available]\n")
        except Exception as e:
            self.stdout_callback(f"[Error sending input: {str(e)}]\n")

    def close(self):
        """Close both bash and Python sessions."""
        # Close Python session first
        if self.python_child and self.python_child.isalive():
            try:
                self.python_child.sendline("exit()")
                self.python_child.expect(pexpect.EOF, timeout=5)
            except (pexpect.TIMEOUT, pexpect.EOF):
                pass
            finally:
                self.python_child.close()
        
        # Close bash session
        if self.child and self.child.isalive():
            try:
                self.child.sendline("exit")
                self.child.close()
            except Exception:
                pass
        
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