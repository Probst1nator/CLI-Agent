import pexpect
import tempfile
import os
import time
import re
from typing import Callable, List, Tuple, Optional

class ComputationalNotebook:
    def __init__(self,
                 stdout_callback: Optional[Callable[[str], None]] = None,
                 stderr_callback: Optional[Callable[[str], None]] = None, # pexpect often merges stderr
                 input_prompt_handler: Optional[Callable[[str, str], str]] = None):

        self.stdout_callback = stdout_callback or (lambda text: print(text, end=''))
        # pexpect primarily gives you stdout; stderr might be mixed or harder to get separately
        # without more complex fd manipulation.
        self.stderr_callback = stderr_callback or (lambda text: print(f"STDERR: {text}", end=''))
        self.input_prompt_handler = input_prompt_handler

        self.bash_prompt_regex = r"([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+:[^#$]*[#$]\s*)" # Common bash prompt
        self.python_repl_prompt_regex = r">>>\s*"
        self.python_repl_continuation_regex = r"\.\.\.\s*"

        self.child = pexpect.spawn('bash', encoding='utf-8', timeout=30) # Start bash
        self._expect_bash_prompt() # Wait for the first prompt

        self.python_interactive_active = False
        self.current_python_globals_file = None # For persisting Python state if not using -i

    def _expect_bash_prompt(self, timeout=30):
        self.child.expect(self.bash_prompt_regex, timeout=timeout)
        self.stdout_callback(self.child.before + self.child.after)

    def _expect_python_prompt(self, timeout=30):
        self.child.expect([self.python_repl_prompt_regex, self.python_repl_continuation_regex], timeout=timeout)
        self.stdout_callback(self.child.before + self.child.after)


    def _handle_interaction_loop(self,
                                 final_expected_pattern, # e.g., bash_prompt_regex or python_repl_prompt_regex
                                 timeout_seconds=30,
                                 input_patterns: Optional[List[str]] = None):
        """
        Manages the interaction loop, watching for expected output, handling input prompts,
        and calling stdout_callback.
        """
        if input_patterns is None:
            input_patterns = [
                # r"\[sudo\] password for .*: ",
                r"Overwrite\? \(y/N\) ",
                r"Do you want to continue\? \[Y/n\] ",
                r"Enter .*: ", # Generic input
                pexpect.TIMEOUT,
                pexpect.EOF
            ]
        
        expected_list = [final_expected_pattern] + input_patterns
        full_output = ""

        while True:
            try:
                index = self.child.expect(expected_list, timeout=timeout_seconds)
                current_output = self.child.before + self.child.after
                full_output += current_output
                self.stdout_callback(current_output)

                if index == 0: # Final expected pattern (e.g., bash prompt)
                    break
                elif index < len(expected_list) - 2: # Matched one of the input_patterns
                    prompt_text = self.child.after.strip()
                    # Send the captured output *before* the prompt to the handler
                    context_for_llm = full_output # or just self.child.before
                    response = self.input_prompt_handler(prompt_text, context_for_llm)
                    self.stdout_callback(f"\n[Sending input: {response}]\n")
                    self.child.sendline(response)
                    full_output = "" # Reset full_output after handling a prompt
                elif index == len(expected_list) - 2: # TIMEOUT
                    self.stdout_callback("\n[TIMEOUT during interaction]\n")
                    # Optionally, try to send a default input or break
                    if self.input_prompt_handler:
                        response = self.input_prompt_handler("TIMEOUT_PROMPT", full_output)
                        self.stdout_callback(f"\n[Sending input after timeout: {response}]\n")
                        self.child.sendline(response)
                        full_output = ""
                    else:
                        break # Or raise an exception
                elif index == len(expected_list) - 1: # EOF
                    self.stdout_callback("\n[EOF reached on child process]\n")
                    break
            except pexpect.exceptions.TIMEOUT:
                self.stdout_callback("\n[PEXPECT TIMEOUT in _handle_interaction_loop]\n")
                # Attempt to recover or re-prompt LLM
                if self.input_prompt_handler:
                    context_for_llm = full_output
                    response = self.input_prompt_handler("STALLED_EXECUTION_OR_UNKNOWN_PROMPT", context_for_llm)
                    self.stdout_callback(f"\n[Sending input after stall: {response}]\n")
                    self.child.sendline(response)
                    full_output = "" # Reset
                else:
                    break # Give up
            except pexpect.exceptions.EOF:
                self.stdout_callback("\n[PEXPECT EOF in _handle_interaction_loop]\n")
                break
        return full_output


    def execute(self, command: str, is_python_code: bool = False, persist_python_state: bool = True):
        """
        Executes a command. If is_python_code, it handles it specially.
        If persist_python_state is True and is_python_code, it tries to use an
        interactive Python session.
        """
        if is_python_code:
            if persist_python_state:
                if not self.python_interactive_active:
                    self.child.sendline("python3 -i") # Start interactive Python
                    self._handle_interaction_loop(self.python_repl_prompt_regex, input_patterns=[self.python_repl_continuation_regex])
                    self.python_interactive_active = True
                    self.stdout_callback("\n[Interactive Python session started]\n")

                # Write python code to a temp file to handle multi-line easily
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmpf:
                    tmpf.write(command)
                    py_script_path = tmpf.name

                # exec avoids issues with __main__ if scripts expect to be run directly
                # It also makes it easier to pass code to the interactive interpreter
                exec_command = f"exec(open('{py_script_path}', 'r').read())"
                self.child.sendline(exec_command)
                # Expect Python prompt or continuation, handle potential Python input()
                self._handle_interaction_loop(
                    self.python_repl_prompt_regex,
                    input_patterns=[
                        self.python_repl_continuation_regex,
                        r"Input required: ", # Example python input prompt
                        r"Enter your name: ",
                        pexpect.TIMEOUT,
                        pexpect.EOF
                    ]
                )
                os.remove(py_script_path)

            else: # Execute Python script non-interactively (state won't persist in interpreter)
                if self.python_interactive_active:
                    self.child.sendline("exit()") # Exit current interactive Python
                    self._expect_bash_prompt()
                    self.python_interactive_active = False
                    self.stdout_callback("\n[Interactive Python session exited]\n")

                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmpf:
                    tmpf.write(command)
                    py_script_path = tmpf.name

                self.child.sendline(f"python3 {py_script_path}")
                self._handle_interaction_loop(self.bash_prompt_regex) # Expect bash prompt after script finishes
                os.remove(py_script_path)
        else: # It's a regular Bash command
            if self.python_interactive_active:
                self.child.sendline("exit()") # Exit current interactive Python first
                self._expect_bash_prompt()
                self.python_interactive_active = False
                self.stdout_callback("\n[Interactive Python session exited to run Bash command]\n")

            self.child.sendline(command)

            self._handle_interaction_loop(self.bash_prompt_regex)

    def close(self):
        if self.python_interactive_active:
            self.child.sendline("exit()") # Exit Python
            try:
                self._expect_bash_prompt(timeout=5)
            except pexpect.exceptions.TIMEOUT:
                self.stdout_callback("\n[Timeout waiting for bash prompt after Python exit()]\n")

        self.child.sendline("exit") # Exit Bash
        self.child.close()
        self.stdout_callback("\n[Interactive Bash session closed]\n")

# --- Example Usage ---
if __name__ == "__main__":
    def my_stdout(text):
        print(f"STDOUT> {text}", end='')

    bash_env = ComputationalNotebook(stdout_callback=my_stdout)

    print("\n--- Running ls ---")
    bash_env.execute("ls -l /nonexistentdir") # Example of a command that might output to stderr (handled by pexpect merging)

    print("\n--- Running a Python script that needs input (with persistence) ---")
    python_code_1 = """
name = input("Enter your name: ")
print(f"Hello, {name}!")
my_variable = 42
print(f"Set my_variable to {my_variable}")
"""
    bash_env.execute(python_code_1, is_python_code=True, persist_python_state=True)

    print("\n--- Running another Python script (with persistence, accessing previous state) ---")
    python_code_2 = """
try:
    print(f"my_variable from previous execution is: {my_variable}")
    my_variable += 10
    print(f"Incremented my_variable to {my_variable}")
except NameError:
    print("my_variable was not found from previous execution.")
new_var = "Python world"
print(f"Set new_var to '{new_var}'")
"""
    bash_env.execute(python_code_2, is_python_code=True, persist_python_state=True)

    print("\n--- Running a sudo command (will trigger input prompt) ---")
    # This will likely fail without a real password or proper sudoers setup
    # For demonstration, it will hit the sudo prompt.
    bash_env.execute("sudo -A echo 'Sudo test'") # Using -A to ensure it asks if no askpass

    print("\n--- Running Python script that doesn't persist state in interpreter ---")
    python_code_no_persist = """
import os
print(f"Current PID (no persist): {os.getpid()}")
# This my_variable would be new, not from the interactive session
try:
    print(f"my_variable (no persist): {my_variable}")
except NameError:
    print("my_variable not found in this non-persistent script.")
"""
    bash_env.execute(python_code_no_persist, is_python_code=True, persist_python_state=False)

    print("\n--- Checking Python interactive state again after non-persistent script ---")
    # Need to re-run code in the persistent session if one was active
    bash_env.execute("print(f'my_variable in persistent session: {my_variable}')", is_python_code=True, persist_python_state=True)


    bash_env.close()