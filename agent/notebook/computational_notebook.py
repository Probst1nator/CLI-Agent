from typing import Callable, Optional
from core.globals import g

class ComputationalNotebook:
    def __init__(self,
                 stdout_callback: Optional[Callable[[str], None]] = None,
                 stderr_callback: Optional[Callable[[str], None]] = None):

        self.stdout_callback = stdout_callback or (lambda text: print(text, end=''))
        self.stderr_callback = stderr_callback or (lambda text: print(f"STDERR: {text}", end=''))

        # Clean up old temporary files from previous runs
        import logging
        from termcolor import colored
        
        g.cleanup_temp_py_files()
        
        logging.info(colored("  - Computational Notebook coordinator ready.", "cyan"))

    def execute(self, command: str, is_python_code: bool = False, persist_python_state: bool = True):
        """
        DEPRECATED: Execute method is deprecated. 
        Use the individual bash and python tools instead for lazy loading.
        This method now delegates to the appropriate tool.
        """
        try:
            if is_python_code:
                # Import and use the python tool
                from tools.python import python
                return python.run(command)
            else:
                # Import and use the bash tool  
                from tools.bash import bash
                return bash.run(command)
        except Exception as e:
            self.stdout_callback(f"\n[Error delegating to tool: {str(e)}]\n")
            return f"Error: {e}"

    def send_input(self, text: str, session_type: str = "shell"):
        """
        DEPRECATED: send_input is deprecated.
        Access the session directly through the appropriate tool instead.
        """
        self.stdout_callback("[send_input is deprecated - access sessions through tools directly]\n")

    def close(self):
        """
        DEPRECATED: close is deprecated.
        Sessions are now managed by individual tools.
        """
        self.stdout_callback("\n[ComputationalNotebook coordinator closed]\n")