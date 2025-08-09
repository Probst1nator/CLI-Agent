# File: globals.py
import os
import shutil
import socket
from typing import List, Optional, Any, Callable, Dict
import logging
import builtins
from termcolor import colored
import json

from py_classes.enum_ai_strengths import AIStrengths


class Globals:
    FORCE_LOCAL: bool = False
    FORCE_ONLINE: bool = False
    FORCE_FAST: bool = False
    DEBUG_CHATS: bool = False
    USE_SANDBOX: bool = False
    LLM: Optional[str] = None
    MCT: Optional[int] = None
    LLM_STRENGTHS: List[AIStrengths] = []
    SELECTED_UTILS: List[str] = []  # Store selected utilities
    SELECTED_LLMS: List[str] = []  # Store selected LLMs
    SSH_CONNECTION: Optional[str] = None  # Store SSH connection details (user@hostname:port)
    
    # Ollama host configuration
    ollama_host_env: List[str] = []
    
    # Configuration settings
    _user_config: Dict[str, Any] = {}
    
    if (os.getenv("USE_ONLINE_HOSTNAME", "") == socket.gethostname()):
        LLM_STRENGTHS = [AIStrengths.ONLINE]
    
    CLIAGENT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLIAGENT_PERSISTENT_STORAGE_PATH = os.path.join(CLIAGENT_ROOT_PATH, '.cliagent')
    CLIAGENT_TEMP_STORAGE_PATH = os.path.join(CLIAGENT_PERSISTENT_STORAGE_PATH, '.temp')
    CLIAGENT_ENV_FILE_PATH = os.path.join(CLIAGENT_ROOT_PATH, '.env')
    # Path to the system instruction Markdown file
    CLIAGENT_INSTRUCTION_FILE: str = os.path.join(CLIAGENT_PERSISTENT_STORAGE_PATH, 'CLIAGENT.md')
    AGENTS_SANDBOX_DIR = os.path.join(CLIAGENT_PERSISTENT_STORAGE_PATH, "agents_sandbox")
    os.makedirs(AGENTS_SANDBOX_DIR, exist_ok=True)
    
    # User configuration file path
    USER_CONFIG_PATH = os.path.join(CLIAGENT_PERSISTENT_STORAGE_PATH, 'user_config.json')
    
    # Model limits
    # Generate a daily model token limits file path with date suffix
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    PROJ_MODEL_LIMITS_PATH = os.path.join(CLIAGENT_PERSISTENT_STORAGE_PATH, 'model_limits')
    os.makedirs(PROJ_MODEL_LIMITS_PATH, exist_ok=True)
    MODEL_TOKEN_LIMITS_PATH = os.path.join(PROJ_MODEL_LIMITS_PATH, f'{today}_model_token_limits.json')
    MODEL_RATE_LIMITS_PATH = os.path.join(PROJ_MODEL_LIMITS_PATH, f'{today}_model_rate_limits.json')
    
    # Finetuning
    UNCONFIRMED_FINETUNING_PATH = os.path.join(CLIAGENT_TEMP_STORAGE_PATH, 'unconfirmed_finetuning_data')
    CONFIRMED_FINETUNING_PATH = os.path.join(CLIAGENT_PERSISTENT_STORAGE_PATH, 'confirmed_finetuning_data')
    
    # Web server instance
    web_server = None
    
    # Store the original print function
    original_print: Callable = builtins.print

    os.makedirs(CLIAGENT_PERSISTENT_STORAGE_PATH, exist_ok=True)
    
    if os.path.exists(CLIAGENT_TEMP_STORAGE_PATH):
        shutil.rmtree(CLIAGENT_TEMP_STORAGE_PATH)
    os.makedirs(CLIAGENT_TEMP_STORAGE_PATH, exist_ok=True)
    
    def load_user_config(self) -> Dict[str, Any]:
        """Load user configuration from JSON file."""
        try:
            if os.path.exists(self.USER_CONFIG_PATH):
                with open(self.USER_CONFIG_PATH, 'r') as f:
                    self._user_config = json.load(f)
            else:
                # Initialize with default values
                self._user_config = {}
                self.save_user_config()
        except Exception as e:
            logging.error(f"Failed to load user config: {e}")
            self._user_config = {}
        return self._user_config
    
    def save_user_config(self) -> None:
        """Save user configuration to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.USER_CONFIG_PATH), exist_ok=True)
            with open(self.USER_CONFIG_PATH, 'w') as f:
                json.dump(self._user_config, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save user config: {e}")
    
    def get_config_value(self, key: str, default_value: Any = None) -> Any:
        """Get a configuration value with fallback to default."""
        if not self._user_config:
            self.load_user_config()
        return self._user_config.get(key, default_value)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set a configuration value and save to file."""
        if not self._user_config:
            self.load_user_config()
        self._user_config[key] = value
        self.save_user_config()
    
    def cleanup_temp_py_files(self):
        """Remove temporary Python files from previous runs."""
        import re
        try:
            for f in os.listdir(self.CLIAGENT_TEMP_STORAGE_PATH):
                if f.endswith('.py') and re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.py$', f):
                    os.remove(os.path.join(self.CLIAGENT_TEMP_STORAGE_PATH, f))
        except:
            pass
    
    def debug_log(self, message: str, color: str = None, end: str = '\n', 
                  with_title: bool = True, is_error: bool = False, force_print: bool = False,
                  prefix: str = "") -> None:
        """
        A centralized debug log function to replace the various debug_print and log_print functions.
        
        Args:
            message (str): The message to print
            color (str, optional): Color for the message
            end (str): End character
            with_title (bool): Whether to include the chat title
            is_error (bool): Whether this is an error message
            force_print (bool): Force printing to console even for info messages
            prefix (str): Prefix to add before the message (replaces chat.get_debug_title_prefix())
        """
        does_contain_whitespace = prefix.endswith(" ") or message.startswith(" ")
        if does_contain_whitespace:
            log_message = f"{prefix}" + message.removeprefix('\n')
        else:
            log_message = f"{prefix} " + message.removeprefix('\n')
        
        # Log to appropriate logger level (ignoring color)
        if is_error:
            # logging.error(log_message)
            # For errors, always print to console
            if color:
                print(colored(log_message, color), end=end)
            else:
                print(log_message, end=end)
        else:
            # For info level, log to logger
            # logging.info(log_message)
            # Only print to console if forced
            if force_print:
                if color:
                    print(colored(log_message, color), end=end)
                else:
                    print(log_message, end=end)
    
    def print_token(self, token: str, color_func = None) -> None:
        """
        Print a token with optional coloring for stream processing.
        
        Args:
            token (str): The token to print
            color_func: Function to apply color to the token
        """
        if color_func:
            print(color_func(token), end="", flush=True)
        else:
            print(token, end="", flush=True)
    
g = Globals()

# Initialize user configuration
g.load_user_config()

# Initialize Ollama host environment variable
ollama_host_env = os.getenv("OLLAMA_HOST", "")
if ollama_host_env:
    g.ollama_host_env = ollama_host_env.split(",")
else:
    g.ollama_host_env = []

def configure_logging():
    # Configure root logger 
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)  # Only show ERROR and above in console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler for all logs
    os.makedirs(g.CLIAGENT_PERSISTENT_STORAGE_PATH, exist_ok=True)
    log_file_path = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, 'app.log') 
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)  # Still log INFO and above to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set overall logger level - this needs to be the lowest level you want to capture
    root_logger.setLevel(logging.INFO)
    
    # Silence specific loggers that are too verbose
    logging.getLogger('py_classes.cls_llm_router').setLevel(logging.ERROR)
    
    logging.info("Logging configured")

# Create a custom print function that also logs to web interface
def custom_print(*args: Any, **kwargs: Any) -> None:
    """
    Custom print function that intercepts print calls and also sends them to the web interface.
    This function preserves all functionality of the original print function.
    """
    # Call the original print function to preserve normal console output
    g.original_print(*args, **kwargs)
    
    # If web server is initialized and GUI mode is enabled, log to web interface
    if g.web_server is not None:
        try:
            g.web_server.log_print(*args, **kwargs)
        except Exception:
            # In case of any error, fall back to the original print function
            pass

# Configure logging after Globals is instantiated
configure_logging()

# Only replace the print function after everything is set up
builtins.print = custom_print