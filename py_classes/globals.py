# py_classes/globals.py
"""
This file defines a singleton `g` object to hold global state for the CLI-Agent application.
Using a class-based singleton pattern ensures that all parts of the application
share the same state instance, which is crucial for managing settings like
LLM selection, force flags, and other runtime configurations.
"""
import os
import shutil
import json
import logging
import datetime
from pathlib import Path
from typing import List, Optional, Any, Callable, Dict
from termcolor import colored

# --- Helper Function for Path Management ---
def _get_persistent_storage_path() -> str:
    """
    Determines the appropriate path for persistent storage based on the OS.
    This helps in keeping user-specific data and configurations in a conventional location.
    """
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ.get('APPDATA', ''), 'cli-agent')
    else:  # macOS, Linux, and other UNIX-like systems
        return os.path.join(Path.home(), '.cli-agent')

# --- Main Globals Class ---
class Globals:
    """
    A singleton class to hold and manage the global state of the application.
    This includes paths, runtime flags, and selected model configurations.
    """
    def __init__(self):
        # --- Path Configurations ---
        self.CLIAGENT_ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.CLIAGENT_PERSISTENT_STORAGE_PATH: str = _get_persistent_storage_path()
        self.CLIAGENT_TEMP_STORAGE_PATH: str = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, ".temp")
        self.AGENTS_SANDBOX_DIR: str = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, "sandbox")
        
        # --- File Path Configurations ---
        self.CLIAGENT_ENV_FILE_PATH: str = os.path.join(self.CLIAGENT_ROOT_PATH, ".env")
        self.USER_CONFIG_PATH: str = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, 'user_config.json')
        self.LLM_CONFIG_PATH: str = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, 'llm_config.json')
        
        # --- Model Limit Paths with Daily Rotation ---
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        model_limits_dir = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, 'model_limits')
        os.makedirs(model_limits_dir, exist_ok=True)
        self.MODEL_TOKEN_LIMITS_PATH: str = os.path.join(model_limits_dir, f'{today}_model_token_limits.json')
        self.MODEL_RATE_LIMITS_PATH: str = os.path.join(model_limits_dir, f'{today}_model_rate_limits.json')
        
        self.UNCONFIRMED_FINETUNING_PATH: str = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, "finetuning_data", "unconfirmed")
        self.CONFIRMED_FINETUNING_PATH: str = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, "finetuning_data", "confirmed")

        # --- LLM and Agent Configuration ---
        self.LLM: Optional[str] = None
        self.SELECTED_LLMS: List[str] = []
        self.EVALUATOR_LLMS: List[str] = []
        self.MCT: int = 1
        self.DEFAULT_OLLAMA_HOSTS: List[str] = ["http://localhost:11434"]
        
        # --- Forcing Flags (runtime modifiers) ---
        from py_classes.enum_ai_strengths import AIStrengths # Local import
        self.FORCE_LOCAL: bool = False
        self.FORCE_ONLINE: bool = False
        self.FORCE_FAST: bool = False
        self.FORCE_STRONG: bool = False
        self.LLM_STRENGTHS: List[AIStrengths] = []

        # --- Debug and UI Flags ---
        self.DEBUG_CHATS: bool = False
        self.USE_SANDBOX: bool = False
        
        # --- Utility and Tool Management ---
        self.SELECTED_UTILS: List[str] = []

        # --- Output Truncation Settings ---
        self.OUTPUT_TRUNCATE_HEAD_SIZE: int = 2000
        self.OUTPUT_TRUNCATE_TAIL_SIZE: int = 2000

        # --- Cross-module Communication & State ---
        self.web_server: Optional[Any] = None
        self.print_token: Callable[[str], None] = lambda token: print(token, end="", flush=True)
        self.debug_log: Callable[..., None] = self._default_debug_log
        self._user_config: Dict[str, Any] = {}
        self._llm_config: Dict[str, Any] = {}

        # --- Initial Setup on Instantation ---
        self._initialize_directories()
        self.load_user_config()
        self.load_llm_config() # Load LLM config at startup
        self._configure_ollama_hosts()

    def _initialize_directories(self):
        """Create necessary directories and clean up temporary ones."""
        os.makedirs(self.CLIAGENT_PERSISTENT_STORAGE_PATH, exist_ok=True)
        os.makedirs(self.AGENTS_SANDBOX_DIR, exist_ok=True)
        
        if os.path.exists(self.CLIAGENT_TEMP_STORAGE_PATH):
            shutil.rmtree(self.CLIAGENT_TEMP_STORAGE_PATH)
        os.makedirs(self.CLIAGENT_TEMP_STORAGE_PATH, exist_ok=True)

    def _configure_ollama_hosts(self):
        """Load Ollama hosts from environment variables."""
        ollama_host_env = os.getenv("OLLAMA_HOSTS")
        if ollama_host_env:
            hosts = [host.strip() for host in ollama_host_env.split(',') if host.strip()]
            if hosts:
                self.DEFAULT_OLLAMA_HOSTS = hosts

    def _default_debug_log(self, message: str, color: str = None, force_print: bool = False, **kwargs):
        """A simple default logger in case the main one isn't set up yet."""
        if force_print:
            if color:
                print(colored(f"DEBUG: {message}", color))
            else:
                print(f"DEBUG: {message}")

    def load_user_config(self) -> None:
        """Load user configuration from JSON file."""
        if os.path.exists(self.USER_CONFIG_PATH):
            try:
                with open(self.USER_CONFIG_PATH, 'r') as f:
                    self._user_config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Could not load user config file, it may be corrupt: {e}")
                self._user_config = {}
        else:
            self._user_config = {}

    def save_user_config(self) -> None:
        """Save the current user configuration to a JSON file."""
        try:
            with open(self.USER_CONFIG_PATH, 'w') as f:
                json.dump(self._user_config, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save user config: {e}")

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a fallback default."""
        return self._user_config.get(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set a configuration value and save it to the file immediately."""
        self._user_config[key] = value
        self.save_user_config()

    def load_llm_config(self) -> None:
        """Load the LLM selection/configuration from its JSON file."""
        if os.path.exists(self.LLM_CONFIG_PATH):
            try:
                with open(self.LLM_CONFIG_PATH, 'r') as f:
                    self._llm_config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Could not load LLM config file, it may be corrupt: {e}")
                self._llm_config = {}
        else:
            self._llm_config = {}

    def get_llm_config(self) -> Dict[str, Any]:
        """Get the loaded LLM configuration."""
        return self._llm_config

    def cleanup_temp_py_files(self):
        """Remove temporary Python files from previous runs."""
        import re
        try:
            if not os.path.exists(self.CLIAGENT_TEMP_STORAGE_PATH):
                return
            for f in os.listdir(self.CLIAGENT_TEMP_STORAGE_PATH):
                if f.endswith('.py') and re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.py$', f):
                    os.remove(os.path.join(self.CLIAGENT_TEMP_STORAGE_PATH, f))
        except (OSError, IOError) as e:
            logging.warning(f"Could not clean up temporary files: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during temp file cleanup: {e}")

g = Globals()