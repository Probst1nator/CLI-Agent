DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"
# py_classes/globals.py
"""
This file defines a singleton `g` object to hold global state for the CLI-Agent application.
Using a class-based singleton pattern ensures that all parts of the application
share the same state instance, which is crucial for managing settings like
LLM selection, force flags, and other runtime configurations.
"""
import os
import atexit
import shutil
import json
import logging
import datetime
from pathlib import Path
from typing import List, Optional, Any, Callable, Dict, Set, Protocol
from typing_extensions import TypedDict, NotRequired
from termcolor import colored

# --- Helper Functions for XDG Base Directory Path Management ---
def _get_xdg_config_dir() -> str:
    """Get XDG config directory for CLI-Agent configurations."""
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ.get('APPDATA', ''), 'cli-agent')
    else:  # macOS, Linux, and other UNIX-like systems
        return os.path.join(os.environ.get('XDG_CONFIG_HOME', os.path.join(Path.home(), '.config')), 'cli-agent')

def _get_xdg_data_dir() -> str:
    """Get XDG data directory for CLI-Agent persistent data."""
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ.get('LOCALAPPDATA', ''), 'cli-agent')
    else:  # macOS, Linux, and other UNIX-like systems
        return os.path.join(os.environ.get('XDG_DATA_HOME', os.path.join(Path.home(), '.local', 'share')), 'cli-agent')

def _get_xdg_cache_dir() -> str:
    """Get XDG cache directory for CLI-Agent cache and temporary files."""
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ.get('TEMP', ''), 'cli-agent')
    else:  # macOS, Linux, and other UNIX-like systems
        return os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.join(Path.home(), '.cache')), 'cli-agent')

def _get_xdg_state_dir() -> str:
    """Get XDG state directory for CLI-Agent ephemeral state data."""
    if os.name == 'nt':  # Windows - use same as data dir on Windows
        return os.path.join(os.environ.get('LOCALAPPDATA', ''), 'cli-agent', 'state')
    else:  # macOS, Linux, and other UNIX-like systems
        return os.path.join(os.environ.get('XDG_STATE_HOME', os.path.join(Path.home(), '.local', 'state')), 'cli-agent')

def _get_persistent_storage_path() -> str:
    """
    Legacy function for backward compatibility.
    Returns the old storage path for migration purposes.
    """
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ.get('APPDATA', ''), 'cli-agent')
    else:  # macOS, Linux, and other UNIX-like systems
        return os.path.join(Path.home(), '.cli-agent')

# --- TypedDict Definitions for Structured Data ---

class LLMModelConfig(TypedDict):
    """Configuration for an individual LLM model."""
    selected: NotRequired[bool]
    host: NotRequired[str]
    branches: NotRequired[List[str]]
    eval: NotRequired[bool]
    guard: NotRequired[bool]
    router: NotRequired[bool]
    planner: NotRequired[bool]
    provider: NotRequired[str]
    available: NotRequired[bool]
    status: NotRequired[str]

class LLMRoleConfig(TypedDict, total=False):
    """LLM role configuration structure."""
    planner: Optional[str]
    router: Optional[str]
    fallbacks: List[str]

class LLMConfig(TypedDict, total=False):
    """Complete LLM configuration structure."""
    roles: LLMRoleConfig

class UserConfig(TypedDict, total=False):
    """User configuration structure."""
    tts_engine_priority: List[str]
    tts_disabled_engines: List[str]
    tts_context_settings: Dict[str, Dict[str, Any]]
    tts_defaults: Dict[str, Any]

class TTSContextSettings(TypedDict):
    """TTS settings for a specific context."""
    speed: float
    volume: float
    voice: str

class ModelVariant(TypedDict):
    """Model variant information."""
    name: str
    size: NotRequired[str]
    parameters: NotRequired[str]
    quantization: NotRequired[str]
    available: NotRequired[bool]

class ModelInfo(TypedDict):
    """Model information in discovery cache."""
    name: str
    provider: str
    variants: Dict[str, ModelVariant]
    description: NotRequired[str]
    tags: NotRequired[List[str]]

class ModelDiscoveryCache(TypedDict):
    """Structure for model discovery cache."""
    timestamp: float
    model_data: Dict[str, ModelInfo]
    version: int

class DebugLogProtocol(Protocol):
    """Protocol for debug logging function."""
    def __call__(self, message: str, color: Optional[str] = None, force_print: bool = False, **kwargs: Any) -> None: ...

class PrintTokenProtocol(Protocol):
    """Protocol for print token function."""
    def __call__(self, token: str) -> None: ...

# --- Main Globals Class ---
class Globals:
    """
    A singleton class to hold and manage the global state of the application.
    This includes paths, runtime flags, and selected model configurations.
    """
    def __init__(self):
        # --- Path Configurations (XDG Base Directory Compliant) ---
        self.CLIAGENT_ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # XDG Base Directory paths
        self.CLIAGENT_CONFIG_DIR: str = _get_xdg_config_dir()
        self.CLIAGENT_DATA_DIR: str = _get_xdg_data_dir()
        self.CLIAGENT_CACHE_DIR: str = _get_xdg_cache_dir()
        self.CLIAGENT_STATE_DIR: str = _get_xdg_state_dir()

        # Legacy path for migration
        self.CLIAGENT_PERSISTENT_STORAGE_PATH: str = _get_persistent_storage_path()

        # Organized directory structure
        self.CLIAGENT_TEMP_STORAGE_PATH: str = os.path.join(self.CLIAGENT_CACHE_DIR, "temp")
        self.AGENTS_SANDBOX_DIR: str = os.path.join(self.CLIAGENT_DATA_DIR, "sandbox")

        # --- File Path Configurations ---
        self.CLIAGENT_ENV_FILE_PATH: str = os.path.join(self.CLIAGENT_ROOT_PATH, ".env")
        self.USER_CONFIG_PATH: str = os.path.join(self.CLIAGENT_CONFIG_DIR, 'user_config.json')
        self.LLM_CONFIG_PATH: str = os.path.join(self.CLIAGENT_CONFIG_DIR, 'llm_config.json')

        # --- Model Limit Paths with Daily Rotation (moved to STATE_DIR) ---
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        model_limits_dir = os.path.join(self.CLIAGENT_STATE_DIR, 'model_limits')
        os.makedirs(model_limits_dir, exist_ok=True)
        self.MODEL_TOKEN_LIMITS_PATH: str = os.path.join(model_limits_dir, f'{today}_model_token_limits.json')
        self.MODEL_RATE_LIMITS_PATH: str = os.path.join(model_limits_dir, f'{today}_model_rate_limits.json')

        self.UNCONFIRMED_FINETUNING_PATH: str = os.path.join(self.CLIAGENT_DATA_DIR, "finetuning_data", "unconfirmed")
        self.CONFIRMED_FINETUNING_PATH: str = os.path.join(self.CLIAGENT_DATA_DIR, "finetuning_data", "confirmed")

        # --- LLM and Agent Configuration ---
        self.LLM: Optional[str] = None
        self.llm_roles: LLMRoleConfig = {
            'planner': None,
            'router': None,
            'fallbacks': [],
        }
        self.DEFAULT_OLLAMA_HOSTS: List[str] = ["http://localhost:11434"]

        # --- Forcing Flags (runtime modifiers) ---
        import time
        import_start = time.time()
        from core.ai_strengths import AIStrengths # Local import
        import_time = (time.time() - import_start) * 1000
        if import_time > 1:  # Only log if > 1ms
            logging.info(f"🚀 GLOBALS: AIStrengths import took {import_time:.1f}ms")

        self.FORCE_LOCAL: bool = False
        self.LLM_STRENGTHS: List[AIStrengths] = []

        # --- Debug and UI Flags ---
        self.DEBUG_CHATS: bool = False
        self.DEBUG_MODE: bool = False
        self.VERBOSE_DEBUG: bool = False  # Set to True to enable verbose API debugging
        self.GUI_MODE: bool = False  # Set to True when running in GUI mode

        # --- Agent Control Flags ---
        self.CONTINUE_CONVERSATION: bool = False  # Set by CLI args for continuous conversation
        self.AGENT_RESET_REQUESTED: bool = False  # Set when agent reset is requested
        self.autonomous_iteration_count: int = 0  # Counter for autonomous iterations
        self.total_autonomous_count: int = 0  # Total autonomous iterations counter
        self.USER_INTERRUPT_REQUESTED: bool = False  # Set when user requests interrupt

        # --- Mode Flags ---
        self.auto_execute: bool = False  # Auto-execute mode flag
        self.voice_mode: bool = False  # Voice mode flag
        self.local_mode: bool = False  # Local mode flag
        self.online_mode: bool = False  # Online mode flag

        # --- Utility and Tool Management ---
        self.SELECTED_UTILS: List[str] = []
        self.AGENT_IS_COMPACTING: bool = False

        # --- Output Truncation Settings ---
        self.OUTPUT_TRUNCATE_HEAD_SIZE: int = 2000
        self.OUTPUT_TRUNCATE_TAIL_SIZE: int = 2000

        # --- API Timeout Settings ---
        self.GOOGLE_API_TIMEOUT_SECONDS: int = 300  # 5 minutes to respect rate limit delays

        # --- Cross-module Communication & State ---
        self.web_server: Optional[Any] = None
        self.print_token: PrintTokenProtocol = lambda token: print(token, end="", flush=True)
        self.debug_log: DebugLogProtocol = self._default_debug_log
        self._user_config: UserConfig = {}
        self._llm_config: LLMConfig = {}

        # --- Tool and Agent State ---
        self.last_executed_tools_info: Optional[Any] = None  # Stores information about last executed tools
        self.current_images: List[str] = []  # Current images for vision models

        # --- GUI Communication ---
        import queue
        self.visual_input_queue: queue.Queue = queue.Queue()  # Queue for visual input from web interface

        # --- TTS Configuration ---
        # Default: Local models first, then remote APIs (prioritize privacy and offline capability)
        self.TTS_ENGINE_PRIORITY: List[str] = ["piper", "coqui", "huggingface", "system", "openai", "elevenlabs"]
        self.TTS_DISABLED_ENGINES: List[str] = []
        self.TTS_DEFAULT_VOICE: str = "default"
        self.TTS_DEFAULT_SPEED: float = 1.0
        self.TTS_DEFAULT_VOLUME: float = 0.8  # Volume level (0.0 to 1.0)
        self.TTS_DEFAULT_LANGUAGE: str = "auto"
        self.TTS_DEFAULT_GENDER: str = "female"
        self.TTS_DEFAULT_QUALITY: str = "medium"
        self.TTS_TIMEOUT_SECONDS: int = 30
        self.TTS_MAX_RETRIES: int = 2
        self.TTS_FALLBACK_ENABLED: bool = True
        self.TTS_CACHE_ENABLED: bool = True

        # --- TTS Voice Mappings ---
        self.TTS_VOICE_MAPPINGS: Dict[str, Dict[str, str]] = {
            "openai": {
                "default": "nova",
                "female": "nova",
                "male": "onyx",
                "calm": "alloy",
                "energetic": "echo",
                "storyteller": "fable",
                "smooth": "shimmer"
            },
            "elevenlabs": {
                "default": "rachel",
                "female": "rachel",
                "male": "drew",
                "professional": "clyde",
                "friendly": "dave"
            },
            "piper": {
                "default": "en_GB-alba-medium",
                "german_male": "de_DE-thorsten-high",
                "german_female": "de_DE-kerstin-medium",
                "english_male": "en_GB-alan-medium",
                "english_female": "en_GB-alba-medium"
            }
        }

        # --- TTS Context-Specific Settings ---
        self.TTS_CONTEXT_SETTINGS: Dict[str, Dict[str, Any]] = {
            "system_message": {
                "speed": 0.9,
                "volume": 0.6,
                "voice": "default"  # Use default instead of "calm" which doesn't exist
            },
            "permission_prompt": {
                "speed": 0.8,
                "volume": 0.7,
                "voice": "default"  # Use default instead of "professional" which doesn't exist
            },
            "error_message": {
                "speed": 0.8,
                "volume": 0.5,
                "voice": "default"  # Use default instead of "calm" which doesn't exist
            },
            "user_response": {
                "speed": 1.0,
                "volume": 0.7,
                "voice": "default"
            },
            "tool_execution": {
                "speed": 1.0,
                "volume": 0.7,
                "voice": "default"
            },
            "test": {
                "speed": 1.0,
                "volume": 1.0,
                "voice": "default"
            }
        }


        # --- Model Discovery Cache ---
        self._model_discovery_cache: Optional[Dict[str, ModelInfo]] = None
        self._model_discovery_task: Optional[Any] = None  # Background asyncio task
        self._model_discovery_timestamp: float = 0
        self.MODEL_CACHE_PATH: str = os.path.join(self.CLIAGENT_CACHE_DIR, 'model_cache.json')

        # --- Initial Setup on Instantation ---
        import time
        startup_start = time.time()
        logging.info("🚀 GLOBALS INIT: Starting initialization...")

        step_start = time.time()
        self._initialize_directories()
        logging.info(f"🚀 GLOBALS INIT: _initialize_directories took {(time.time() - step_start)*1000:.1f}ms")

        step_start = time.time()
        self._migrate_legacy_files()  # Migrate files from old structure
        logging.info(f"🚀 GLOBALS INIT: _migrate_legacy_files took {(time.time() - step_start)*1000:.1f}ms")

        step_start = time.time()
        self.load_user_config()
        logging.info(f"🚀 GLOBALS INIT: load_user_config took {(time.time() - step_start)*1000:.1f}ms")

        step_start = time.time()
        self.load_llm_config() # Load LLM config at startup
        logging.info(f"🚀 GLOBALS INIT: load_llm_config took {(time.time() - step_start)*1000:.1f}ms")

        step_start = time.time()
        self.load_llm_roles() # Load LLM role configuration at startup
        logging.info(f"🚀 GLOBALS INIT: load_llm_roles took {(time.time() - step_start)*1000:.1f}ms")

        step_start = time.time()
        self.load_tts_config_from_user_settings() # Load TTS config at startup
        logging.info(f"🚀 GLOBALS INIT: load_tts_config_from_user_settings took {(time.time() - step_start)*1000:.1f}ms")

        step_start = time.time()
        self._configure_ollama_hosts()
        logging.info(f"🚀 GLOBALS INIT: _configure_ollama_hosts took {(time.time() - step_start)*1000:.1f}ms")

        total_time = time.time() - startup_start
        logging.info(f"🚀 GLOBALS INIT: Total initialization took {total_time*1000:.1f}ms")

        # --- Cleanup callbacks ---
        self._cleanup_callbacks: List[Callable] = []
        self._persistent_services: Set[Callable] = set()  # Services that should persist
        atexit.register(self._run_cleanup)

    def _initialize_directories(self):
        """Create necessary XDG-compliant directories and clean up temporary ones."""
        # Create XDG-compliant directory structure
        os.makedirs(self.CLIAGENT_CONFIG_DIR, exist_ok=True)
        os.makedirs(self.CLIAGENT_DATA_DIR, exist_ok=True)
        os.makedirs(self.CLIAGENT_CACHE_DIR, exist_ok=True)
        os.makedirs(self.CLIAGENT_STATE_DIR, exist_ok=True)
        os.makedirs(self.AGENTS_SANDBOX_DIR, exist_ok=True)

        # Clean up and recreate temp directory
        if os.path.exists(self.CLIAGENT_TEMP_STORAGE_PATH):
            shutil.rmtree(self.CLIAGENT_TEMP_STORAGE_PATH)
        os.makedirs(self.CLIAGENT_TEMP_STORAGE_PATH, exist_ok=True)

    def _migrate_legacy_files(self):
        """Migrate files from old ~/.cli-agent structure to XDG-compliant directories."""
        legacy_path = self.CLIAGENT_PERSISTENT_STORAGE_PATH

        # Skip migration if legacy directory doesn't exist
        if not os.path.exists(legacy_path):
            return

        # Skip migration if this IS the legacy path (Windows compatibility)
        if legacy_path == self.CLIAGENT_CONFIG_DIR:
            return

        migration_log = []
        migrated_count = 0

        try:
            # Migration mapping: (old_path, new_path, description)
            migrations = [
                # Configuration files
                (os.path.join(legacy_path, 'user_config.json'),
                 self.USER_CONFIG_PATH, 'User configuration'),
                (os.path.join(legacy_path, 'llm_config.json'),
                 self.LLM_CONFIG_PATH, 'LLM configuration'),

                # Data files
                (os.path.join(legacy_path, 'finetuning_data'),
                 os.path.join(self.CLIAGENT_DATA_DIR, 'finetuning_data'), 'Finetuning data'),
                (os.path.join(legacy_path, 'sandbox'),
                 os.path.join(self.CLIAGENT_DATA_DIR, 'sandbox'), 'Sandbox directory'),

                # State files (ephemeral application state)
                (os.path.join(legacy_path, 'model_limits'),
                 os.path.join(self.CLIAGENT_STATE_DIR, 'model_limits'), 'Model limits state'),

                # Cache files
                (os.path.join(legacy_path, 'model_cache.json'),
                 self.MODEL_CACHE_PATH, 'Model cache'),
                (os.path.join(legacy_path, 'cache'),
                 os.path.join(self.CLIAGENT_CACHE_DIR, 'legacy_cache'), 'Legacy cache'),
                (os.path.join(legacy_path, 'temp'),
                 os.path.join(self.CLIAGENT_CACHE_DIR, 'legacy_temp'), 'Legacy temp files'),

                # Vector databases and search cache
                (os.path.join(legacy_path, 'searchweb_vectordb'),
                 os.path.join(self.CLIAGENT_DATA_DIR, 'vector_db', 'searchweb_vectordb'), 'Search vector database'),
            ]

            for old_path, new_path, description in migrations:
                if os.path.exists(old_path):
                    try:
                        # Ensure target directory exists
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)

                        # Move the file or directory
                        if os.path.isdir(old_path):
                            if os.path.exists(new_path):
                                # If target exists, merge contents
                                self._merge_directories(old_path, new_path)
                                shutil.rmtree(old_path)
                            else:
                                shutil.move(old_path, new_path)
                        else:
                            if not os.path.exists(new_path):
                                shutil.move(old_path, new_path)

                        migration_log.append(f"✓ Migrated {description}: {old_path} → {new_path}")
                        migrated_count += 1

                    except Exception as e:
                        migration_log.append(f"✗ Failed to migrate {description}: {e}")

            # Move permissions file from home directory to config directory
            home_permissions_file = os.path.join(Path.home(), 'permissions.json')
            config_permissions_file = os.path.join(self.CLIAGENT_CONFIG_DIR, 'permissions.json')

            if os.path.exists(home_permissions_file) and not os.path.exists(config_permissions_file):
                try:
                    shutil.move(home_permissions_file, config_permissions_file)
                    migration_log.append(f"✓ Migrated permissions file: {home_permissions_file} → {config_permissions_file}")
                    migrated_count += 1
                except Exception as e:
                    migration_log.append(f"✗ Failed to migrate permissions file: {e}")

            # Log migration results
            if migrated_count > 0:
                logging.info(f"🚚 Migrated {migrated_count} items to XDG-compliant structure")
                for log_entry in migration_log:
                    logging.debug(log_entry)

                # Create migration marker to avoid re-migration
                marker_file = os.path.join(self.CLIAGENT_CONFIG_DIR, '.xdg_migration_completed')
                with open(marker_file, 'w') as f:
                    f.write(f"Migration completed at {datetime.datetime.now().isoformat()}\n")
                    f.write(f"Migrated {migrated_count} items\n")

                # Only remove legacy directory if it's now empty (safe cleanup)
                try:
                    if os.path.isdir(legacy_path) and not os.listdir(legacy_path):
                        os.rmdir(legacy_path)
                        logging.info(f"🗂️ Removed empty legacy directory: {legacy_path}")
                except OSError:
                    pass  # Directory not empty or permission issue

            # Secondary migration: Move model_limits from data to state directory (for existing XDG installations)
            data_model_limits = os.path.join(self.CLIAGENT_DATA_DIR, 'model_limits')
            state_model_limits = os.path.join(self.CLIAGENT_STATE_DIR, 'model_limits')

            if os.path.exists(data_model_limits) and not os.path.exists(state_model_limits):
                try:
                    shutil.move(data_model_limits, state_model_limits)
                    logging.info(f"🚚 Moved model_limits to state directory: {data_model_limits} → {state_model_limits}")
                    migrated_count += 1
                except Exception as e:
                    logging.warning(f"Failed to move model_limits to state directory: {e}")

        except Exception as e:
            logging.warning(f"Migration process encountered an error: {e}")

    def _merge_directories(self, src_dir: str, dst_dir: str):
        """Merge contents of source directory into destination directory."""
        for item in os.listdir(src_dir):
            src_item = os.path.join(src_dir, item)
            dst_item = os.path.join(dst_dir, item)

            if os.path.isdir(src_item):
                if os.path.exists(dst_item):
                    self._merge_directories(src_item, dst_item)
                    shutil.rmtree(src_item)
                else:
                    shutil.move(src_item, dst_item)
            else:
                if not os.path.exists(dst_item):
                    shutil.move(src_item, dst_item)
                else:
                    # File exists in destination, backup source with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{item}.backup_{timestamp}"
                    backup_path = os.path.join(dst_dir, backup_name)
                    shutil.move(src_item, backup_path)

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

    def get_llm_config(self, force_reread: bool = False) -> LLMConfig:
        """Get the loaded LLM configuration."""
        if force_reread:
            self.load_llm_config()
        return self._llm_config

    @property
    def llm_config(self) -> LLMConfig:
        """Property access to LLM configuration."""
        return self._llm_config

    def save_llm_config(self, config: LLMConfig) -> None:
        """Save LLM configuration to file."""
        self._llm_config = config
        try:
            os.makedirs(os.path.dirname(self.LLM_CONFIG_PATH), exist_ok=True)
            with open(self.LLM_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save LLM config: {e}")


    def cleanup_temp_py_files(self):
        """Remove temporary Python files from previous runs using enhanced file manager."""
        try:
            # Try to use the new file manager for comprehensive cleanup
            from core.file_manager import get_file_manager

            file_manager = get_file_manager()

            # Clean up tracked Python temp files
            python_stats = file_manager.cleanup_by_category('python_temp', force_all=True)

            # Clean up legacy untracked files
            legacy_stats = file_manager.cleanup_legacy_files()

            total_removed = python_stats['files_removed'] + legacy_stats['files_removed']
            total_freed = python_stats['bytes_freed'] + legacy_stats['bytes_freed']

            if total_removed > 0:
                logging.info(f"🧹 Cleaned up {total_removed} temporary files ({total_freed / 1024:.1f} KB freed)")

        except ImportError:
            # Fallback to original implementation if file manager is not available
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
        except Exception as e:
            logging.debug(f"File manager cleanup failed, using fallback: {e}")
            # Fallback to original implementation
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

    def start_background_model_discovery(self, force_refresh: bool = False) -> None:
        """Start background model discovery task."""
        import asyncio
        import time

        # First, try to load from persistent cache for immediate availability
        if not force_refresh:
            persistent_cache = self.load_persistent_model_cache()
            if persistent_cache:
                self._model_discovery_cache = persistent_cache
                self._model_discovery_timestamp = time.time()

                # Check if cache is fresh enough to skip background refresh
                cache_age_hours = self.get_persistent_cache_age_hours()
                if cache_age_hours < 6:  # Cache less than 6 hours old, skip background refresh
                    logging.debug(f"Using fresh persistent cache ({cache_age_hours:.1f}h old), skipping background refresh")
                    return
                else:
                    logging.debug(f"Loaded persistent cache ({cache_age_hours:.1f}h old), will refresh in background")

        async def discover_models():
            """Background task to discover and cache model information."""
            try:
                from core.providers.cls_ollama_interface import OllamaClient

                # Get ollama hosts
                ollama_hosts = []
                for host_url in self.DEFAULT_OLLAMA_HOSTS:
                    if "://" in host_url:
                        host = host_url.split("://")[1].split("/")[0]
                    else:
                        host = host_url
                    if ":" in host:
                        host = host.split(":")[0]
                    ollama_hosts.append(host)

                # Discover models in background
                logging.debug("Starting background model discovery...")
                start_time = time.time()

                model_status = OllamaClient.get_comprehensive_downloadable_models(ollama_hosts)

                discovery_time = time.time() - start_time
                variant_count = sum(len(info.get('variants', {})) for info in model_status.values())

                # Cache the results
                self._model_discovery_cache = model_status
                self._model_discovery_timestamp = time.time()

                # Save to persistent cache
                self.save_persistent_model_cache(model_status)

                logging.debug(f"Background model discovery completed in {discovery_time:.1f}s: {len(model_status)} model bases, {variant_count} variants")

            except Exception as e:
                logging.debug(f"Background model discovery failed: {e}")
                # Don't overwrite existing cache on failure
                if self._model_discovery_cache is None:
                    self._model_discovery_cache = {}
                    self._model_discovery_timestamp = time.time()

        # Create and start the background task
        try:
            loop = asyncio.get_event_loop()
            self._model_discovery_task = loop.create_task(discover_models())
        except RuntimeError:
            # If no event loop is running, we'll do lazy loading instead
            logging.debug("No event loop available for background model discovery")

    def get_cached_model_discovery(self) -> Optional[Dict[str, ModelInfo]]:
        """Get cached model discovery results."""
        return self._model_discovery_cache

    def is_model_discovery_ready(self) -> bool:
        """Check if model discovery cache is ready."""
        return self._model_discovery_cache is not None

    def wait_for_model_discovery(self, timeout: float = 10.0) -> bool:
        """Wait for model discovery to complete, with timeout."""
        if self._model_discovery_cache is not None:
            return True

        if self._model_discovery_task is None:
            return False

        import asyncio
        import time

        start_time = time.time()
        try:
            loop = asyncio.get_event_loop()
            # Use run_until_complete to properly await the task
            loop.run_until_complete(asyncio.wait_for(self._model_discovery_task, timeout=timeout))
            return self._model_discovery_cache is not None
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logging.debug(f"Model discovery timeout after {elapsed:.1f}s")
            return False
        except RuntimeError:
            # No event loop running - can't wait synchronously
            # Poll the task completion status instead
            elapsed = 0
            while elapsed < timeout:
                if self._model_discovery_cache is not None:
                    return True
                time.sleep(0.1)
                elapsed = time.time() - start_time
            logging.debug(f"Model discovery polling timeout after {elapsed:.1f}s")
            return False
        except Exception as e:
            logging.debug(f"Model discovery wait failed: {e}")
            return False

    def invalidate_model_discovery_cache(self) -> None:
        """Invalidate the model discovery cache to force refresh."""
        self._model_discovery_cache = None
        self._model_discovery_timestamp = 0
        if self._model_discovery_task:
            try:
                self._model_discovery_task.cancel()
            except:
                pass
            self._model_discovery_task = None
        logging.debug("Model discovery cache invalidated")

    def refresh_model_discovery(self) -> None:
        """Refresh model discovery cache by starting a new background task."""
        self.invalidate_model_discovery_cache()
        self.start_background_model_discovery()
        logging.debug("Model discovery refresh initiated")

    def is_model_cache_stale(self, max_age_minutes: int = 30) -> bool:
        """Check if the model cache is stale and needs refresh."""
        if self._model_discovery_cache is None:
            return True

        import time
        age_seconds = time.time() - self._model_discovery_timestamp
        age_minutes = age_seconds / 60

        return age_minutes > max_age_minutes

    def load_persistent_model_cache(self) -> Optional[Dict[str, ModelInfo]]:
        """Load model cache from persistent storage."""
        if not os.path.exists(self.MODEL_CACHE_PATH):
            return None

        try:
            with open(self.MODEL_CACHE_PATH, 'r') as f:
                cache_data = json.load(f)

            # Check cache age (default: cache valid for 24 hours)
            cache_timestamp = cache_data.get('timestamp', 0)
            import time
            age_hours = (time.time() - cache_timestamp) / 3600

            if age_hours > 24:  # Cache older than 24 hours
                logging.debug(f"Persistent cache is {age_hours:.1f} hours old, will refresh in background")
                return cache_data.get('model_data', {})  # Return old data for immediate use

            logging.debug(f"Loaded persistent model cache: {len(cache_data.get('model_data', {}))} model bases (age: {age_hours:.1f}h)")
            return cache_data.get('model_data', {})

        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Failed to load persistent model cache: {e}")
            return None

    def save_persistent_model_cache(self, model_data: Dict[str, ModelInfo]) -> None:
        """Save model cache to persistent storage."""
        try:
            import time
            cache_data = {
                'timestamp': time.time(),
                'model_data': model_data,
                'version': 1  # For future compatibility
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.MODEL_CACHE_PATH), exist_ok=True)

            # Write to temporary file first, then rename (atomic operation)
            temp_path = self.MODEL_CACHE_PATH + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)  # default=str handles datetime objects

            os.rename(temp_path, self.MODEL_CACHE_PATH)

            variant_count = sum(len(info.get('variants', {})) for info in model_data.values())
            logging.debug(f"Saved persistent model cache: {len(model_data)} model bases, {variant_count} variants")

        except Exception as e:
            logging.error(f"Failed to save persistent model cache: {e}")

    def get_persistent_cache_age_hours(self) -> float:
        """Get the age of persistent cache in hours."""
        if not os.path.exists(self.MODEL_CACHE_PATH):
            return float('inf')

        try:
            with open(self.MODEL_CACHE_PATH, 'r') as f:
                cache_data = json.load(f)
            cache_timestamp = cache_data.get('timestamp', 0)
            import time
            return (time.time() - cache_timestamp) / 3600
        except:
            return float('inf')

    # --- TTS Configuration Methods ---

    def get_tts_engine_priority(self) -> List[str]:
        """Get TTS engine priority list, excluding disabled engines."""
        return [engine for engine in self.TTS_ENGINE_PRIORITY if engine not in self.TTS_DISABLED_ENGINES]

    def set_tts_engine_priority(self, engines: List[str]) -> None:
        """Set TTS engine priority order and save to config."""
        self.TTS_ENGINE_PRIORITY = engines
        self.set_config_value('tts_engine_priority', engines)

    def disable_tts_engine(self, engine: str) -> None:
        """Disable a specific TTS engine."""
        if engine not in self.TTS_DISABLED_ENGINES:
            self.TTS_DISABLED_ENGINES.append(engine)
            self.set_config_value('tts_disabled_engines', self.TTS_DISABLED_ENGINES)

    def enable_tts_engine(self, engine: str) -> None:
        """Enable a previously disabled TTS engine."""
        if engine in self.TTS_DISABLED_ENGINES:
            self.TTS_DISABLED_ENGINES.remove(engine)
            self.set_config_value('tts_disabled_engines', self.TTS_DISABLED_ENGINES)

    def get_tts_voice_for_engine(self, engine: str, voice_type: str = "default") -> str:
        """Get the appropriate voice ID for a specific engine and voice type."""
        engine_voices = self.TTS_VOICE_MAPPINGS.get(engine, {})
        return engine_voices.get(voice_type, voice_type)  # Fallback to voice_type if not mapped

    def get_tts_context_settings(self, context: str) -> TTSContextSettings:
        """Get TTS settings for a specific context."""
        default_settings: TTSContextSettings = {
            "speed": self.TTS_DEFAULT_SPEED,
            "volume": self.TTS_DEFAULT_VOLUME,
            "voice": self.TTS_DEFAULT_VOICE
        }
        stored_settings = self.TTS_CONTEXT_SETTINGS.get(context, default_settings)
        # Ensure all required fields are present
        return {
            "speed": stored_settings.get("speed", self.TTS_DEFAULT_SPEED),
            "volume": stored_settings.get("volume", self.TTS_DEFAULT_VOLUME),
            "voice": stored_settings.get("voice", self.TTS_DEFAULT_VOICE)
        }

    def set_tts_context_settings(self, context: str, settings: TTSContextSettings) -> None:
        """Set TTS settings for a specific context."""
        self.TTS_CONTEXT_SETTINGS[context] = settings
        # Save to user config
        tts_contexts = self.get_config_value('tts_context_settings', {})
        tts_contexts[context] = settings
        self.set_config_value('tts_context_settings', tts_contexts)


    def load_tts_config_from_user_settings(self) -> None:
        """Load TTS configuration from saved user settings."""
        # Load engine priority
        saved_priority = self.get_config_value('tts_engine_priority')
        if saved_priority:
            self.TTS_ENGINE_PRIORITY = saved_priority

        # Load disabled engines
        saved_disabled = self.get_config_value('tts_disabled_engines')
        if saved_disabled:
            self.TTS_DISABLED_ENGINES = saved_disabled

        # Load context settings
        saved_contexts = self.get_config_value('tts_context_settings')
        if saved_contexts:
            self.TTS_CONTEXT_SETTINGS.update(saved_contexts)

        # Load default settings
        saved_defaults = self.get_config_value('tts_defaults', {})
        self.TTS_DEFAULT_VOICE = saved_defaults.get('voice', self.TTS_DEFAULT_VOICE)
        self.TTS_DEFAULT_SPEED = saved_defaults.get('speed', self.TTS_DEFAULT_SPEED)
        self.TTS_DEFAULT_VOLUME = saved_defaults.get('volume', self.TTS_DEFAULT_VOLUME)
        self.TTS_DEFAULT_LANGUAGE = saved_defaults.get('language', self.TTS_DEFAULT_LANGUAGE)
        self.TTS_DEFAULT_GENDER = saved_defaults.get('gender', self.TTS_DEFAULT_GENDER)
        self.TTS_DEFAULT_QUALITY = saved_defaults.get('quality', self.TTS_DEFAULT_QUALITY)

    def save_tts_defaults(self, voice: str = None, speed: float = None, volume: float = None,
                         language: str = None, gender: str = None, quality: str = None) -> None:
        """Save TTS default settings."""
        defaults = self.get_config_value('tts_defaults', {})

        if voice is not None:
            self.TTS_DEFAULT_VOICE = voice
            defaults['voice'] = voice
        if speed is not None:
            self.TTS_DEFAULT_SPEED = speed
            defaults['speed'] = speed
        if volume is not None:
            self.TTS_DEFAULT_VOLUME = volume
            defaults['volume'] = volume
        if language is not None:
            self.TTS_DEFAULT_LANGUAGE = language
            defaults['language'] = language
        if gender is not None:
            self.TTS_DEFAULT_GENDER = gender
            defaults['gender'] = gender
        if quality is not None:
            self.TTS_DEFAULT_QUALITY = quality
            defaults['quality'] = quality

        self.set_config_value('tts_defaults', defaults)

    # --- LLM Role Management ---
    @property
    def planner_llm(self) -> Optional[str]:
        """Backward compatibility property for planner LLM."""
        return self.llm_roles.get('planner')

    @property
    def router_llm(self) -> Optional[str]:
        """Backward compatibility property for router LLM."""
        return self.llm_roles.get('router')

    @property
    def FALLBACK_MODELS(self) -> List[str]:
        """Backward compatibility property for fallback models."""
        return self.llm_roles.get('fallbacks', [])

    @FALLBACK_MODELS.setter
    def FALLBACK_MODELS(self, value: List[str]) -> None:
        """Setter for fallback models."""
        self.llm_roles['fallbacks'] = value
        self.save_llm_roles()

    @property
    def SELECTED_LLMS(self) -> List[str]:
        """Backward compatibility property for selected LLMs. Now points to fallbacks array."""
        return self.llm_roles.get('fallbacks', [])

    @SELECTED_LLMS.setter
    def SELECTED_LLMS(self, value: List[str]) -> None:
        """Setter for selected LLMs. Now updates fallbacks array."""
        self.llm_roles['fallbacks'] = value
        self.save_llm_roles()

    def update_llm_role(self, role: str, model_key: Optional[str] = None, model_keys: Optional[List[str]] = None) -> None:
        """
        Unified method to update LLM roles with automatic persistence.

        Args:
            role: The role to update ('planner', 'router', 'fallbacks')
            model_key: Single model key for planner/router roles
            model_keys: List of model keys for fallbacks role
        """
        if role in ['planner', 'router']:
            if model_keys is not None:
                raise ValueError(f"Role '{role}' expects model_key, not model_keys")
            self.llm_roles[role] = model_key
        elif role == 'fallbacks':
            if model_key is not None:
                raise ValueError(f"Role '{role}' expects model_keys, not model_key")
            self.llm_roles[role] = model_keys or []
        elif role == 'selected':
            # Redirect selected to fallbacks for backward compatibility
            if model_key is not None:
                raise ValueError(f"Role '{role}' expects model_keys, not model_key")
            self.llm_roles['fallbacks'] = model_keys or []
        else:
            raise ValueError(f"Unknown role: {role}. Valid roles: planner, router, fallbacks")

        self.save_llm_roles()
        logging.info(f"🔧 LLM role '{role}' updated to: {model_key or model_keys}")

    def save_llm_roles(self) -> None:
        """Save LLM role configuration to disk."""
        config = self.get_llm_config()
        config['roles'] = self.llm_roles
        self.save_llm_config(config)

    def load_llm_roles(self) -> None:
        """Load LLM role configuration from disk, migrating from old format if needed."""
        import time
        load_start = time.time()

        step_start = time.time()
        config = self.get_llm_config()
        logging.info(f"🚀 ROLES: get_llm_config took {(time.time() - step_start)*1000:.1f}ms")

        # Check if we have the new unified format with valid data
        has_valid_roles = (
            'roles' in config and
            isinstance(config['roles'], dict) and
            # Check if roles has been properly populated (not just empty defaults)
            (config['roles'].get('planner') or config['roles'].get('router') or
             config['roles'].get('fallbacks'))
        )

        if has_valid_roles:
            # Use new format
            step_start = time.time()
            for key, default_value in [('planner', None), ('router', None), ('fallbacks', [])]:
                self.llm_roles[key] = config['roles'].get(key, default_value)
            logging.info(f"🚀 ROLES: Loading existing roles took {(time.time() - step_start)*1000:.1f}ms")
        else:
            # Migrate from old format
            step_start = time.time()
            logging.info("🔄 Migrating LLM configuration from old format to unified roles")

            # Migrate planner and router
            self.llm_roles['planner'] = config.get('__planner_llm__')
            self.llm_roles['router'] = config.get('__router_llm__')

            # Migrate selected models to fallbacks array (eliminating separate selected array)
            selected_models = []
            for model_key, model_config in config.items():
                if isinstance(model_config, dict) and model_config.get('selected', False):
                    selected_models.append(model_key)

            # Sort by order if available
            selected_models.sort(key=lambda m: config.get(m, {}).get('order', 999))

            # Store selected models as fallbacks (unified array)
            self.llm_roles['fallbacks'] = selected_models
            
            # If no models were migrated, apply default configuration for new setups
            if not selected_models and not config.get('__planner_llm__') and not config.get('__router_llm__'):
                logging.info("🔧 No existing configuration found, applying default LLM configuration")
                self.llm_roles['planner'] = 'gemini-2.5-pro'
                self.llm_roles['router'] = 'gemini-2.5-flash-lite-preview-06-17'
                self.llm_roles['fallbacks'] = [
                    'gemini-2.5-flash-preview-05-20',
                    'gemini-2.5-flash', 
                    'gemini-2.5-flash-lite-preview-06-17'
                ]

            # Save the migrated configuration and clean up old format
            self.save_llm_roles()

            # Clean up old format keys from config file
            cleanup_start = time.time()
            self._cleanup_old_format_config()
            logging.info(f"🚀 ROLES: Cleanup took {(time.time() - cleanup_start)*1000:.1f}ms")

            migration_time = time.time() - step_start
            logging.info(f"✅ Migration completed in {migration_time*1000:.1f}ms: {len(selected_models)} selected models, planner: {self.llm_roles['planner']}, router: {self.llm_roles['router']}")

        total_roles_time = time.time() - load_start
        logging.info(f"🚀 ROLES: Total load_llm_roles took {total_roles_time*1000:.1f}ms")

    def _cleanup_old_format_config(self) -> None:
        """Remove old format keys from config file after successful migration."""
        try:
            config = self.get_llm_config()
            cleaned = False

            # Remove old special keys
            if '__planner_llm__' in config:
                del config['__planner_llm__']
                cleaned = True
            if '__router_llm__' in config:
                del config['__router_llm__']
                cleaned = True

            # Remove 'selected' array from roles (now unified with fallbacks)
            if 'roles' in config and isinstance(config['roles'], dict):
                if 'selected' in config['roles']:
                    del config['roles']['selected']
                    cleaned = True

            # Remove 'selected' and 'order' from model entries (they're now in roles)
            for model_key, model_config in list(config.items()):
                if isinstance(model_config, dict) and model_key != 'roles':
                    if 'selected' in model_config:
                        del model_config['selected']
                        cleaned = True
                    if 'order' in model_config:
                        del model_config['order']
                        cleaned = True
                    # Remove empty model configs
                    if not model_config or model_config == {'model_key': model_key}:
                        del config[model_key]
                        cleaned = True

            if cleaned:
                self.save_llm_config(config)
                logging.info("🧹 Cleaned up old format configuration keys")

        except Exception as e:
            logging.warning(f"Failed to cleanup old format config: {e}")

    # --- Cleanup registration and execution ---
    def register_cleanup(self, callback: Callable) -> None:
        """
        Register a function to be called on application exit.
        Tools with persistent background processes should use this to ensure
        graceful shutdown.
        """
        try:
            if callback not in self._cleanup_callbacks:
                self._cleanup_callbacks.append(callback)
                cb_name = getattr(callback, '__qualname__', repr(callback))
                logging.info(f"Registered cleanup callback: {cb_name}")
        except Exception as e:
            logging.debug(f"Failed to register cleanup callback: {e}")

    def register_persistent_service(self, callback: Callable) -> None:
        """
        Register a service that should persist across normal cleanup cycles.
        Persistent services are only cleaned up on explicit shutdown.
        Used for services like visual response servers that should remain active.
        """
        try:
            self._persistent_services.add(callback)
            cb_name = getattr(callback, '__qualname__', repr(callback))
            logging.debug(f"Registered persistent service: {cb_name}")
        except Exception as e:
            logging.debug(f"Failed to register persistent service: {e}")

    def unregister_persistent_service(self, callback: Callable) -> None:
        """
        Remove a service from the persistent services set, allowing it to be cleaned up normally.
        """
        try:
            self._persistent_services.discard(callback)
            cb_name = getattr(callback, '__qualname__', repr(callback))
            logging.debug(f"Unregistered persistent service: {cb_name}")
        except Exception as e:
            logging.debug(f"Failed to unregister persistent service: {e}")

    def _run_cleanup(self, force_all: bool = False) -> None:
        """
        Run registered cleanup callbacks (invoked automatically at exit).

        Args:
            force_all: If True, cleanup all services including persistent ones.
                      If False, skip persistent services during normal cleanup.
        """
        if not self._cleanup_callbacks:
            return

        logging.info("Running application cleanup...")
        cleaned_count = 0
        skipped_count = 0

        # Run in reverse order of registration to respect dependency teardown
        for callback in reversed(self._cleanup_callbacks):
            try:
                # Skip persistent services unless forced
                if not force_all and callback in self._persistent_services:
                    cb_name = getattr(callback, '__qualname__', repr(callback))
                    logging.debug(f"Skipping persistent service cleanup: {cb_name}")
                    skipped_count += 1
                    continue

                # Execute cleanup
                callback()
                cleaned_count += 1

            except Exception as e:
                cb_name = getattr(callback, '__qualname__', repr(callback))
                logging.error(f"Exception during cleanup of {cb_name}: {e}")

        if skipped_count > 0:
            logging.info(f"Cleanup complete: {cleaned_count} services cleaned, {skipped_count} persistent services preserved")
        else:
            logging.info(f"Cleanup complete: {cleaned_count} services cleaned")

    def force_cleanup_all(self) -> None:
        """Force cleanup of all services, including persistent ones."""
        logging.warning("Forcing cleanup of all services including persistent ones...")
        self._run_cleanup(force_all=True)

# --- Type Validation Helpers ---

def validate_llm_config(config: Dict[str, Any]) -> LLMConfig:
    """Validate and convert a dictionary to LLMConfig type."""
    # Ensure the config is a dict-like object
    if not isinstance(config, dict):
        raise TypeError(f"LLM config must be a dictionary, got {type(config)}")

    # Create a validated config with proper typing
    validated_config: LLMConfig = {}

    # Handle roles configuration
    if 'roles' in config:
        roles = config['roles']
        if not isinstance(roles, dict):
            raise TypeError("roles must be a dictionary")

        validated_roles: LLMRoleConfig = {}

        # Validate individual role fields
        if 'planner' in roles:
            planner = roles['planner']
            if planner is not None and not isinstance(planner, str):
                raise TypeError("planner must be a string or None")
            validated_roles['planner'] = planner

        if 'router' in roles:
            router = roles['router']
            if router is not None and not isinstance(router, str):
                raise TypeError("router must be a string or None")
            validated_roles['router'] = router

        if 'fallbacks' in roles:
            fallbacks = roles['fallbacks']
            if not isinstance(fallbacks, list) or not all(isinstance(x, str) for x in fallbacks):
                raise TypeError("fallbacks must be a list of strings")
            validated_roles['fallbacks'] = fallbacks

        # Backward compatibility: ignore 'selected' key as it's now unified with fallbacks

        validated_config['roles'] = validated_roles

    # Handle other dynamic model configurations (they remain as Any for flexibility)
    for key, value in config.items():
        if key not in ['roles']:
            validated_config[key] = value  # type: ignore

    return validated_config

def validate_user_config(config: Dict[str, Any]) -> UserConfig:
    """Validate and convert a dictionary to UserConfig type."""
    if not isinstance(config, dict):
        raise TypeError(f"User config must be a dictionary, got {type(config)}")

    # UserConfig allows arbitrary keys with total=False, so we can return as-is after validation
    validated_config: UserConfig = {}

    # Validate known keys if they exist
    if 'tts_engine_priority' in config:
        tts_priority = config['tts_engine_priority']
        if not isinstance(tts_priority, list) or not all(isinstance(x, str) for x in tts_priority):
            raise TypeError("tts_engine_priority must be a list of strings")
        validated_config['tts_engine_priority'] = tts_priority

    if 'tts_disabled_engines' in config:
        tts_disabled = config['tts_disabled_engines']
        if not isinstance(tts_disabled, list) or not all(isinstance(x, str) for x in tts_disabled):
            raise TypeError("tts_disabled_engines must be a list of strings")
        validated_config['tts_disabled_engines'] = tts_disabled

    # Copy other fields as-is (they remain flexible)
    for key, value in config.items():
        if key not in ['tts_engine_priority', 'tts_disabled_engines']:
            validated_config[key] = value  # type: ignore

    return validated_config

def validate_tts_context_settings(settings: Dict[str, Any]) -> TTSContextSettings:
    """Validate and convert a dictionary to TTSContextSettings type."""
    if not isinstance(settings, dict):
        raise TypeError(f"TTS context settings must be a dictionary, got {type(settings)}")

    # Validate required fields
    speed = settings.get('speed')
    if not isinstance(speed, (int, float)):
        raise TypeError("TTS speed must be a number")

    volume = settings.get('volume')
    if not isinstance(volume, (int, float)):
        raise TypeError("TTS volume must be a number")

    voice = settings.get('voice')
    if not isinstance(voice, str):
        raise TypeError("TTS voice must be a string")

    return {
        'speed': float(speed),
        'volume': float(volume),
        'voice': voice
    }

g = Globals()
