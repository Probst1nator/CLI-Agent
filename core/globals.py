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
from typing import List, Optional, Any, Callable, Dict, Set
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
        return os.path.join(Path.home(), 'cli-agent')

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
        self.CLIAGENT_TEMP_STORAGE_PATH: str = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, "temp")
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
        from core.ai_strengths import AIStrengths # Local import
        self.FORCE_LOCAL: bool = False
        self.LLM_STRENGTHS: List[AIStrengths] = []

        # --- Debug and UI Flags ---
        self.DEBUG_CHATS: bool = False
        self.DEBUG_MODE: bool = False
        self.SUMMARY_MODE: bool = False
        self.VERBOSE_DEBUG: bool = False  # Set to True to enable verbose API debugging
        
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
        self.print_token: Callable[[str], None] = lambda token: print(token, end="", flush=True)
        self.debug_log: Callable[..., None] = self._default_debug_log
        self._user_config: Dict[str, Any] = {}
        self._llm_config: Dict[str, Any] = {}
        
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
                "german_female": "de_DE-mls-medium", 
                "english_female": "en_GB-alba-medium",
                "english_male": "en_GB-alan-medium"
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
        self._model_discovery_cache: Optional[Dict[str, Any]] = None
        self._model_discovery_task: Optional[Any] = None  # Background asyncio task
        self._model_discovery_timestamp: float = 0
        self.MODEL_CACHE_PATH: str = os.path.join(self.CLIAGENT_PERSISTENT_STORAGE_PATH, 'model_cache.json')

        # --- Initial Setup on Instantation ---
        self._initialize_directories()
        self.load_user_config()
        self.load_llm_config() # Load LLM config at startup
        self.load_tts_config_from_user_settings() # Load TTS config at startup
        self._configure_ollama_hosts()

        # --- Cleanup callbacks ---
        self._cleanup_callbacks: List[Callable] = []
        self._persistent_services: Set[Callable] = set()  # Services that should persist
        atexit.register(self._run_cleanup)

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
        if force_print and not self.SUMMARY_MODE:
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
    
    def get_cached_model_discovery(self) -> Optional[Dict[str, Any]]:
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
    
    def load_persistent_model_cache(self) -> Optional[Dict[str, Any]]:
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
    
    def save_persistent_model_cache(self, model_data: Dict[str, Any]) -> None:
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
    
    def get_tts_context_settings(self, context: str) -> Dict[str, Any]:
        """Get TTS settings for a specific context."""
        return self.TTS_CONTEXT_SETTINGS.get(context, {
            "speed": self.TTS_DEFAULT_SPEED,
            "voice": self.TTS_DEFAULT_VOICE
        })
    
    def set_tts_context_settings(self, context: str, settings: Dict[str, Any]) -> None:
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

g = Globals()