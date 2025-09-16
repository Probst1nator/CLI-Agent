# main.py
"""
CLI-Agent Main Entry Point - CURRENT/ACTIVE VERSION

This is the stable, production entry point for CLI-Agent (3,273 lines).
Contains all recent bug fixes and active development, including:
- Complete Monte Carlo Action Engine implementation
- Fixed ComputationalNotebook architecture (now uses direct tool calls)
- Full user interaction loop and tool invocation system
- All production features and stability improvements

For development and deployment, use this file.
Note: main_refactored.py is experimental and not ready for production use.
"""

# CRITICAL: Suppress warnings BEFORE any imports that might trigger them
import sys
import os
import warnings
import contextlib

# --- COMPREHENSIVE WARNING SUPPRESSION ---
# Must be set BEFORE importing ANY libraries that might generate warnings
# These environment variables affect the entire Python process lifecycle
if '--debug' not in sys.argv:
    # Suppress gRPC/protobuf/abseil warnings (affects runtime, not just imports)
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GRPC_TRACE'] = ''
    os.environ['GRPC_PYTHON_LOG_LEVEL'] = 'ERROR'
    
    # Suppress TensorFlow/ML library warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['ABSL_MIN_LOG_LEVEL'] = '3' 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Suppress Python warnings system-wide
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    
    # Additional library-specific suppression
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Hugging Face warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Transformers library
    
    # Create comprehensive stderr suppression context
    @contextlib.contextmanager 
    def suppress_stderr_completely():
        """Completely suppress stderr output during context"""
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
                
    @contextlib.contextmanager
    def suppress_warnings_runtime():
        """Suppress warnings during runtime operations"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_stderr = sys.stderr
            stderr_buffer = None
            # Redirect stderr for known noisy operations
            if '--debug' not in sys.argv:
                stderr_buffer = open(os.devnull, 'w')
                sys.stderr = stderr_buffer
            try:
                yield
            finally:
                if '--debug' not in sys.argv and stderr_buffer is not None:
                    stderr_buffer.close()
                sys.stderr = old_stderr
else:
    # Debug mode: show ALL warnings and verbose output
    warnings.filterwarnings('default')
    os.environ['GRPC_VERBOSITY'] = 'DEBUG'
    
    # Dummy context managers for debug mode
    @contextlib.contextmanager
    def suppress_stderr_completely():
        yield
        
    @contextlib.contextmanager 
    def suppress_warnings_runtime():
        yield

# NOW safe to import other modules
import time
import logging
import re
import asyncio
import threading  
import json
import argparse
import signal
from pathlib import Path
from termcolor import colored
from typing import Dict, List, Optional, Callable, Any
from pyfiglet import figlet_format

# Import with suppression in non-debug mode
if '--debug' not in sys.argv:
    with suppress_stderr_completely():
        from tools.viewpermissions import viewpermissions
else:
    from tools.viewpermissions import viewpermissions

start_time = time.time() # Global timer for startup and operations

# --- Global Interrupt Flag ---
# Used for signal-based interruption instead of KeyboardInterrupt exceptions
_interrupt_flag = threading.Event()

# --- Prompt Toolkit for Command Completion ---

# --- Project Path Setup ---
# Ensure all project modules can be imported correctly.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Tool and Core Imports ---
from tools.primitives.todos import todos
from tools.readfile import readfile
from tools.bash import bash
from shared.utils_audio import text_to_speech
from core.globals import g
from core.llm_router import LlmRouter, StreamInterruptedException, UserInterruptedException
from core.chat import Chat, Role
from agent.playbook.playbook_manager import PlaybookManager
from agent.utils_manager.utils_manager import UtilsManager
from utils.tool_context import tool_context
from core.permissions.enhanced_permissions import enhanced_permission_prompt
from shared.common_utils import extract_blocks
import traceback
import linecache

# --- Suppress Noisy Library Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")

# Import command system from extracted module

# Import display utilities from extracted module
from shared.utils.display_utils import colorize_model_list, print_startup_summary, show_unified_bindings_display

# Import logging utilities from extracted module  
from shared.utils.logging_utils import setup_logging, swap_to_simple_logging

# Import tool discovery from extracted module
from agent.core.tool_discovery import _generate_tool_documentation

# Import input handlers from extracted module
from agent.core.input_handlers import (
    handle_multiline_input, get_user_input_with_bindings, handle_screenshot_capture
)

# Import Monte Carlo engine from extracted module
from agent.monte_carlo.monte_carlo_engine import (
    select_best_branch, generate_branch
)

# Command system functions moved to agent.core.command_system

# Command completion and prompt session moved to agent.core.command_system

# --- Global Fallback Iterator for Cognitive Diversity ---
class FallbackModelIterator:
    """
    Global iterator to cycle through fallback models across all system components
    (branch generation, voting, guarding) to maximize cognitive diversity.
    """
    
    def __init__(self):
        self._fallback_models = []
        self._current_index = 0
        self._usage_count = {}  # Track how often each model has been used
        self._failure_log = []  # Track (failed_model, reason, replacement) tuples
        
    def update_fallback_models(self, llm_config: Dict):
        """Update the list of available fallback models from current config."""
        from core.llm_router import LlmRouter
        
        new_fallback_models = [
            key for key, data in llm_config.items()
            if (data.get('selected') and 
                key not in LlmRouter().failed_models and
                data.get('eval', 0) == 0 and
                data.get('guard', 0) == 0)
        ]
        
        if new_fallback_models != self._fallback_models:
            self._fallback_models = new_fallback_models
            self._current_index = 0  # Reset when models change
            self._usage_count = {model: 0 for model in self._fallback_models}
            
    def get_next_fallback(self, exclude: Optional[List[str]] = None) -> Optional[str]:
        """
        Get the next fallback model in round-robin fashion.
        
        Args:
            exclude: List of models to exclude from selection
            
        Returns:
            Next fallback model, or None if no models available
        """
        if not self._fallback_models:
            return None
            
        exclude = exclude or []
        available_models = [m for m in self._fallback_models if m not in exclude]
        
        if not available_models:
            return None
            
        # Start from current index and find next available model
        attempts = 0
        while attempts < len(self._fallback_models):
            candidate = self._fallback_models[self._current_index % len(self._fallback_models)]
            self._current_index = (self._current_index + 1) % len(self._fallback_models)
            
            if candidate in available_models:
                self._usage_count[candidate] = self._usage_count.get(candidate, 0) + 1
                return candidate
                
            attempts += 1
            
        return None
        
    def get_multiple_fallbacks(self, count: int, exclude: Optional[List[str]] = None) -> List[str]:
        """
        Get multiple fallback models, cycling through the list to maximize diversity.
        
        Args:
            count: Number of fallbacks needed
            exclude: List of models to exclude
            
        Returns:
            List of fallback models (may be shorter than count if not enough available)
        """
        exclude = exclude or []
        fallbacks = []
        
        for _ in range(count):
            fallback = self.get_next_fallback(exclude + fallbacks)
            if fallback:
                fallbacks.append(fallback)
            else:
                break
                
        return fallbacks
        
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for debugging."""
        return self._usage_count.copy()
    
    def record_failure(self, failed_model: str, reason: str, replacement: Optional[str] = None):
        """Record a model failure and its replacement."""
        self._failure_log.append((failed_model, reason, replacement))
        
    def get_failure_log(self) -> List[tuple]:
        """Get the failure log for this session."""
        return self._failure_log.copy()

# Separate iterators for different system components to maximize cognitive diversity
branch_fallback_iterator = FallbackModelIterator()
guard_fallback_iterator = FallbackModelIterator() 
evaluation_fallback_iterator = FallbackModelIterator()

# --- Signal Handling for Graceful Interruption ---
def setup_signal_handlers():
    """Setup signal handlers for graceful interruption during AI generation."""
    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) by setting interrupt flag and attempting immediate cancellation."""
        _interrupt_flag.set()
        print(colored("\n🛑 INTERRUPT SIGNAL RECEIVED - Cancelling AI generation...", "red", attrs=['bold']))

        # Attempt immediate provider cancellation if available
        try:
            from core.llm_router import LlmRouter
            router = LlmRouter()
            if hasattr(router, 'last_used_model') and router.last_used_model:
                # Get the current model's provider
                current_model = router.get_model([router.last_used_model])
                if current_model:
                    cancel_func = getattr(current_model.provider, 'cancel_current_generation', None)
                    if callable(cancel_func):
                        cancel_func()
                        print(colored("  → Provider cancellation initiated", "yellow"))
        except Exception:
            # Don't let provider cancellation errors break the signal handler
            pass

        print(colored("  → Returning to user input...", "yellow"))

    signal.signal(signal.SIGINT, signal_handler)

def check_interrupt():
    """Check if interrupt was requested and clear the flag."""
    if _interrupt_flag.is_set():
        _interrupt_flag.clear()
        return True
    return False

def offer_intelligent_recovery_options(context_chat):
    """
    Analyze interrupt metadata and offer context-appropriate recovery options.
    Returns user choice for how to proceed after interrupt.
    """
    if not hasattr(context_chat, 'metadata') or 'last_interrupt' not in context_chat.metadata:
        return 'continue'

    interrupt_info = context_chat.metadata['last_interrupt']
    print(colored("\n🔄 Recovery Options:", "cyan", attrs=['bold']))

    # Analyze the interrupt context and offer relevant options
    options = []

    if interrupt_info.get('had_partial_response', False):
        options.append(("partial", "Continue with partial response (if meaningful content was generated)"))

    if interrupt_info.get('monte_carlo_mode', False):
        options.append(("single", "Switch to single-model mode (faster, no branching)"))

    if interrupt_info.get('models_attempted', []):
        failed_models = interrupt_info['models_attempted']
        options.append(("retry_diff", f"Retry with different model (avoid {', '.join(failed_models[:2])}...)"))

    options.extend([
        ("retry_same", "Retry with same settings"),
        ("continue", "Continue normally"),
        ("exit", "Exit to user input")
    ])

    # Display options
    for i, (key, description) in enumerate(options, 1):
        print(f"  {i}. {description}")

    # Get user choice with timeout
    print(colored("\nChoose option (1-{}) or press Enter for default [continue]: ".format(len(options)), "cyan"), end="", flush=True)

    try:
        import select

        # Non-blocking input with 5-second timeout
        ready, _, _ = select.select([sys.stdin], [], [], 5.0)
        if ready:
            choice = input().strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    selected_key = options[idx][0]
                    print(colored(f"  → Selected: {options[idx][1]}", "green"))
                    return selected_key

        # Default choice after timeout
        print(colored("  → Using default: Continue normally", "yellow"))
        return 'continue'

    except Exception:
        # Fallback for systems without select support
        return 'continue'

# --- Custom Logging Setup ---
def parse_cli_args() -> argparse.Namespace:
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""
    
    parser = argparse.ArgumentParser(
        description="AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )

    parser.add_argument("-h", "--help", action="store_true", default=False,
                        help="Display this help")
    parser.add_argument("-a", "--auto", action="store_true", default=False,
                        help="Automatically execute safe commands...")
    parser.add_argument("-c", action="store_true", default=False,
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-l", "--local", action="store_true", default=False,
                        help="Use the local Ollama backend for processing. Sets g.FORCE_LOCAL=True.")
    parser.add_argument("-m", "--message", type=str, default=[], nargs='*', 
                        help="Enter one or more messages to process in sequence. Multiple messages can be passed like: -m 'first message' 'second message'. If no message is provided, multiline input mode will be activated.")
    parser.add_argument("-r", "--regenerate", action="store_true", default=False,
                        help="Regenerate the last response.")
    parser.add_argument("-v", "--voice", action="store_true", default=False,
                        help="Enable microphone input and text-to-speech output.")
    parser.add_argument("--online_voice", action="store_true", default=False,
                        help="Enable cloud-based speech transcription and TTS (Groq/OpenAI/ElevenLabs) with local fallback. By default, uses local engines only.")

    parser.add_argument("--full_output", dest='full_output_mode', action="store_true", default=True,
                        help="Full output mode. Shows live model output instead of hiding it.")
    parser.add_argument("-f", "--fast", action="store_true", default=False,
                        help="Fast mode (deprecated - kept for backward compatibility).")
    parser.add_argument("-img", "--image", action="store_true", default=False,
                        help="Take a screenshot using Spectacle for region selection (with automatic fallbacks if not available).")
    parser.add_argument("-o", "--online", action="store_true", default=False,
                        help="Force use of cloud AI.")
    parser.add_argument("-mct", "--monte-carlo", action="store_true", default=False,
                        help="Enable multi-LLM mode with branch generation, council evaluation, and execution guards.")
    
    parser.add_argument("-llm", "--llm", type=str, nargs='?', const="__select__", default=None,
                        help="Specify the LLM model key to use (e.g., 'gemini-2.5-flash', 'gemma3n:e2b'). Use without value to open selection menu.")
    
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open a web interface for the chat")
    parser.add_argument("--debug-chats", action="store_true", default=False,
                        help="Enable debug windows for chat contexts without full debug logging. Sets g.DEBUG_CHATS=True.")
    parser.add_argument("--private_remote_wake_detection", action="store_true", default=False,
                        help="Use private remote wake detection")
    
    # --- Logging Arguments ---
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable detailed debug logging to the console.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to a file to write detailed logs.")
    # --- End Logging Arguments ---

    parser.add_argument("-e", "--exit", action="store_true", default=False,
                        help="Exit after all automatic messages have been parsed successfully")
    
    # CORRECTED: Parse directly from sys.argv, as the pre-processing logic was removed.
    args, unknown_args = parser.parse_known_args(sys.argv[1:])

    if unknown_args or args.help:
        if not args.help:
            # Need a temporary logger setup for this specific warning
            logging.basicConfig()
            logging.getLogger().warning(colored(f"Unrecognized arguments {' '.join(unknown_args)}.", "yellow"))
        parser.print_help()
        exit(1)
    
    return args

# ConsoleFormatter moved to shared.utils.logging_utils

# SimpleFormatter moved to shared.utils.logging_utils

# FileFormatter moved to shared.utils.logging_utils

# setup_logging function moved to shared.utils.logging_utils

# swap_to_simple_logging function moved to shared.utils.logging_utils

# --- End Custom Logging Setup ---


# Display utility functions moved to shared.utils.display_utils




# print_startup_summary function moved to shared.utils.display_utils

# --- Eager setup of args ---
# This is done before heavy imports to allow --help to work quickly.
args = parse_cli_args()

# Store continue flag in global state for access by tools
g.CONTINUE_CONVERSATION = args.c

# Import figlet for later use
from pathlib import Path


# --- Main Application Imports (Lazy/Normal) ---
# Imports are moved below arg parsing and logging setup to allow for faster startup
# and to ensure logging is configured before any modules start logging.
from dotenv import load_dotenv
import warnings
import subprocess

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")
warnings.filterwarnings("ignore", message="Unrecognized FinishReason enum value*", module="proto.marshal.rules.enums", category=UserWarning)

# Removed ComputationalNotebook import - using direct tool calls instead
from core.ai_strengths import AIStrengths
from core.permissions.path_detector import PathDetector
from agent.text_painter import TextStreamPainter



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Lazy load functions from shared audio utils to avoid heavy imports at startup
def get_listen_microphone():
    from shared.utils_audio import listen_microphone  # noqa: E402 - Use improved Faster-Whisper with small model
    return listen_microphone

try:
    from shared.utils_tts import TtsUtil
except ImportError:
    class TtsUtil:
        @staticmethod
        def run(text, **kwargs):
            logging.warning(f"TtsUtil not found. Cannot speak: {text}")
            return json.dumps({"status": "error", "message": "TtsUtil not found"})

def get_extract_blocks():
    """
    Returns the extract_blocks function for code block extraction.
    This function acts as a getter to maintain compatibility with existing code.
    """
    return extract_blocks


def detect_malformed_xml(text: str) -> List[Dict[str, str]]:
    """
    Detects malformed XML blocks where opening and closing tags don't match.

    This function specifically looks for tag mismatches that would cause
    extract_blocks() to fail, such as <type>content</action>.

    Args:
        text (str): The text to analyze for malformed XML

    Returns:
        List[Dict[str, str]]: List of detected malformed XML issues
    """
    errors = []

    # Simple approach: Look for patterns like <tag1>content</tag2> where tag1 != tag2
    # This regex finds opening tag, content, and closing tag as separate groups
    mismatch_pattern = r'<([a-zA-Z0-9_]+)([^>]*?)>(.*?)</([a-zA-Z0-9_]+)>'

    for match in re.finditer(mismatch_pattern, text, re.DOTALL):
        opening_tag = match.group(1).strip()
        _ = match.group(2).strip()
        content = match.group(3).strip()
        closing_tag = match.group(4).strip()
        full_block = match.group(0)

        # Check if opening and closing tags match
        if opening_tag != closing_tag:
            # Make sure this isn't a nested structure by checking if content contains complete XML
            # If content has complete XML blocks, this might be a valid outer structure
            inner_complete_blocks = len(re.findall(r'<([a-zA-Z0-9_]+)[^>]*>.*?</\1>', content, re.DOTALL))
            content_tags = len(re.findall(r'<[a-zA-Z0-9_]+[^>]*>', content))

            # If content has tags but no complete blocks, this is likely a real mismatch
            if content_tags > 0 and inner_complete_blocks == 0:
                continue  # Skip, this is probably a complex nested case
            elif content_tags == 0:  # Simple content with no nested tags - likely a real error
                errors.append({
                    'type': 'tag_mismatch',
                    'opening_tag': opening_tag,
                    'closing_tag': closing_tag,
                    'error_description': f"Opening tag '<{opening_tag}>' does not match closing tag '</{closing_tag}>'",
                    'suggested_fix': f"Change '</{closing_tag}>' to '</{opening_tag}>'",
                    'block_content': full_block,
                    'line_context': content[:100] + "..." if len(content) > 100 else content
                })

    return errors


def generate_xml_error_context(errors: List[Dict[str, str]]) -> str:
    """
    Generate a context block with XML error information for the model.

    Args:
        errors: List of XML errors from detect_malformed_xml()

    Returns:
        str: Formatted context block with error details and suggestions
    """
    if not errors:
        return ""

    context_lines = ["XML Syntax Error Detected:"]
    context_lines.append("")

    for i, error in enumerate(errors, 1):
        if error['type'] == 'tag_mismatch':
            context_lines.append(f"Error {i}: {error['error_description']}")
            context_lines.append(f"  Found: <{error['opening_tag']}>...</{error['closing_tag']}>")
            context_lines.append(f"  Fix: {error['suggested_fix']}")
            context_lines.append("")

            # Show the problematic content with highlighting
            context_lines.append("Malformed XML:")
            context_lines.append(f"  {error['block_content']}")
            context_lines.append("")

        elif error['type'] == 'incomplete_block':
            context_lines.append(f"Error {i}: {error['error_description']}")
            context_lines.append(f"  Fix: {error['suggested_fix']}")
            context_lines.append("")

    context_lines.append("Due to these XML syntax errors, the affected tool blocks were ignored.")
    context_lines.append("Please correct the XML syntax and try again.")

    return f"<context>\n{chr(10).join(context_lines)}\n</context>"


def apply_default_llm_config():
    """
    Loads the cached LLM configuration from disk and applies it as the default
    for the current session. This makes LLM selections persistent.
    """
    llm_config = g.get_llm_config()
    if not llm_config:
        return # No saved config to apply

    # Preserve chronological selection order by using g.SELECTED_LLMS if available
    if g.SELECTED_LLMS:
        selected_llms = [key for key in g.SELECTED_LLMS if llm_config.get(key, {}).get('selected')]
    else:
        selected_llms = [key for key, data in llm_config.items() if data.get('selected')]
    
    # For evaluator LLMs, preserve order by checking g.SELECTED_LLMS first
    if g.SELECTED_LLMS:
        evaluator_llms = [key for key in g.SELECTED_LLMS if llm_config.get(key, {}).get('eval', 0) > 0]
    else:
        evaluator_llms = [key for key, data in llm_config.items() if data.get('eval', 0) > 0]
    
    # Only apply if there are actual selections, to not override command line flags unintentionally
    if selected_llms:
        g.SELECTED_LLMS = selected_llms
        g.EVALUATOR_LLMS = evaluator_llms
        # Count unique models by provider
        unique_models = list(set(selected_llms))  # Deduplicate
        provider_counts = {}
        for model in unique_models:
            emoji, provider_name = get_provider_emoji_and_name(model)
            provider_key = f"{emoji} {provider_name}"
            provider_counts[provider_key] = provider_counts.get(provider_key, 0) + 1
        
        provider_summary = ", ".join([f"{provider}: {count}" for provider, count in provider_counts.items()])
        logging.info(f"📋 Loaded latest LLM configuration. Total: {colored(str(len(unique_models)), 'green')} models ({provider_summary})")

def apply_monte_carlo_config():
    """
    Loads or creates a Monte Carlo configuration with branching, evaluation, and guarding.
    If no config exists, creates a default with the first 3 available models.
    """
    import json
    
    llm_config = g.get_llm_config()
    
    # Check if Monte Carlo config already exists (has beams/eval/guard values)
    has_monte_carlo_config = any(
        data.get('beams', 0) > 0 or data.get('eval', 0) > 0 or data.get('guard', 0) > 0 
        for data in llm_config.values()
    )
    
    if has_monte_carlo_config:
        # Load existing Monte Carlo configuration
        selected_llms = [key for key, data in llm_config.items() if data.get('selected')]
        evaluator_llms = [key for key, data in llm_config.items() if data.get('eval', 0) > 0]
        
        if selected_llms:
            g.SELECTED_LLMS = selected_llms
            g.EVALUATOR_LLMS = evaluator_llms
            unique_models = list(set(selected_llms))
            provider_counts = {}
            for model in unique_models:
                emoji, provider_name = get_provider_emoji_and_name(model)
                provider_key = f"{emoji} {provider_name}"
                provider_counts[provider_key] = provider_counts.get(provider_key, 0) + 1
            
            provider_summary = ", ".join([f"{provider}: {count}" for provider, count in provider_counts.items()])
            logging.info(f"🎯 Loaded Monte Carlo configuration. Total: {colored(str(len(unique_models)), 'green')} models ({provider_summary})")
        return
    
    # Create default Monte Carlo configuration with first 3 available models
    from core.llm_router import LlmRouter
    router = LlmRouter()
    
    # Discover available models
    available_models = []
    try:
        # Get all available models
        all_models = router.get_models()
        available_models.extend([model.model_key for model in all_models])
    except:
        # Fallback to hardcoded reliable models
        available_models = ['gemini-2.5-flash', 'gemini-2.5-flash-preview-05-20', 'gemma3:12b']
    
    # Take first 3 unique models
    selected_models = list(set(available_models))[:3]
    
    if not selected_models:
        logging.warning(colored("⚠️ No models available for Monte Carlo configuration", "yellow"))
        return
    
    # Create default Monte Carlo configuration
    new_config = {}
    for i, model in enumerate(selected_models):
        if i == 0:
            # First model: main branching model
            new_config[model] = {"selected": True, "beams": 2, "eval": 0, "guard": 1}
        elif i == 1:
            # Second model: evaluator
            new_config[model] = {"selected": True, "beams": 1, "eval": 2, "guard": 0}
        else:
            # Third model: guard
            new_config[model] = {"selected": True, "beams": 0, "eval": 0, "guard": 1}
    
    # Save the new configuration
    try:
        os.makedirs(os.path.dirname(g.LLM_CONFIG_PATH), exist_ok=True)
        with open(g.LLM_CONFIG_PATH, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Reload the configuration
        g.load_llm_config()
        g.SELECTED_LLMS = selected_models
        g.EVALUATOR_LLMS = [model for model, data in new_config.items() if data.get('eval', 0) > 0]
        
        logging.info(f"🎯 Created default Monte Carlo configuration with {colored(str(len(selected_models)), 'green')} models: {', '.join(selected_models)}")
        
    except Exception as e:
        logging.error(f"❌ Failed to save Monte Carlo configuration: {e}")
        # Continue with in-memory config
        g._llm_config.update(new_config)
        g.SELECTED_LLMS = selected_models
        g.EVALUATOR_LLMS = [model for model, data in new_config.items() if data.get('eval', 0) > 0]

async def llm_selection(args: argparse.Namespace, preselected_llms: Optional[List[str]] = None) -> List[str]:
    """
    Handles the LLM selection process using the enhanced LlmSelector UI.
    This function initializes the selector, awaits user input, and processes
    the results to configure the agent's LLM settings.
    """
    from agent.llm_selection import LlmSelector
    selector = LlmSelector()
    
    # The selector now handles its own UI, logic, and persistence.
    result = await selector.get_selection(
        preselected_llms=preselected_llms,
        force_local=args.local,
        save_selection=True  # Let the selector handle saving the config
    )

    # --- Process the structured result from the selector ---
    if result.get("status") != "Success":
        logging.warning(colored(f"# cli-agent: {result.get('message', 'LLM selection cancelled or failed.')}", "yellow") )
        return []

    # Handle the special 'any_local' case
    if result.get("selection_type") == "any_local":
        g.FORCE_LOCAL = True
        args.local = True
        args.llm = None # No specific LLM is set
        logging.info(colored("# cli-agent: 'Any but local' option selected. Local mode enabled.", "green") )
        return []

    # Handle specific model selections
    selected_configs = result.get("selected_configs", [])
    
    # Identify evaluator models and set them in globals
    evaluator_models = [config['model_key'] for config in selected_configs if config.get('eval', 0) > 0]
    g.EVALUATOR_LLMS = evaluator_models
    
    selected_llms = [config['model_key'] for config in selected_configs]
    
    # Update globals with the latest selection
    args.llm = selected_llms
    g.SELECTED_LLMS = selected_llms

    # Reload the config in `g` so the main loop gets the fresh settings
    g.load_llm_config()

    if not args.llm:
        logging.info(colored("# cli-agent: LLM set to auto, type (-h) for info", "green") )
    else:
        # Count unique models by provider and configuration
        # Use selected_llms directly to preserve the exact selection order from LLM selector
        # Deduplicate while preserving order (maintain user's selection order)
        unique_models = []
        seen = set()
        for model in selected_llms:  # Use selected_llms instead of args.llm
            if model not in seen:
                unique_models.append(model)
                seen.add(model)
        provider_counts = {}
        branches_count = eval_count = guard_count = 0
        
        for model in unique_models:
            emoji, provider_name = get_provider_emoji_and_name(model)
            provider_key = f"{emoji} {provider_name}"
            provider_counts[provider_key] = provider_counts.get(provider_key, 0) + 1
        
        for config in selected_configs:
            if config.get('beams', 0) > 0 or config.get('branches', 0) > 0:  # Support both old and new terminology
                branches_count += 1
            if config.get('eval', 0) > 0:
                eval_count += 1
            if config.get('guard', 0) > 0:
                guard_count += 1
        
        # Display Monte Carlo Action Engine overview instead of simple summary
        def get_count_color(count: int) -> str:
            """Returns a color string based on the model count for at-a-glance status."""
            if count == 1: return 'red'
            if count == 2: return 'yellow' 
            if count == 3: return 'green'
            if count > 3: return 'light_green'
            return 'white'
        
        # Categorize models by their roles
        branch_models = [config['model_key'] for config in selected_configs if config.get('beams', 0) > 0 or config.get('branches', 0) > 0]
        eval_models = [config['model_key'] for config in selected_configs if config.get('eval', 0) > 0]
        guard_models = [config['model_key'] for config in selected_configs if config.get('guard', 0) > 0]
        
        # Calculate fallback models (selected but not specialized)
        specialized_models = set(branch_models + eval_models + guard_models)
        fallback_models = [model for model in unique_models if model not in specialized_models]
        
        logging.info(colored("---" + " Monte Carlo Action Engine " + "---", "yellow"))

        if branch_models:
            count = 0
            # Count branches based on 'beams'
            for config in selected_configs:
                if config.get('beams', 0) > 0 or config.get('branches', 0) > 0:
                    count += config.get('beams', config.get('branches', 1))
            color = get_count_color(count)
            logging.info(colored(f"  🌿 {count} Branch Generators:".ljust(27), color) + " " + colorize_model_list(branch_models))
        
        if eval_models:
            count = len(eval_models)
            color = get_count_color(count)
            logging.info(colored(f"  ⚖️  {count} Branch Evaluators:".ljust(28), color) + " " + colorize_model_list(eval_models))
        
        if guard_models:
            count = len(guard_models)
            color = get_count_color(count)
            logging.info(colored(f"  🛡️  {count} Execution Guards:".ljust(28), color) + " " + colorize_model_list(guard_models))
        
        # Always show fallback models for visibility, even if none are configured
        count = len(fallback_models)
        color = "light_green" if count > 0 else "red"
        logging.info(colored(f"  🔄 {count} Fallback Models:".ljust(27), color) + " " + colorize_model_list(fallback_models))
        
        # Auto-sync args.auto with guard LLMs when selection changes
        if guard_count > 0 and not args.auto:
            args.auto = True
            logging.info(colored(f"🔄 Auto-execution enabled: {guard_count} guard LLMs active", "green"))
        elif guard_count == 0 and args.auto:
            # Inform user that auto is on but no guards are available
            logging.info(colored("❗ Auto-execution (args.auto) automatically disabled: no guard LLMs configured (manual confirmation will be required)", "red"))

    return selected_llms



def create_interruption_callback(
    response_buffer: List[str],
    painter: TextStreamPainter,
    lock: Optional[asyncio.Lock] = None,
):
    """
    Factory to create a stateful callback for handling LLM streams.

    This callback prints the stream, accumulates it into a buffer, and can
    interrupt by raising StreamInterruptedException when a complete code block
    (e.g., ```...```) or a complete tag for a known tool (e.g., <bash>...</bash>) is detected.
    This logic is consistent with the `extract_blocks` utility.

    Args:
        response_buffer: A list with a single string to act as a mutable buffer.
        painter: The TextStreamPainter for coloring output.
        lock: An optional asyncio.Lock for concurrent-safe printing.

    Returns:
        An async callback function.
    """
    # Pre-compile regex for efficiency. re.DOTALL allows '.' to match newlines.
    # CODE_BLOCK_REGEX = re.compile(r"```.*?(```)", re.DOTALL)
    
    # Dynamically build a regex for registered tool blocks. This is more robust
    # and prevents premature interruption on non-tool XML tags like <filepath>.
    # Define tags that should NOT interrupt the stream, allowing the agent to output them fully before pausing.
    non_interrupting_tags = {'think', 'todos'}

    # Get all tool delimiters and add the final <response> tag.
    all_delimiters = list(getattr(g, 'tool_classes', {}).keys()) + ['response']
    
    # In GUI mode, also exclude 'response' from interrupting since we want visualresponse only
    if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
        non_interrupting_tags = non_interrupting_tags.union({'response'})

    # Filter this list to get only the tags that SHOULD cause an interruption.
    interrupting_delimiters = [d for d in all_delimiters if d not in non_interrupting_tags]
    
    # Build the regex from the filtered list of interrupting tags.
    tool_tag_pattern = "|".join(re.escape(d) for d in interrupting_delimiters)
    XML_TOOL_BLOCK_REGEX = re.compile(f"<({tool_tag_pattern})[^>]*>.*?</\\1>", re.DOTALL) if tool_tag_pattern else None
    
    async def _callback(chunk: str):
        # Check for user interrupt immediately at start of callback for faster response
        if check_interrupt():
            if not g.SUMMARY_MODE:
                print("\n")
            raise StreamInterruptedException(response_buffer[0])

        # Batch processing for better performance while maintaining interrupt responsiveness
        char_batch = []
        batch_size = min(50, len(chunk))  # Process in small batches for responsiveness

        for i, char in enumerate(chunk):
            # Check for user interrupt every 10 characters or at batch boundaries for efficiency
            if i % 10 == 0 and check_interrupt():
                if not g.SUMMARY_MODE:
                    print("\n")
                raise StreamInterruptedException(response_buffer[0] + ''.join(char_batch) + char)

            char_batch.append(char)

            # Process batch when full or at special characters
            should_process_batch = (
                len(char_batch) >= batch_size or
                char in '`><' or
                i == len(chunk) - 1  # Last character
            )

            if should_process_batch:
                batch_str = ''.join(char_batch)

                # Print the batch
                if not g.SUMMARY_MODE or g.DEBUG_MODE:
                    if lock:
                        async with lock:
                            print(painter.apply_color(batch_str), end="", flush=True)
                    else:
                        print(painter.apply_color(batch_str), end="", flush=True)

                # Append batch to shared buffer
                response_buffer[0] += batch_str
                current_buffer = response_buffer[0]

                # Clear batch
                char_batch = []

                # Check for interruption conditions when we encounter special characters
                if char in '`><':
                    # # Check for a complete Markdown code block
                    # code_match = CODE_BLOCK_REGEX.search(current_buffer)
                    # if code_match:
                    #     full_block = code_match.group(0).strip()
                    #     if not g.SUMMARY_MODE:
                    #         print("\n")
                    #     raise StreamInterruptedException(full_block)

                    # Check for a complete block of a known tool
                    if XML_TOOL_BLOCK_REGEX:
                        tool_match = XML_TOOL_BLOCK_REGEX.search(current_buffer)
                        if tool_match:
                            full_block = tool_match.group(0).strip()
                            if not g.SUMMARY_MODE:
                                print("\n")
                            raise StreamInterruptedException(full_block)

    return _callback

def get_provider_emoji_and_name(model_key: str, llm_router=None) -> tuple[str, str]:
    """
    Get the emoji and provider name for a model based on its provider class.
    Returns a tuple of (emoji, provider_name)
    """
    # Handle local Ollama models (contain colon)
    if ':' in model_key:
        return "🏠", "Ollama"
    
    # Try to get provider from LLM router if available
    if llm_router:
        try:
            llm_obj = llm_router.get_model([model_key])
            if llm_obj and llm_obj.provider:
                provider_class_name = llm_obj.provider.__class__.__name__
                
                # Determine emoji based on provider characteristics
                if 'Ollama' in provider_class_name:
                    emoji = "🏠"
                elif 'Human' in provider_class_name:
                    emoji = "👤"
                else:
                    emoji = "☁️  "
                
                # Clean provider name from class name
                provider_name = provider_class_name.replace('API', '').replace('Client', '').replace('Interface', '')
                
                return emoji, provider_name
        except:
            pass
    
    # Fallback: use model name prefix to determine provider
    if model_key.startswith('gemini'):
        return "☁️", "Google"
    elif model_key.startswith('gpt') or model_key.startswith('o1'):
        return "☁️", "OpenAI"
    elif model_key.startswith('claude'):
        return "☁️", "Anthropic"
    elif any(model_key.startswith(prefix) for prefix in ['llama', 'qwen', 'moonshotai']):
        return "☁️", "Groq"
    else:
        # Generic cloud provider
        provider_name = model_key.split('-')[0].title()
        return "☁️", provider_name


# Note: The type hint for 'UtilsManager' is a string to avoid circular dependency issues.
# The full definition of other functions like get_extract_blocks, g, LlmRouter, etc.
# are assumed to exist elsewhere in the project.


async def get_gui_input() -> str:
    """
    Get input from browser interface instead of terminal.
    Monitors the visual_input_queue from the web server.
    """
    import queue
    
    if not hasattr(g, 'visual_input_queue'):
        raise Exception("Visual input queue not available for GUI input")
    
    logging.info(colored("🌐 Waiting for browser input...", "blue"))
    
    # Wait for input from the browser
    while True:
        try:
            # Check if we have input from the browser
            if not g.visual_input_queue.empty():
                user_input = g.visual_input_queue.get_nowait()

                # Check for special reset command from browser
                if user_input.strip() == "__AGENT_RESET__":
                    logging.info(colored("🔄 Received agent reset request from browser", "yellow"))
                    # Set a flag to trigger reset (will be handled in main loop)
                    g.AGENT_RESET_REQUESTED = True
                    return ""  # Return empty to trigger loop continuation

                logging.info(colored(f"🌐 Received browser input: {user_input[:50]}...", "green"))
                return user_input
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Error checking visual input queue: {e}")
            
        # Small delay to prevent busy waiting
        await asyncio.sleep(0.1)


async def construct_engineered_prompt(voice_agent: str, gui_mode: bool = False) -> Chat:
    """
    Constructs the initial Chat object with a dynamically generated system prompt.
    This prompt includes the agent's core principles and a list of available tools.
    """
    voice_mode_text = 'The user is interacting with you in speech mode, please be aware of potential transcription issues. Provide concise responses and be helpful to the user.\n' if voice_agent else ''
    base_instructions = f'''**SYSTEM: Agent Protocol & Capabilities**

You are a sophisticated AI agent capable of thinking and using tools to achieve your todos. Your primary directive is to understand user requests, formulate a plan, and perform actions to achieve the current needs.
{voice_mode_text}
### Guiding Principles (Your Core Logic)
The following loop is a suggestion of how to behave effectively, you may freely reorder your usage of the xml tools if it fits your needs, you aim for long term stability and progress in your responses.
1.  **DECOMPOSE**: Analyze the user's request and break it down into the required steps. Start with: `<think>...</think>`.
2.  **PLAN & TRACK**: Display/update your <todos>...</todos> to align your next step(s) with the stated long term goal(s).'''
    
    step_4 = "4.  **OBSERVE & ATTEND**: After executing a tool, ALWAYS use <think> and/or <todos> to update your observations and plan your next move for as long as needed, then, continue with the next action."

    # In GUI mode, modify the tool documentation and response behavior
    if gui_mode:
        # Replace 'response' instructions with 'visualresponse'
        base_instructions = base_instructions.replace("response", "visualresponse")
        tool_docs = _generate_tool_documentation(["response"])
    else:
        tool_docs = _generate_tool_documentation()

    

    
    # Combine all parts into the final instruction message
    default_inst = f"{base_instructions}\n{tool_docs}\n{step_4}"
    
    # Create and configure the new Chat object
    context_chat = Chat(debug_title="Main Context Chat")
    context_chat.set_instruction_message(default_inst)
    
    return context_chat


async def reset_agent_state(args: argparse.Namespace, context_chat: Chat) -> Chat:
    """
    Completely reset the agent state to a fresh start.
    Clears all conversation history, tool context, images, and reconstructs the engineered prompt.
    """
    from core.globals import g

    logging.info(colored("🔄 Performing complete agent reset...", "yellow"))

    # Clear chat context
    if context_chat:
        context_chat.messages.clear()
        if hasattr(context_chat, 'base64_images'):
            context_chat.base64_images.clear()
        if hasattr(context_chat, 'metadata'):
            context_chat.metadata.clear()

    # Reset tool context
    tool_context.reset()

    # Clear global tool execution state
    if hasattr(g, 'last_executed_tools_info'):
        g.last_executed_tools_info = None

    # Clear reset flag
    g.AGENT_RESET_REQUESTED = False

    # Reconstruct fresh engineered prompt
    fresh_context_chat = await construct_engineered_prompt(args.voice, args.gui)

    # Clear visual interface
    try:
        from tools.visualresponse import visualresponse
        session = visualresponse._get_session()
        if session and session.is_healthy():
            clear_payload = {
                "id": f"reset-clear-{int(time.time() * 1000000)}",
                "action": "clear",
                "type": "system"
            }
            session.send_update(clear_payload)
            logging.debug("Sent clear command to visual interface")
    except Exception as e:
        logging.debug(f"Failed to clear visual interface: {e}")

    logging.info(colored("✅ Agent reset completed - fresh start initialized", "green"))
    return fresh_context_chat


async def save_current_llm_config_as_default():
    """Saves the current session's LLM selection as the new default."""
    from core.globals import g # Lazy import
    logging.info(colored("💾 Saving current LLM selection as the new default...", "magenta"))
    
    # Get the complete, most recent configuration data for ALL models
    # force_reread ensures we have the latest base config before modifying.
    full_config = g.get_llm_config(force_reread=True) 
    
    # Get the set of models active in THIS session
    currently_selected_models = set(g.SELECTED_LLMS)
    
    # Update the 'selected' status in the full config based on the current session
    for model_key, data in full_config.items():
        data['selected'] = model_key in currently_selected_models
    
    # Save the modified configuration back to the persistent file
    g.save_llm_config(full_config)
    
    logging.info(colored(f"✅ Default configuration updated with {len(currently_selected_models)} selected models.", "green"))


async def main(
    input_event: Optional[asyncio.Event] = None, 
    input_lock: Optional[threading.Lock] = None, 
    shared_input: Optional[Dict[str, Any]] = None,
    tui_callback: Optional[Callable] = None
) -> None:
    try:
        # In TUI mode, create default args instead of parsing CLI
        if input_event:  # TUI mode
            # Create default args for TUI mode
            global args
            args = argparse.Namespace(
                debug=False,
                log_file=None,
                auto=False,
                local=False,
                message=[],
                regenerate=False,
                voice=False,
                online_voice=False,
                private_remote_wake_detection=False,
                gui=False,
                c=False,
                llm=None,
                fast=False,
                min=False,
                image=False,
                full_output_mode=False,
                exit=False,
                debug_chats=False,
                monte_carlo=False,
                online=False
            )
            
            # Setup logging properly for TUI mode (output goes through print/stdout)
            setup_logging(args.debug, args.log_file, tui_mode=True)
            
            # Setup signal handlers for graceful interruption
            setup_signal_handlers()
            
            # Display header in TUI mode (stdout is redirected before agent starts)
            header_dashes = colored("# # # # # # # # # # # # # # # # # # # # # # # # # #", "blue") 
            header = figlet_format("CLI-Agent", font="slant")
            print(header_dashes)
            print(colored(header, "cyan", attrs=["bold"]))
            print(header_dashes)
            logging.info("Starting CLI-Agent...")
        
        # If not in TUI mode, display the header and perform standard logging setup.
        elif not tui_callback:
            setup_logging(args.debug, args.log_file)
            g.DEBUG_MODE = args.debug
            
            # Setup signal handlers for graceful interruption
            setup_signal_handlers()
            
            header_dashes = colored("# # # # # # # # # # # # # # # # # # # # # # # # # #", "blue")
            header = figlet_format("CLI-Agent", font="slant")
            print(header_dashes)
            print(colored(header, "cyan", attrs=["bold"]))
            print(header_dashes)
            logging.info("Starting CLI-Agent...")

        load_dotenv(g.CLIAGENT_ENV_FILE_PATH)
        
        if args.monte_carlo:
            apply_monte_carlo_config()
        elif not args.llm and not args.local and not args.fast:
            apply_default_llm_config()

        # Create the UtilsManager first, as it's needed for prompt generation
        # Suppress noisy gRPC/SSL warnings from web search tools unless in debug mode
        if '--debug' not in sys.argv:
            with suppress_stderr_completely():
                utils_manager = UtilsManager()
        else:
            utils_manager = UtilsManager()

        # Configure TTS engine priority based on --remote-tts flag
        if getattr(args, 'online_voice', False):
            # User explicitly requested remote TTS - prioritize API engines
            remote_priority = ["openai", "elevenlabs", "piper", "coqui", "huggingface", "system"]
            g.set_tts_engine_priority(remote_priority)
            if not tui_callback: logging.info(colored("# cli-agent: Online voice enabled - using cloud TTS and transcription APIs", "green"))
        
        stdout_buffer, stderr_buffer = "", ""
        def stdout_callback(text: str):
            nonlocal stdout_buffer
            print(text, end="")
            stdout_buffer += text
        def stderr_callback(text: str):
            nonlocal stderr_buffer
            print(colored(text, "red"), end="")
            stderr_buffer += text
            
        web_server = None
        if args.gui:
            # Set GUI mode flag for behavior modifications
            g.GUI_MODE = True
            
            # Auto-start visualresponse server for GUI mode (this opens the main interface)
            try:
                from tools.visualresponse import visualresponse
                logging.info(colored("🌐 Starting GUI mode - opening visualresponse interface...", "blue"))
                # Get a session which will start the server and open the browser
                session = visualresponse._get_session()
                logging.info(colored(f"✅ GUI mode active - interface available at http://127.0.0.1:{session.port}", "green"))
            except ImportError as e:
                logging.error(colored(f"❌ Could not start visualresponse server: {e}", "red"))
            except Exception as e:
                logging.error(colored(f"❌ Error starting visualresponse: {e}", "red"))
        
        context_chat: Optional[Chat] = None
        if args.c or args.regenerate:
            try:
                context_chat = Chat.load_from_json()
                if args.regenerate:
                    if not tui_callback: logging.info(colored("Loading previous chat for regeneration.", "green"))
                    if not context_chat or len(context_chat.messages) < 2:
                        if not tui_callback: logging.critical(colored("# cli-agent: No sufficient chat history found.", "red") )
                        exit(1)
                    if context_chat.messages[-1][0] == Role.ASSISTANT:
                        context_chat.messages.pop()
                    if not tui_callback: logging.info(colored("# cli-agent: Will regenerate response.", "green"))
                else:
                    if not tui_callback: logging.info(colored("Continuing previous chat.", "green"))
            except FileNotFoundError:
                if args.regenerate:
                    if not tui_callback: logging.critical(colored("No previous chat found to regenerate. Exiting.", "red") )
                    exit(1)
                else:
                    if not tui_callback: logging.warning(colored("No previous chat found. Starting a new chat.", "yellow") )
                    context_chat = None

        base64_images: List[str] = []
        if args.image:
            if not tui_callback: logging.info(colored("# cli-agent: Taking screenshot due to --img flag...", "green"))
            base64_images.extend(await handle_screenshot_capture())
            args.image = False
        
        if tui_callback:
            tui_callback("Initializing agent...", "system")

        if args.local and not g.SELECTED_LLMS:
            if not tui_callback: logging.info(colored("# cli-agent: Local mode (-l) detected. Selecting all available local models.", "green"))
            local_models = LlmRouter.get_models(force_local=True)
            if local_models:
                g.SELECTED_LLMS = [model.model_key for model in local_models]
            else:
                if not tui_callback: logging.warning(colored("# cli-agent: Local mode enabled, but no local models were found.", "yellow"))

        if args.llm == "__select__":
            await llm_selection(args, preselected_llms=g.SELECTED_LLMS)
            args.llm = None
        elif args.llm:
            g.SELECTED_LLMS = [args.llm]
            
            # If -mct and -llm are used together, update Monte Carlo config with this model
            if args.monte_carlo:
                import json
                try:
                    current_config = g.get_llm_config()
                    # Add the specified model to the Monte Carlo config
                    current_config[args.llm] = {"selected": True, "beams": 2, "eval": 1, "guard": 1}
                    
                    # Save the updated configuration
                    os.makedirs(os.path.dirname(g.LLM_CONFIG_PATH), exist_ok=True)
                    with open(g.LLM_CONFIG_PATH, 'w') as f:
                        json.dump(current_config, f, indent=2)
                    
                    g.load_llm_config()  # Reload to apply changes
                    logging.info(f"🎯 Updated Monte Carlo configuration with {colored(args.llm, 'green')}")
                except Exception as e:
                    logging.error(f"❌ Failed to update Monte Carlo configuration: {e}")
        
        if not args.monte_carlo:
            args.auto = False
        else:
            llm_config = g.get_llm_config()
            guard_count = sum(data.get('guard', 0) for data in llm_config.values() if data.get('selected'))
            if guard_count > 0 and not args.auto:
                args.auto = True

        is_tui = bool(tui_callback)
        print_startup_summary(args, tui_mode=is_tui)
        
        # Model discovery is now manual-only via 'u' key in LLM selection menu
        # if not tui_callback:
        #     logging.info(colored("  - Starting background model discovery...", "blue"))
        # else:
        #     logging.info(colored("  - Starting background model discovery...", "blue"))

        # g.start_background_model_discovery()
        
        # Store the callbacks globally for lazy notebook initialization
        g.stdout_callback = stdout_callback
        g.stderr_callback = stderr_callback
        g.tui_callback = tui_callback  # Store tui_callback as well for consistent logging
        
        if tui_callback: tui_callback("  - Loading utilities and playbooks...", "system")
        playbook_manager = PlaybookManager(vector_db=utils_manager.vector_db)
        
        if tui_callback: tui_callback("✔ Initialization complete.", "system_ok")
        else: 
            # Display tool loading information
            if hasattr(g, 'loaded_tool_info') and g.loaded_tool_info:
                tool_info = g.loaded_tool_info
                core_tools_list = ", ".join(tool_info['loaded_tools'])
                logging.info(colored("--- Tool Configuration ---", "blue"))
                logging.info(colored(f"  🔧 {tool_info['tool_count']} Core Tools: {core_tools_list}", "cyan"))
                
                # Display failed/skipped tools with warnings
                if tool_info['failed_tools']:
                    failed_list = ", ".join(tool_info['failed_tools'][:3])  # Show first 3
                    if len(tool_info['failed_tools']) > 3:
                        failed_list += f" (and {len(tool_info['failed_tools']) - 3} more)"
                    logging.info(colored(f"  ⚠️  Failed to load: {failed_list}", "yellow"))
                
                if tool_info['skipped_tools']:
                    skipped_list = ", ".join(tool_info['skipped_tools'])
                    logging.info(colored(f"  ⏭️  Skipped: {skipped_list}", "yellow"))
                
                # Display GUI tools if any
                if tool_info['gui_tools']:
                    gui_status = ", ".join(tool_info['gui_tools'])
                    logging.info(colored(f"  📁 GUI Tools: {gui_status}", "cyan"))
                
                logging.info(colored("--- Ready for User Input ---", "blue"))
            
            logging.info(colored("Tools ready.", "blue"))

        if context_chat is None:
            context_chat = await construct_engineered_prompt(args.voice, args.gui)
            
        # Install visual input bridge for browser integration
        try:
            from agent.visual_input_bridge import install_visual_input_bridge
            install_visual_input_bridge()
        except (ImportError, Exception) as e:
            logging.debug(f"Visual input bridge not available - browser integration disabled: {e}")

        if '-m' in sys.argv or '--message' in sys.argv:
            if not args.message:
                multiline_input = handle_multiline_input()
                if multiline_input.strip():
                    args.message.append(multiline_input)

        if not tui_callback:
            logging.info(colored("Agent Ready.", "magenta"))
            show_unified_bindings_display()
            swap_to_simple_logging()

        user_interrupt = False
        if not (args.c or args.regenerate):
            context_chat.add_message(Role.USER, "Please list the 5 most recently modified files of the cwd")
            if (args.gui):
                bootstrap_message_1 = f"""<think>
To list the 5 most recently modified files in the current working directory (excluding directories), a combination of ls, grep, and head commands can be used.
</think><todos>
- [ ] List 5 most recently modified files.
</todos><bash>
ls -tp | grep -v /$ | head -5
</bash><context>
{await bash.run_silent("ls -tp | grep -v /$ | head -5")}
</context><todos>
- [x] List 5 most recently modified files.
</todos><visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
### Most Recently Modified Files:
{await bash.run_silent("ls -tp | grep -v /$ | head -5 | sed 's/^/- /'")}
  </content>
</visualresponse>"""
                bootstrap_message_2 = """<visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
I have successfully viewed my permissions. Is there anything else I can assist you with?
  </content>
</visualresponse>"""
            else:
                bootstrap_message_1 = f"""<think>
To list the 5 most recently modified files in the current working directory (excluding directories), a combination of ls, grep, and head commands can be used.
</think><todos>
- [ ] List 5 most recently modified files.
</todos><bash>
ls -tp | grep -v /$ | head -5
</bash><context>
{await bash.run_silent("ls -tp | grep -v /$ | head -5")}
</context><todos>
- [x] List 5 most recently modified files.
</todos><response>
The 5 most recently modified files have been listed. Would you like me to inspect any of them?
</response>"""
                bootstrap_message_2 = """<response>
I have successfully viewed my permissions. Is there anything else I can assist you with?
</response>"""
            
            context_chat.add_message(Role.ASSISTANT, bootstrap_message_1)
            permissions_str = viewpermissions.run()
            if ("No automatic permissions set" not in permissions_str):
                context_chat.add_message(Role.USER, "view your permissions")
                context_chat.add_message(Role.ASSISTANT, f"""<todos>
- [ ] View permissions.
</todos><viewpermissions>
</viewpermissions><context>
{permissions_str}
</context><todos>
- [x] View permissions.
</todos>""" + bootstrap_message_2)
        
        assistant_response = ""

        initializing: bool = True
        while True:
            LlmRouter().failed_models.clear()
            user_input: Optional[str] = None
            
            g.LLM_STRENGTHS = []
            assistant_response = ""
            
            if not args.monte_carlo:
                args.auto = False
            else:
                llm_config = g.get_llm_config()
                guard_count = sum(
                    data.get('guard', 0) 
                    for data in llm_config.values() 
                    if data.get('selected')
                )
                if guard_count > 0 and not args.auto:
                    args.auto = True
            
            g.FORCE_LOCAL = args.local
            g.DEBUG_CHATS = args.debug_chats
            g.LLM = args.llm
            g.SUMMARY_MODE = not args.full_output_mode
            
            # # Auto compact chat length limit in chars
            # if (len(str(context_chat)) > 100000):
            #     print("# cli-agent: 100.000 char threshold reached: Auto compacting conversation...")
            #     g.AGENT_IS_COMPACTING = True

            if args.regenerate:
                user_input = ""
                args.regenerate = False
            elif args.message:
                user_input = args.message.pop(0)
                print(colored(f"💬 Processing message: {user_input}" + (f" (⏳ {len(args.message)} more queued)" if args.message else ""), 'blue', attrs=['bold']), flush=True)

                # Send user message to visual interface in GUI mode
                if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
                    try:
                        from tools.visualresponse import visualresponse
                        result = visualresponse.send_user_message(user_input)
                        logging.info(f"User message send result: {result} - Message: {user_input[:50]}...")
                    except Exception as e:
                        logging.debug(f"Failed to send user message to visual interface: {e}")

                # CRITICAL SECURITY: Process commands BEFORE sending to AI
                # Commands like /help, /exit, etc. should NEVER reach the AI
                # This prevents users from accidentally exposing system commands to AI processing
                if user_input.startswith('/'):
                    from agent.core.command_system import COMMAND_MAP, _COMMAND_DEFINITIONS
                    
                    # Extract command and arguments
                    parts = user_input.split()
                    command_alias = parts[0]
                    canonical_command = COMMAND_MAP.get(command_alias)
                    
                    if not canonical_command:
                        logging.error(colored(f"# cli-agent: Unknown command '{command_alias}'. Type '/help' for a list of commands.", "red"))
                        continue  # Skip this message and process next one
                    
                    # Handle each command type - these should NEVER go to the AI
                    if canonical_command == 'help':
                        # Show startup summary and command list
                        print_startup_summary(args)
                        print(colored("\n--- Available Commands ---", "yellow"))
                        CMD_WIDTH = 28
                        for cmd_info in _COMMAND_DEFINITIONS:
                            aliases = ', '.join(cmd_info['aliases'])
                            line = f"  {colored(aliases.ljust(CMD_WIDTH), 'white')}{cmd_info['description']}"
                            print(line)
                        continue  # Skip AI processing
                        
                    elif canonical_command == 'exit':
                        logging.info(colored("# cli-agent: Exiting...", "green"))
                        exit(0)
                        
                    elif canonical_command == 'clear':
                        if context_chat:
                            context_chat.messages.clear()
                        context_chat = await construct_engineered_prompt(args.voice, args.gui)
                        logging.info(colored("# cli-agent: Chat cleared.", "green"))
                        continue  # Skip AI processing

                    elif canonical_command == 'reset':
                        context_chat = await reset_agent_state(args, context_chat)
                        continue  # Skip AI processing

                    elif canonical_command == 'save':
                        await save_current_llm_config_as_default()
                        continue  # Skip AI processing
                        
                    elif canonical_command == 'local':
                        args.local = not args.local
                        g.FORCE_LOCAL = args.local
                        logging.info(colored(f"# cli-agent: Local mode toggled {'on' if args.local else 'off'}.", "green"))
                        continue  # Skip AI processing
                        
                    elif canonical_command == 'voice_mode':
                        args.voice = not args.voice
                        logging.info(colored(f"# cli-agent: Voice mode toggled {'on' if args.voice else 'off'}.", "green"))
                        continue  # Skip AI processing
                        
                    elif canonical_command == 'auto_execute':
                        args.auto = not args.auto
                        logging.info(colored(f"# cli-agent: Auto-execution toggled {'on' if args.auto else 'off'}.", "green"))
                        continue  # Skip AI processing
                        
                    elif canonical_command == 'online':
                        args.online = not args.online
                        logging.info(colored(f"# cli-agent: Online mode toggled {'on' if args.online else 'off'}.", "green"))
                        continue  # Skip AI processing
                        
                    else:
                        # Handle other commands that might need more complex processing
                        # For now, log and continue to prevent AI access
                        logging.warning(colored(f"# cli-agent: Command '{canonical_command}' not fully implemented in message mode.", "yellow"))
                        continue  # Skip AI processing
            elif args.voice:
                user_input = ""
                
                # --- NEW: Robustly extract the final response for TTS ---
                # Get the full content of the agent's last message.
                last_message_content = context_chat.messages[-1][1]
                spoken_text = ""
                
                try:
                    # Use a greedy search to find the *last* occurrence of the <response> block.
                    # '.*' at the start is greedy and consumes the string, then backtracks
                    # to find the rightmost possible match. This is more efficient than findall.
                    match = re.search(r'.*<response[^>]*>(.*?)</response>', last_message_content, re.DOTALL)
                    
                    if match and not initializing:
                        # If a match is found, extract its content from capture group 1.
                        spoken_text = match.group(1).strip()
                    elif not initializing:
                        logging.info(colored("WARNING: No <response> tag found in the last message for TTS.", "yellow"))
                        
                except Exception as e:
                    # Fails gracefully if last_message_content is not a string or another error occurs.
                    logging.error(colored(f"Error extracting response for TTS: {e}", "red"))
                    spoken_text = ""
                # --- END of new logic ---

                while (user_input == ""):
                    # The text-to-speech call now happens inside the loop.
                    if args.voice and spoken_text:
                        try:
                            text_to_speech(text=spoken_text, use_piper=True)
                            logging.info(colored("TTS call completed.", "cyan"))
                            # IMPORTANT: Clear the text after speaking it to prevent it from
                            # being repeated if the user doesn't say anything.
                            spoken_text = ""
                        except Exception as tts_e:
                            logging.error(colored(f"TTS failed: {tts_e}", "red"), exc_info=True)
                            spoken_text = "" # Also clear on failure to prevent loops

                    # Listen for the user's next command.
                    user_input, _, wake_word_used = get_listen_microphone()(
                        private_remote_wake_detection=args.private_remote_wake_detection, 
                        remote_voice_enabled=args.online_voice
                    )

            # Handle agent reset request from visual interface
            elif hasattr(g, 'AGENT_RESET_REQUESTED') and g.AGENT_RESET_REQUESTED:
                context_chat = await reset_agent_state(args, context_chat)
                continue  # Skip AI processing and continue with fresh state

            elif g.AGENT_IS_COMPACTING:
                g.AGENT_IS_COMPACTING = False
                context_chat = await construct_engineered_prompt(args.voice, args.gui)
                context_chat.add_message(Role.USER, "Hi can you please check your memory for the latest summary?")
                context_chat.add_message(Role.ASSISTANT, """<think>
The user wants me to search my memory for a the latest summary.
I need to create a todo and call the summary tool with the order key set to latest.
</think><todos>
- [ ] Check my memory
</todos><memory>
<memory>
latest summary LIMIT 1
</memory>""")
            else:
                agent_logger = logging.getLogger("agent_main") if input_event else None
                if agent_logger:
                    agent_logger.info("About to call get_user_input_with_bindings")
                user_input, new_images = await get_user_input_with_bindings(
                    args, context_chat, force_input=user_interrupt, 
                    input_event=input_event, input_lock=input_lock, shared_input=shared_input
                )
                if agent_logger:
                    agent_logger.info(f"Got user input from get_user_input_with_bindings: '{user_input}'")
                if new_images:
                    base64_images.extend(new_images)
                
                if base64_images:
                    transcription_chat = Chat(
                        instruction_message="You are an image transcription assistant. Your only goal is to describe the provided image in detail.",
                        debug_title="Image Transcription"
                    )
                    transcription_chat.add_message(Role.USER, "Describe the following image.")
                    
                    try:
                        transcription = await LlmRouter.generate_completion(
                            transcription_chat,
                            strengths=[AIStrengths.VISION],
                            base64_images=base64_images
                        )
                        user_input = f"{user_input}\n\n{transcription}" if user_input.strip() else transcription
                    except Exception:
                        error_msg = "\n\n[SYSTEM: The attempt to analyze the provided screenshot failed.]"
                        user_input = f"{user_input}{error_msg}"

                    base64_images = []
                
                try:
                    from utils.viewfiles import ViewFiles
                    local_paths = [path for path, _ in PathDetector.extract_paths_from_xml(user_input, "viewfiles")]
                    for path in local_paths:
                        expanded_path = os.path.expanduser(path)
                        if os.path.exists(expanded_path) and (os.path.isfile(expanded_path) or os.path.isdir(expanded_path)):
                            view_result_str = ViewFiles.run(paths=[expanded_path])
                            view_result = json.loads(view_result_str)
                            if view_result.get("result"):
                                user_input += f"\n\n# Content of: {path}\n{view_result['result']}"
                except (ImportError, FileNotFoundError, subprocess.CalledProcessError, Exception):
                    pass
                # Offer intelligent recovery options if we had an interrupt
                if user_interrupt and hasattr(context_chat, 'metadata') and 'last_interrupt' in context_chat.metadata:
                    recovery_choice = offer_intelligent_recovery_options(context_chat)

                    # Handle recovery choices
                    if recovery_choice == 'exit':
                        user_interrupt = False
                        continue  # Skip AI generation, go back to user input
                    elif recovery_choice == 'single':
                        # Force single-model mode for this request
                        if hasattr(g, 'FORCE_SINGLE_MODEL'):
                            g.FORCE_SINGLE_MODEL = True
                        print(colored("  → Switching to single-model mode for better responsiveness", "cyan"))
                    elif recovery_choice == 'retry_diff':
                        # This would be implemented in model selection logic
                        print(colored("  → Will attempt with different models", "cyan"))

                user_interrupt = False

            if LlmRouter.has_unconfirmed_data():
                LlmRouter.confirm_finetuning_data()

            text_stream_painter = TextStreamPainter()

            if (user_input):
                context_chat.add_message(Role.USER, user_input)
                # --- START RE-ADDED HINT/PLAYBOOK LOGIC ---
                try:
                    prompt_subfix = ""
                    # if not code_execution_just_denied:
                    #     user_context = user_input
                    #     if user_context:
                    #         guidance_prompt, score_info = utils_manager.get_relevant_tools_with_scores(user_context, top_k=5)
                    #         if guidance_prompt:
                    #             tool_names = re.findall(r'\*\*(.*?)\*\*', guidance_prompt)
                    #             if tool_names and score_info:
                    #                 score_dict = {item.split("(")[0]: item.split("(")[1].rstrip(")") for item in score_info.split(", ") if "(" in item}
                    #                 logging.info(colored("🔧 Suggested utilities based on your request:", "cyan", attrs=["bold"]))
                    #                 formatted_tools = [f"   - {name} ({score_dict.get(name, '')})" for name in tool_names]
                    #                 logging.info(colored("\n".join(formatted_tools), "light_blue"))
                    #             elif tool_names:
                    #                 logging.info(colored("🔧 Suggested utilities based on your request:", "cyan", attrs=["bold"]))
                    #                 logging.info(colored("\n".join([f"   - {name}" for name in tool_names]), "light_blue"))
                    #         if guidance_prompt:
                    #             prompt_subfix += f"\n\n# RELEVANT TOOLS\n{guidance_prompt}"

                    #         guidance_hints = utils_manager.get_relevant_guidance(user_context, top_k=2)
                    #         if guidance_hints:
                    #             logging.info(colored("💡 Relevant guidance hints:", "cyan", attrs=["bold"]))
                    #             guidance_texts = []
                    #             for hint in guidance_hints:
                    #                 guidance_text = hint.get('guidance_text', '')
                    #                 if guidance_text:
                    #                     logging.info(colored(f"   • {hint.get('keyword', 'general')} ({hint.get('score', 0):.3f}): {guidance_text[:120]}{'...' if len(guidance_text) > 120 else ''}", "light_blue"))
                    #                     guidance_texts.append(guidance_text)
                    #             if guidance_texts:
                    #                 prompt_subfix += "\n\n# GUIDANCE HINTS (EXAMPLES)\n" + "\n\n".join(guidance_texts)

                    #         playbook = playbook_manager.get_relevant_playbook(user_context, threshold=0.7)
                    #         if playbook and playbook.get("thoughts"):
                    #             logging.info(colored("📖 Strategic guidance found:", "magenta", attrs=["bold"]))
                    #             logging.info(colored(f"   • Strategy: {playbook['name']}", "light_magenta"))
                    #             prompt_subfix += "\n\n# STRATEGIC HINTS (SUGGESTED PLAN)\n"
                    #             prompt_subfix += "A relevant strategy was found. Consider this plan:\n" + "".join([f"{i+1}. {thought}\n" for i, thought in enumerate(playbook["thoughts"])])
                    # else:
                    #     logging.info(colored("ℹ️  Skipping tool suggestions and hints (code execution was denied)", "yellow"))

                    if prompt_subfix:
                        context_chat.add_message(Role.USER, prompt_subfix)
                except Exception as e:
                    logging.warning(f"Could not generate hints: {e}")
                # --- END RE-ADDED HINT/PLAYBOOK LOGIC ---

            initializing = False
            last_action_signature: Optional[str] = None
            stall_counter: int = 0
            MAX_STALLS: int = 2
            
            # Agent loop
            while True:
                try:
                    response_branches: List[str] = []
                    try:
                        if assistant_response:
                            context_chat.add_message(Role.ASSISTANT, assistant_response)
                            assistant_response = ""

                        if context_chat:
                            context_chat.save_to_json()

                        UNCOMPRESSED_HISTORY = 4 # Compresses reasoning traces after x messages
                        if (len(context_chat.messages) > UNCOMPRESSED_HISTORY):
                            for i in range(UNCOMPRESSED_HISTORY,UNCOMPRESSED_HISTORY+1): # do prompt engineering to the 7th and 8th messages automatically
                                role = context_chat.messages[-i][0]
                                msg = context_chat.messages[-i][1]
                                if "<think>" in msg:
                                    # Extract content within <think> tags
                                    think_content_match = re.search(r'<think>(.*?)</think>', msg, re.DOTALL)
                                    if think_content_match:
                                        think_content = think_content_match.group(1).strip()
                                        # Find the first line until a newline
                                        first_line_match = re.match(r'^([^\n]*)', think_content)
                                        if first_line_match:
                                            first_line = first_line_match.group(1).strip()
                                            # Replace the original <think> content with only the first line
                                            msg = re.sub(r'<think>(.*?)</think>', f'<think>\n{first_line}\n</think>', msg, flags=re.DOTALL)
                                context_chat.messages[-i] = (role, msg)
                        
                        llm_config = g.get_llm_config()
                        models_for_tasks = []
                        if not args.monte_carlo:
                            # Default mode: single LLM with all available models as fallback
                            # Use g.SELECTED_LLMS to preserve chronological selection order
                            available_models = [
                                key for key in g.SELECTED_LLMS
                                if llm_config.get(key, {}).get('selected') and key not in LlmRouter().failed_models
                            ]
                            if available_models:
                                # Store all models for fallback, but use only one for branching
                                g.DEFAULT_FALLBACK_MODELS = available_models
                                models_for_tasks = [available_models[0]]  # Single model for default mode
                            else:
                                # Initialize default fallback models when none are selected
                                default_fallbacks = ['gemini-2.5-flash', 'gemini-2.5-flash-preview-05-20', 'gemma3n:e2b']
                                g.DEFAULT_FALLBACK_MODELS = default_fallbacks
                                models_for_tasks = [default_fallbacks[0]]
                                logging.debug(colored("🔧 No models configured - using default fallbacks: " + ", ".join(default_fallbacks), "yellow"))
                        else:
                            # Use g.SELECTED_LLMS to preserve chronological selection order for monte_carlo mode too
                            models_for_tasks = [
                                key for key in g.SELECTED_LLMS
                                if (llm_config.get(key, {}).get('selected') and 
                                    key not in LlmRouter().failed_models and 
                                    llm_config.get(key, {}).get('beams', 0) > 0)
                                for _ in range(llm_config.get(key, {}).get('beams', 0))
                            ]
                            if not models_for_tasks:
                                branch_fallback_iterator.update_fallback_models(llm_config)
                                models_for_tasks = branch_fallback_iterator.get_multiple_fallbacks(count=3) or [key for key in g.SELECTED_LLMS if llm_config.get(key, {}).get('selected') and key not in LlmRouter().failed_models]
                                if not models_for_tasks:
                                    logging.error(colored("❌ No available models to process the request. Check LLM selection.", "red"))
                                    break
                        
                        num_branches = len(models_for_tasks)
                        g.MCT = num_branches
                        temperature = 0.85 if num_branches > 1 else 0

                        if num_branches > 1:
                            print_lock = asyncio.Lock()
                            branch_model_map = {}

                            # Use TaskGroup for better cancellation support in Monte Carlo mode
                            try:
                                # Try to use Python 3.11+ TaskGroup for robust branch cancellation
                                if sys.version_info >= (3, 11):
                                    from asyncio import TaskGroup
                                    async with TaskGroup() as tg:
                                        tasks = [
                                            tg.create_task(
                                                generate_branch(model_key, i, print_lock, context_chat, text_stream_painter,
                                                             temperature, base64_images, llm_config, branch_model_map, num_branches)
                                            )
                                            for i, model_key in enumerate(models_for_tasks)
                                        ]
                                    branch_results = [task.result() for task in tasks if not task.cancelled()]
                                else:
                                    # Fallback to asyncio.gather for older Python versions
                                    tasks = [generate_branch(model_key, i, print_lock, context_chat, text_stream_painter, temperature, base64_images, llm_config, branch_model_map, num_branches) for i, model_key in enumerate(models_for_tasks)]
                                    branch_results = await asyncio.gather(*tasks)

                                response_branches = [res for res in branch_results if res and res.strip()]

                            except asyncio.CancelledError:
                                print(colored("  → Monte Carlo branches cancelled", "yellow"))
                                response_branches = []
                                user_interrupt = True
                                break
                        else:
                            response_buffer_list = [""]
                            interruption_callback = create_interruption_callback(response_buffer_list, text_stream_painter)
                            current_todo = f"<current_todo>\n{todos().get_next_unchecked_task()}\n</current_todo>"
                            # for assistant_prefix in ["<", current_todo, "<think", "<thinking>"]:
                            for tool_prefix in ["<think>", "<todos>", "<"]:
                                if ("<think>" != tool_prefix):
                                    print(colored(f"Model did not generate text, attempting generation with prefix: {tool_prefix}", "yellow"))
                                # Force thinking after context
                                # if ("" == tool_prefix and context_chat.messages[-1][1].endswith("</context>")):
                                #     tool_prefix = "<think>"
                                
                                # In default mode, use all available models for fallback
                                fallback_models = getattr(g, 'DEFAULT_FALLBACK_MODELS', models_for_tasks)
                                l_context_chat = context_chat.deep_copy()
                                l_temperature = context_chat.deep_copy()
                                generated_text = ""  # Initialize to prevent UnboundLocalError
                                validated_text = ""  # Initialize to prevent UnboundLocalError
                                try:
                                    for i in range(10):
                                        await LlmRouter.generate_completion(
                                            chat=l_context_chat,
                                            preferred_models=fallback_models,
                                            force_preferred_model=True,
                                            temperature=temperature,
                                            base64_images=base64_images,
                                            generation_stream_callback=interruption_callback,
                                            assistant_prefix=tool_prefix
                                        )
                                        generated_text = response_buffer_list[0]
                                        # If response did not end with '>' then it should be forcibly kept generating
                                        if generated_text.endswith(">"):
                                            validated_text = generated_text
                                            break
                                        else:
                                            l_context_chat.add_message(Role.ASSISTANT, generated_text[:-i]) # Remove last few chars incrementatlly to force model to be regenerative
                                except StreamInterruptedException as e: # This exception is called by the tool delimiting mechanism at the end of any xml block
                                    validated_text = e.response
                                
                                if (validated_text and validated_text != tool_prefix):
                                    response_branches.append(validated_text)
                                    break
                        
                        base64_images = []

                    except UserInterruptedException:
                        # User interrupted model generation with Ctrl+C - handle gracefully with state preservation
                        print(colored("\n⚠️  Model generation interrupted by user (Ctrl+C)", "yellow", attrs=['bold']))

                        # Immediate chat state preservation
                        try:
                            context_chat.save_to_json()
                            print(colored("  → Chat context saved successfully", "green"))
                        except Exception as e:
                            print(colored(f"  ⚠️  Warning: Could not save chat context: {e}", "yellow"))

                        # Store interrupt metadata for intelligent recovery
                        if hasattr(context_chat, 'metadata'):
                            import time
                            context_chat.metadata['last_interrupt'] = {
                                'timestamp': time.time(),
                                'branch_count': len(response_branches) if 'response_branches' in locals() else 0,
                                'had_partial_response': len(response_branches) > 0 if 'response_branches' in locals() else False,
                                'models_attempted': list(branch_model_map.keys()) if 'branch_model_map' in locals() else [],
                                'monte_carlo_mode': num_branches > 1 if 'num_branches' in locals() else False
                            }

                        user_interrupt = True
                        break
                    except Exception as e:
                        print(colored(f"An Exception was raised, e: {e}", "red"))
                        traceback.print_exc()
                        break

                    if response_branches:
                        if len(response_branches) > 1:
                            selected_branch_index, _ = await select_best_branch(context_chat, response_branches, branch_model_map)
                            assistant_response = response_branches[selected_branch_index]
                        else:
                            assistant_response = response_branches[0]
                    else:
                        break
                    
                    if last_action_signature and assistant_response == last_action_signature:
                        stall_counter += 1
                    else:
                        stall_counter = 0
                    last_action_signature = assistant_response

                    if stall_counter >= MAX_STALLS:
                        print(colored("The agent is spamming its last tool use. Aborting...", "red"))
                        stall_counter, last_action_signature = 0, None
                        break

                    context_chat.add_message(Role.ASSISTANT, assistant_response)

                    # Note: AI response to visual interface will be sent after tool execution
                    # to properly detect if visualresponse was used

                    # Display the response in summary mode (since streaming doesn't show it)
                    if g.SUMMARY_MODE and assistant_response.strip():
                        print(assistant_response)

                    # --- XML VALIDATION ---
                    # Check for malformed XML before attempting tool execution
                    xml_errors = detect_malformed_xml(assistant_response)
                    if xml_errors:
                        # Generate context error message and add to chat
                        error_context = generate_xml_error_context(xml_errors)
                        context_chat.add_message(Role.ASSISTANT, error_context)
                        assistant_response = ""
                        print(colored("⚠️  XML syntax errors detected and reported to model", "yellow"))
                        continue  # Skip tool execution and get corrected response

                    # --- NEW: Generic Tool Execution Loop ---
                    # Find all unique tool tags present in the assistant's response
                    tool_classes = getattr(g, 'tool_classes', {})
                    tool_tags_found = []
                    for tool_name, tool_class in tool_classes.items():
                        # Use a more robust regex to find any instance of the tool tag
                        if re.search(f'<{tool_class.get_delim()}[> ]', assistant_response, re.DOTALL):
                            tool_tags_found.append(tool_class)

                    # If any tool tags are found, process them
                    if tool_tags_found:
                        
                        # --- Enhanced Path-Aware Permission Checking ---
                        # Check permissions using path detection for tools with file operations
                        should_execute = False
                        if len(tool_tags_found) == 1 and any(k in (tool_tags_found[0].__module__ or "") for k in ["todos", "think", "searchweb", "grepsearch"]):
                            should_execute = True
                        elif len(tool_tags_found) > 0:
                            # Analyze each tool for path-based permissions
                            permission_needed = False
                            tool_analyses = []
                            
                            for tool in tool_tags_found:
                                delim = tool.get_delim()
                                blocks = get_extract_blocks()(assistant_response, [delim])
                                
                                for block in blocks:
                                    analysis = PathDetector.analyze_tool_block(delim, block)
                                    tool_analyses.append(analysis)
                                    
                                    if analysis['needs_permission']:
                                        permission_needed = True
                            
                            if not permission_needed:
                                # No permission needed for safe tools
                                should_execute = True
                            else:
                                # Check permissions for each tool that needs it
                                should_execute = True  # Start optimistic
                                
                                for analysis in tool_analyses:
                                    if not analysis['needs_permission']:
                                        continue
                                        
                                    # Use enhanced permission system for path-aware tools
                                    permission_granted = await enhanced_permission_prompt(
                                        args,
                                        tool_name=analysis['tool_name'],
                                        action_description=analysis['action_description'],
                                        file_path=analysis['primary_path'],
                                        command_hash=analysis['command_hash'],
                                        command_pattern=analysis['command_pattern'],
                                        has_paths=analysis['has_paths'],
                                        xml_content=analysis.get('xml_content', ''),
                                        input_event=input_event,
                                        input_lock=input_lock,
                                        shared_input=shared_input,
                                        get_user_input_func=get_user_input_with_bindings
                                    )
                                    
                                    if not permission_granted:
                                        logging.info(colored("⛔ Code execution denied", "green"))

                                        should_execute = False
                                        break
                                    logging.info(colored("✅ Code execution permitted", "green"))

                        if should_execute:
                            try:
                                # Reset tool context for clean execution
                                tool_context.reset()
                                tool_results = []
                                executed_tools_info = []  # Track tool execution info for visual interface
                                for tool_class in tool_tags_found:
                                    tool_name = tool_class.get_delim()
                                    blocks = get_extract_blocks()(assistant_response, [tool_name])
                                    
                                    if not blocks:
                                        continue

                                    # --- FIX: Execute only the first block for each tool to prevent duplicate execution ---
                                    first_block = blocks[0]
                                    if len(blocks) > 1:
                                        logging.warning(colored(f"⚠️  Duplicate '{tool_name}' blocks detected. Only the first was executed.", "yellow"))
                                    
                                    # Track tool execution timing and status
                                    import time
                                    tool_start_time = time.time()
                                    tool_status = "success"
                                    tool_error = None

                                    try:
                                        # Send live tool start notification to loading area header (skip visualresponse)
                                        if hasattr(g, 'GUI_MODE') and g.GUI_MODE and tool_name != 'visualresponse':
                                            try:
                                                from tools.visualresponse import visualresponse
                                                session = visualresponse._get_session()
                                                if session and session.is_healthy():
                                                    live_payload = {
                                                        "id": f"loading-tools-{int(time.time() * 1000000)}",
                                                        "action": "loading_tool_start",
                                                        "tool_name": tool_name,
                                                        "tool_args": first_block.strip() if first_block else "",
                                                        "timestamp": time.time()
                                                    }
                                                    session.send_update(live_payload)
                                                    logging.debug(f"Sent loading tool start notification for {tool_name}")
                                            except Exception as e:
                                                logging.debug(f"Failed to send loading tool start notification: {e}")

                                        # All tools now share the same simple run signature.
                                        # Suppress runtime warnings from tools (gRPC/SSL/TF) unless debug mode
                                        with suppress_warnings_runtime():
                                            tool_result = tool_class.run(first_block)

                                        # Handle both sync and async tools with proper error handling
                                        try:
                                            if asyncio.iscoroutine(tool_result):
                                                tool_result = await tool_result
                                        except Exception as e:
                                            tool_status = "error"
                                            tool_error = str(e)
                                            logging.error(colored(f"Error awaiting tool {tool_name}: {e}", "red"))
                                            tool_result = f"Error executing {tool_name}: {str(e)}"

                                        # Ensure tool_result is always a string, never a coroutine
                                        if asyncio.iscoroutine(tool_result):
                                            tool_status = "error"
                                            tool_error = "returned unresolved coroutine"
                                            logging.error(colored(f"Tool {tool_name} still returned coroutine after await - converting to error", "red"))
                                            tool_result = f"Error: Tool {tool_name} returned unresolved coroutine"

                                    except Exception as e:
                                        tool_status = "error"
                                        tool_error = str(e)
                                        logging.error(colored(f"Error executing tool {tool_name}: {e}", "red"))
                                        tool_result = f"Error executing {tool_name}: {str(e)}"

                                    # Check if tool result contains error (for tools that return error JSON instead of raising exceptions)
                                    if tool_status == "success" and tool_result:
                                        try:
                                            # Try to parse result as JSON to check for error key
                                            import json
                                            result_json = json.loads(tool_result)
                                            if isinstance(result_json, dict) and "error" in result_json:
                                                tool_status = "error"
                                                tool_error = result_json["error"]
                                                logging.debug(f"Tool {tool_name} returned error in result: {tool_error}")
                                        except (json.JSONDecodeError, TypeError):
                                            # If not valid JSON, keep original status
                                            pass

                                    # Calculate execution time
                                    tool_duration_ms = int((time.time() - tool_start_time) * 1000)

                                    # Send live tool completion notification to loading area header (skip visualresponse)
                                    if hasattr(g, 'GUI_MODE') and g.GUI_MODE and tool_name != 'visualresponse':
                                        try:
                                            from tools.visualresponse import visualresponse
                                            session = visualresponse._get_session()
                                            if session and session.is_healthy():
                                                live_payload = {
                                                    "id": f"loading-tools-{int(time.time() * 1000000)}",
                                                    "action": "loading_tool_complete",
                                                    "tool_name": tool_name,
                                                    "tool_status": tool_status,
                                                    "tool_duration_ms": tool_duration_ms,
                                                    "tool_error": tool_error,
                                                    "tool_result": tool_result[:500] if tool_result else "",  # Truncate for header display
                                                    "timestamp": time.time()
                                                }
                                                session.send_update(live_payload)
                                                logging.debug(f"Sent loading tool completion notification for {tool_name}")
                                        except Exception as e:
                                            logging.debug(f"Failed to send loading tool completion notification: {e}")

                                    # Store tool execution info for visual interface
                                    executed_tools_info.append({
                                        "name": tool_name,
                                        "status": tool_status,
                                        "duration_ms": tool_duration_ms,
                                        "error": tool_error,
                                        "args": first_block.strip() if first_block else "",
                                        "context_returned": tool_result.strip() if tool_result else "",
                                        "context_length": len(str(tool_result)) if tool_result else 0
                                    })

                                    # Wrap tools returned values as tool_ouput
                                    if tool_name and tool_name not in ["todos"] and tool_result:
                                        tool_results.append(tool_result)
                                
                                returned_context = ""
                                for context in tool_results:
                                    context = re.sub(r'\x1b\[[0-9;]*m', '', context) #  remove ANSI escape codes (also known as terminal escape sequences)
                                    if len(context) > readfile.CHUNK_LIMIT:
                                        context = context[:readfile.CHUNK_LIMIT//4] + "\n...output truncated...\n" + context[-readfile.CHUNK_LIMIT//4:]
                                    returned_context += f"<context>\n{context}\n</context>"

                                # Store tool execution info in global context for visual response
                                if executed_tools_info:
                                    g.last_executed_tools_info = executed_tools_info
                                    logging.debug(f"Stored execution info for {len(executed_tools_info)} tools")

                                # Note: Automatic AI responses removed - only explicit visualresponse tool calls appear in interface

                                if (len(returned_context) > 0):
                                    context_chat.add_message(Role.ASSISTANT, returned_context)

                                    # Check for images from tool context and add them to the chat
                                    if tool_context.has_images():
                                        context_chat.base64_images.extend(tool_context.get_images())
                                        logging.info(colored(f"📸 Added {len(tool_context.get_images())} image(s) to chat context for vision analysis", "blue"))
                                        tool_context.clear_images()  # Clear images after adding them
                                    
                                    assistant_response = ""
                                    if ("response" in tool_name and "error" not in returned_context.lower()):
                                        # Break and wait for user input instead of continuing the loop
                                        break
                                    else:
                                        continue
                                else:
                                    # No meaningful context was added (only invisible tools like todos/think were executed)
                                    # Break and wait for user input instead of continuing the loop
                                    break
                            except Exception:
                                error_output = f"<context>\n{traceback.format_exc()}\n</context>"
                                context_chat.add_message(Role.ASSISTANT, error_output)
                                assistant_response = ""
                                continue # Continue the agent loop for the next action
                        else:
                            cancellation_notice = "<context>\nCode execution rejected by user or guard.\n</context>"
                            context_chat.add_message(Role.ASSISTANT, cancellation_notice)
                            assistant_response = ""
                            continue # Continue the agent loop for the next action
                    
                    else:
                        # No tools or code found, this is a conversational response.
                        break # Ask for user input
                except KeyboardInterrupt:
                    user_interrupt = True
                    break # Ask for user input
                except Exception as e:
                    if "ctrl+c" not in str(e).lower():
                        # Get detailed traceback information
                        tb = traceback.extract_tb(e.__traceback__)
                        
                        # Enhanced error message with context
                        error_msg = f"An unexpected error occurred in the agent loop: {e}"
                        
                        # Add specific information for NoneType iteration errors
                        if "argument of type 'NoneType' is not iterable" in str(e):
                            error_msg += "\n🔍 NoneType Iteration Error Details:"
                            
                            # Find the exact line that caused the error
                            for frame in tb:
                                filename = frame.filename
                                line_number = frame.lineno
                                function_name = frame.name
                                line_text = frame.line or linecache.getline(filename, line_number).strip()
                                
                                if any(op in line_text for op in [' in ', '.startswith(', '.endswith(', ' not in ']):
                                    error_msg += f"\n  📁 File: {filename}"
                                    error_msg += f"\n  📍 Line {line_number} in function '{function_name}'"
                                    error_msg += f"\n  💥 Code: {line_text}"
                                    
                                    # Try to identify which variable is None
                                    if ' in ' in line_text:
                                        parts = line_text.split(' in ')
                                        if len(parts) >= 2:
                                            suspect_var = parts[1].split()[0].strip('()[]{}')
                                            error_msg += f"\n  🎯 Suspected None variable: {suspect_var}"
                                    elif '.startswith(' in line_text or '.endswith(' in line_text:
                                        var_name = line_text.split('.')[0].strip()
                                        error_msg += f"\n  🎯 Suspected None variable: {var_name}"
                                    break
                        
                        # Always show the full stack trace in debug mode or for critical errors
                        if args.debug or "NoneType" in str(e):
                            error_msg += "\n\n📋 Full Stack Trace:"
                            for i, frame in enumerate(tb):
                                error_msg += f"\n  {i+1}. {frame.filename}:{frame.lineno} in {frame.name}()"
                                if frame.line:
                                    error_msg += f"\n     → {frame.line.strip()}"
                        
                        logging.critical(colored(error_msg, "red"))
                        
                        # Also log with exc_info for complete traceback in debug mode
                        if args.debug:
                            logging.critical("Complete exception details:", exc_info=True)
                    user_interrupt = True
                    break # Ask for user input
            
            else:
                print("# cli-agent: No conversation to compact.")

            if args.exit and not args.message:
                # Check if there are active visual response servers before exiting
                try:
                    from tools.visualresponse import visualresponse
                    if (hasattr(visualresponse, '_session') and 
                        visualresponse._session and 
                        visualresponse._session.process and 
                        visualresponse._session.process.is_alive()):
                        
                        # Visual server is active - wait for user to close it
                        server_url = f"http://127.0.0.1:{visualresponse._session.port}"
                        print(colored(f"\n🌐 Visual response server is running at {server_url}", "cyan"))
                        print(colored("Press Ctrl+C to exit when you're done viewing the content.", "yellow"))
                        
                        try:
                            # Keep the process alive until user interrupts
                            while (visualresponse._session.process and 
                                   visualresponse._session.process.is_alive()):
                                import time
                                time.sleep(1)
                        except KeyboardInterrupt:
                            print(colored("\n🛑 Shutting down visual server...", "yellow"))
                            visualresponse.close_session()
                            
                except ImportError:
                    pass  # Visual response tool not available
                    
                exit(0)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.warning(colored("CLI-Agent was interrupted. Shutting down gracefully...", "yellow") )
        _perform_graceful_shutdown()
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            logging.critical(colored(f"CLI-Agent encountered a fatal error: {e}", "red"), exc_info=True)
        _perform_graceful_shutdown()


def _perform_graceful_shutdown():
    """Perform graceful shutdown including log cleanup."""
    try:
        from shared.utils.log_manager import get_log_manager
        log_manager = get_log_manager()
        shutdown_stats = log_manager.shutdown()
        
        if shutdown_stats['log_cleanup']['files_removed'] > 0 or shutdown_stats['audio_cleanup']['files_removed'] > 0:
            total_files = shutdown_stats['log_cleanup']['files_removed'] + shutdown_stats['audio_cleanup']['files_removed']
            total_bytes = shutdown_stats['log_cleanup']['bytes_freed'] + shutdown_stats['audio_cleanup']['bytes_freed']
            logging.info(colored(f"🧹 Final cleanup: {total_files} files removed, {total_bytes / 1024:.1f} KB freed", "cyan"))
            
    except ImportError:
        pass  # Log manager not available
    except Exception as e:
        logging.debug(f"Error during graceful shutdown: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow") )
        _perform_graceful_shutdown()
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            print(colored(f"\nCLI-Agent encountered a fatal error during startup: {e}", "red"))
            traceback.print_exc()
        _perform_graceful_shutdown()