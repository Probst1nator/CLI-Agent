# main.py
"""
This script serves as the main entry point for the CLI-Agent. It orchestrates
the primary execution loop, handling user interaction, multi-LLM reasoning via a
Monte Carlo Action Engine, and tool invocation within a computational notebook.
"""
import time
start_time = time.time() # Global timer for startup and operations

import logging
import re
from termcolor import colored
from typing import Dict, List, Optional, Tuple, Callable, Any
import sys
import argparse
import os
import math
import asyncio
import threading
from collections import Counter
import importlib
import json
from pathlib import Path
from pyfiglet import figlet_format

# --- Project Path Setup ---
# Ensure all project modules can be imported correctly.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Tool and Core Imports ---
from tools.primitives.todos import todos
from tools.readfile import readfile
from shared.utils_audio import text_to_speech
from core.globals import g
from core.llm_router import LlmRouter, StreamInterruptedException
from core.chat import Chat, Role
from agent.playbook.playbook_manager import PlaybookManager
from agent.utils_manager.utils_manager import UtilsManager
from utils.tool_context import tool_context
from core.permissions.enhanced_permissions import enhanced_permission_prompt
from shared.common_utils import extract_blocks

# --- Suppress Noisy Library Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")

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

    parser.add_argument("--full_output", dest='full_output_mode', action="store_true", default=True,
                        help="Full output mode. Shows live model output instead of hiding it.")
    parser.add_argument("-f", "--fast", action="store_true", default=False,
                        help="Fast mode (deprecated - kept for backward compatibility).")
    parser.add_argument("-img", "--image", action="store_true", default=False,
                        help="Take a screenshot using Spectacle for region selection (with automatic fallbacks if not available).")
    parser.add_argument("-sbx", "--sandbox", action="store_true", default=False,
                        help="Use weakly sandboxed python execution. Sets g.USE_SANDBOX=True.")
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

class ConsoleFormatter(logging.Formatter):
    """A logging formatter that adds a colored elapsed time prefix for the console."""
    def __init__(self, fmt, **kwargs):
        super().__init__(fmt, **kwargs)
    
    def format(self, record):
        elapsed = time.time() - start_time
        # Create a prefix with the elapsed time
        prefix = colored(f"[{elapsed:6.2f}s] ", 'dark_grey')
        
        # The original message might already be colored
        formatted_message = super().format(record)
        
        # For debug messages, color the whole line for visibility
        if record.levelno == logging.DEBUG:
            return prefix + colored(formatted_message, 'light_blue')
        
        return prefix + formatted_message

class SimpleFormatter(logging.Formatter):
    """A simple formatter that just returns the message, used after startup or in TUI mode."""
    def format(self, record):
        return super().format(record)

class FileFormatter(logging.Formatter):
    """A logging formatter for files, including elapsed time and stripping ANSI color codes."""
    def __init__(self, fmt, **kwargs):
        super().__init__(fmt, **kwargs)

    def format(self, record):
        elapsed = time.time() - start_time
        record.elapsed = f"[{elapsed:6.2f}s]"
        
        # The original message might contain color codes, strip them for the file log.
        if isinstance(record.msg, str):
            record.msg = re.sub(r'\x1b\[[0-9;]*m', '', record.msg)
            
        return super().format(record)

def setup_logging(debug_mode: bool, log_file: Optional[str] = None, tui_mode: bool = False):
    """Configure the root logger for console and optional file output."""
    logger = logging.getLogger() # Get root logger
    logger.setLevel(logging.DEBUG) # Capture all messages at the root

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_log_level = logging.DEBUG if debug_mode else logging.INFO
    console_handler.setLevel(console_log_level)
    
    # Use a simple formatter in TUI mode to avoid cluttering the UI with timestamps
    if tui_mode:
        console_formatter = SimpleFormatter("%(message)s")
    else:
        console_formatter = ConsoleFormatter("%(message)s")

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler (optional) ---
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG) # Always log everything to the file
        file_formatter = FileFormatter("%(elapsed)s [%(levelname)-8s] %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logging.info(colored(f"Logging to file: {log_file}", "magenta")) # Use standard logging so it gets timed

    # --- FIX: Silence Noisy Loggers (Always, even in debug mode) ---
    # These libraries produce networking/infrastructure logs we don't need
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("py_classes.ai_providers.cls_ollama_interface").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

def swap_to_simple_logging():
    """Swaps the console handler's formatter to a simpler one without the timer."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            # We found the console handler, now swap its formatter
            simple_formatter = SimpleFormatter("%(message)s")
            handler.setFormatter(simple_formatter)
            break

# --- End Custom Logging Setup ---


def colorize_model_name(model_name: str, tui_mode: bool = False) -> str:
    """Applies color to a model name based on its provider."""
    # Define colors using Textual's markup format
    tui_colors = {
        'local': 'green',
        'google': 'blue',
        'anthropic': 'yellow',
        'openai': 'magenta',
        'groq': 'red',
        'default': 'cyan'
    }
    
    # Original termcolor colors
    cli_colors = {
        'local': 'green',
        'google': 'blue',
        'anthropic': 'yellow',
        'openai': 'magenta',
        'groq': 'red',
        'default': 'cyan'
    }

    if ':' in model_name:
        color_key = 'local'
    else:
        provider = model_name.split('-')[0].lower()
        if 'gemini' in model_name or 'google' in provider: color_key = 'google'
        elif 'claude' in provider or 'anthropic' in provider: color_key = 'anthropic'
        elif 'openai' in provider or 'gpt' in model_name: color_key = 'openai'
        elif 'groq' in provider: color_key = 'groq'
        else: color_key = 'default'

    if tui_mode:
        color = tui_colors[color_key]
        return f"[{color}]{model_name}[/{color}]"
    else:
        color = cli_colors[color_key]
        return colored(model_name, color)

def colorize_model_list(model_list: List[str], tui_mode: bool = False) -> str:
    """Colorizes a list of model names and joins them into a string."""
    return ', '.join(colorize_model_name(m, tui_mode) for m in model_list)




def print_startup_summary(args: argparse.Namespace, tui_mode: bool = False):
    """Prints a comprehensive summary of the current agent configuration using the logger."""
    
    # --- Start Helper Function for Coloring ---
    def get_count_color(count: int) -> str:
        """Returns a color string based on the model count for at-a-glance status."""
        if count == 1: return 'red'
        if count == 2: return 'yellow'
        if count == 3: return 'green'
        if count > 3: return 'light_green'
        return 'white' # Default fallback
    # --- End Helper Function ---

    # Helper for status strings
    def get_status_str(state, on_text='ON', off_text='OFF'):
        if tui_mode:
            return f"[bold green]{on_text}[/bold green]" if state else f"[bold red]{off_text}[/bold red]"
        else:
            return colored(on_text, 'green', attrs=['bold']) if state else colored(off_text, 'red')

    # Column widths for alignment
    CMD_WIDTH = 24
    DESC_WIDTH = 28
    
    # Use logger.info to ensure timed prefix is applied
    from core.globals import g # Lazy import to avoid circular dependency at top level
    
    header = "[yellow]--- Agent Configuration ---[/yellow]" if tui_mode else colored("--- Agent Configuration ---", "yellow")
    logging.info(header)
    
    # Get detailed LLM configuration
    llm_config = g.get_llm_config()
    # Use g.SELECTED_LLMS to preserve user selection order instead of iterating through config
    selected_models = g.SELECTED_LLMS if g.SELECTED_LLMS else [key for key, data in llm_config.items() if data.get('selected')]
    
    # Categorize models by their roles, preserving selection order
    branch_models = [key for key in selected_models if llm_config.get(key, {}).get('beams', 0) > 0]
    eval_models = [key for key in selected_models if llm_config.get(key, {}).get('eval', 0) > 0]
    guard_models = [key for key in selected_models if llm_config.get(key, {}).get('guard', 0) > 0]
    
    # Calculate fallback models (selected but not specialized)
    specialized_models = set(branch_models + eval_models + guard_models)
    fallback_models = [model for model in selected_models if model not in specialized_models]
    
    # Count totals for quick reference
    guard_count = len(guard_models)
    
    # Auto Execution status
    if args.auto:
        if guard_count > 0:
            auto_status = f'ON ({guard_count} guards)'
        else:
            auto_status = 'ON (manual confirmation)'
    else:
        auto_status = 'OFF'
    
    # LLM Selection status - show local vs cloud breakdown when available
    if selected_models:
        unique_models = list(set(selected_models))
        local_count = len([m for m in unique_models if ':' in m])
        cloud_count = len(unique_models) - local_count
        
        status_parts = []
        if cloud_count > 0:
            if tui_mode:
                status_parts.append(f"[blue]{cloud_count} cloud[/blue]")
            else:
                status_parts.append(f"{colored(str(cloud_count), 'blue')} {colored('cloud', 'blue')}")
        if local_count > 0:
            if tui_mode:
                status_parts.append(f"[bright_green]{local_count} local[/bright_green]")
            else:
                status_parts.append(f"{colored(str(local_count), 'light_green')} {colored('local', 'light_green')}")
            
        if status_parts:
            llm_status = ", ".join(status_parts)
            if len(unique_models) > (local_count + cloud_count): # Failsafe for logic errors
                 llm_status += f" ({len(unique_models)} total)"
        else:
            if tui_mode:
                llm_status = "[red]No models selected[/red]"
            else:
                llm_status = colored('No models selected', 'red')
    else:
        if tui_mode:
            llm_status = "[yellow]Default[/yellow]"
        else:
            llm_status = colored('Default', 'yellow')
    
    # Main configuration display
    if tui_mode:
        config_lines = [
            f"  [white]{'Local Mode (-l)'.ljust(CMD_WIDTH)}[/white] [bright_black]{'Use only local LLMs'.ljust(DESC_WIDTH)}[/bright_black] (Status: {get_status_str(args.local)})",
            f"  [white]{'Auto Execution (-a)'.ljust(CMD_WIDTH)}[/white] [bright_black]{'Automatic code execution'.ljust(DESC_WIDTH)}[/bright_black] (Status: {get_status_str(args.auto, on_text=auto_status, off_text='OFF')})",
            f"  [white]{'LLM Selection (--llm)'.ljust(CMD_WIDTH)}[/white] [bright_black]{'Specific LLM(s) to use'.ljust(DESC_WIDTH)}[/bright_black] (Status: {llm_status})",
            f"  [white]{'Voice Mode (-v)'.ljust(CMD_WIDTH)}[/white] [bright_black]{'Voice input/output'.ljust(DESC_WIDTH)}[/bright_black] (Status: {get_status_str(args.voice)})"
        ]
    else:
        config_lines = [
            f"  {colored('Local Mode (-l)'.ljust(CMD_WIDTH), 'white')} {colored('Use only local LLMs'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.local)})",
            f"  {colored('Auto Execution (-a)'.ljust(CMD_WIDTH), 'white')} {colored('Automatic code execution'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.auto, on_text=auto_status, off_text='OFF')})",
            f"  {colored('LLM Selection (--llm)'.ljust(CMD_WIDTH), 'white')} {colored('Specific LLM(s) to use'.ljust(DESC_WIDTH), 'light_grey')} (Status: {llm_status})",
            f"  {colored('Voice Mode (-v)'.ljust(CMD_WIDTH), 'white')} {colored('Voice input/output'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.voice)})"
        ]
    
    for line in config_lines:
        logging.info(line)
    
    # --- MODEL CONFIGURATION DISPLAY ---
    if selected_models:
        mcae_header = "[yellow]--- Monte Carlo Action Engine ---[/yellow]" if tui_mode else colored("--- Monte Carlo Action Engine ---", "yellow")
        logging.info(mcae_header)

        if branch_models:
            count = 0
            # Correctly count branches based on 'beams'
            for key in branch_models:
                count += llm_config.get(key, {}).get('beams', 1)
            color = get_count_color(count)
            model_list_str = colorize_model_list(branch_models, tui_mode)
            if tui_mode:
                line = f"  🌿 [{color}]{count} Branch Generators:[/{color}] {model_list_str}"
            else:
                line = colored(f"  🌿 {count} Branch Generators:".ljust(27), color) + " " + model_list_str
            logging.info(line)
        
        if eval_models:
            count = len(eval_models)
            color = get_count_color(count)
            model_list_str = colorize_model_list(eval_models, tui_mode)
            if tui_mode:
                line = f"  ⚖️  [{color}]{count} Branch Evaluators:[/{color}] {model_list_str}"
            else:
                line = colored(f"  ⚖️  {count} Branch Evaluators:".ljust(28), color) + " " + model_list_str
            logging.info(line)
        
        if guard_models:
            count = len(guard_models)
            color = get_count_color(count)
            model_list_str = colorize_model_list(guard_models, tui_mode)
            if tui_mode:
                line = f"  🛡️  [{color}]{count} Execution Guards:[/{color}] {model_list_str}"
            else:
                line = colored(f"  🛡️  {count} Execution Guards:".ljust(28), color) + " " + model_list_str
            logging.info(line)
        
        # Show fallback models in Monte Carlo mode
        count = len(fallback_models)
        color = "light_green" if count > 0 else "red"
        model_list_str = colorize_model_list(fallback_models, tui_mode)
        if tui_mode:
            line = f"  🔄 [{color}]{count} Fallback Models:[/{color}] {model_list_str}"
        else:
            line = colored(f"  🔄 {count} Fallback Models:".ljust(27), color) + " " + model_list_str
        logging.info(line)
    
    command_hint = "[grey37]Type '-h' for a full list of commands.[/grey37]" if tui_mode else colored("Type '-h' for a full list of commands.", "dark_grey")
    logging.info(command_hint)

# --- Eager setup of args ---
# This is done before heavy imports to allow --help to work quickly.
args = parse_cli_args()


# Import figlet for later use
from pathlib import Path


# --- Main Application Imports (Lazy/Normal) ---
# Imports are moved below arg parsing and logging setup to allow for faster startup
# and to ensure logging is configured before any modules start logging.
import datetime
import traceback
from dotenv import load_dotenv
import pyperclip
import socket
import warnings
import base64
import tempfile
import subprocess

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")
warnings.filterwarnings("ignore", message="Unrecognized FinishReason enum value*", module="proto.marshal.rules.enums", category=UserWarning)

from agent.notebook import ComputationalNotebook
from core.ai_strengths import AIStrengths
from shared.utils.web import WebServer
from core.permissions.path_detector import PathDetector
from agent.text_painter import TextStreamPainter



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Lazy load functions from shared audio utils to avoid heavy imports at startup
def get_listen_microphone():
    from shared.audio.audio_utils import listen_microphone  # noqa: E402
    return listen_microphone

try:
    from utils.tts import TtsUtil
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

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logging.warning(f"Could not determine local IP: {e}")
        return None

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
        # Get all discovered models
        for provider_models in router.discovered_models.values():
            available_models.extend(provider_models.keys())
    except:
        # Fallback to hardcoded reliable models
        available_models = ['gemini-2.5-flash', 'gpt-4o-mini', 'llama3.2:3b']
    
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
    tool_delimiters = list(getattr(g, 'tool_classes', {}).keys())
    tool_tag_pattern = "|".join(re.escape(d) for d in tool_delimiters)
    XML_TOOL_BLOCK_REGEX = re.compile(f"<({tool_tag_pattern})[^>]*>.*?</\\1>", re.DOTALL) if tool_tag_pattern else None
    
    async def _callback(chunk: str):
        # Process the chunk character by character to ensure proper interruption
        for char in chunk:
            # 1. Print the incoming character
            if not g.SUMMARY_MODE or g.DEBUG_MODE:
                if lock:
                    async with lock:
                        print(painter.apply_color(char), end="", flush=True)
                else:
                    print(painter.apply_color(char), end="", flush=True)

            # 2. Append the character to the shared buffer
            response_buffer[0] += char
            current_buffer = response_buffer[0]

            # 3. Check for interruption conditions only when necessary
            should_check = char in '`><'
            
            if should_check:
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
                    
                # CRITICAL: Also interrupt if we detect hallucinated execution output
                if "<context>" in current_buffer or "</context>" in current_buffer:
                    # logging.warning(colored("🚨 INTERRUPTING: Detected hallucinated context tag, interrupting, splicing and continuing", "red"))
                    # Find the content up to the context tag
                    output_pos = current_buffer.find("<context>")
                    if (output_pos == -1):
                        output_pos = current_buffer.find("</context>")
                    content_before_output = current_buffer[:output_pos].strip()

                    # ! Workaround for bad llms
                    content_before_output = content_before_output.replace("```tool_code", "<python>")
                    content_before_output = content_before_output.replace("<tool_code>", "<python>")
                    content_before_output = content_before_output.replace("</tool_code>", "</python>")
                    raise StreamInterruptedException(content_before_output)

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
            llm_obj = llm_router.get_llm(model_key)
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

def is_response_complete(response: str) -> bool:
    """
    Check if an interrupted response actually contains complete, usable content.
    Returns True if the response has complete code blocks or meaningful content.
    """
    if not response or not response.strip():
        return False
    
    # Use the same regex patterns as the interruption callback
    CODE_BLOCK_REGEX = re.compile(r"```.*?(```)", re.DOTALL)
    XML_CODE_BLOCK_REGEX = re.compile(r"<(python|bash|shell)([^>]*)>.*?</\1>", re.DOTALL)
    TAG_BLOCK_REGEX = re.compile(r"<([a-zA-Z0-9_]+)[^>]*>.*?</\1>", re.DOTALL)
    
    # Check for complete code blocks
    if CODE_BLOCK_REGEX.search(response) or XML_CODE_BLOCK_REGEX.search(response):
        return True
    
    # Check for complete XML tags (excluding reasoning tags)
    tag_match = TAG_BLOCK_REGEX.search(response)
    if tag_match:
        tag_name = tag_match.group(1)
        if not (tag_name.startswith('antml:') or tag_name in ['thinking', 'think', 'plan', 'planning']):
            return True
    
    # If response has substantial content and ends naturally, consider it complete
    response_stripped = response.strip()
    if len(response_stripped) > 50 and (
        response_stripped.endswith('.') or 
        response_stripped.endswith('!') or 
        response_stripped.endswith('?') or
        response_stripped.endswith(':') or
        response_stripped.endswith(';')
    ):
        return True
    
    return False

# Note: The type hint for 'UtilsManager' is a string to avoid circular dependency issues.
# The full definition of other functions like get_extract_blocks, g, LlmRouter, etc.
# are assumed to exist elsewhere in the project.

async def confirm_code_execution(
    args: argparse.Namespace, 
    code_to_execute: str, 
    utils_manager: 'UtilsManager', 
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Handles code execution confirmation with a resilient, multi-stage Guard council.
    It attempts to gather a minimum of 3 votes. If initial models fail, it dynamically
    fetches diverse fallback models to ensure a decision can be made.
    """
    # Fast-path for simple, non-destructive shell commands
    if not get_extract_blocks()(code_to_execute, "python"):
        always_permitted_bash = ["ls ", "pwd ", "cd ", "echo ", "print ", "cat ", "head ", "tail ", "grep ", "sed ", "awk ", "sort "]
        bash_code = "\n".join(get_extract_blocks()(code_to_execute, ["bash", "shell"]))
        bash_code_lines = [line for line in bash_code.split("\n") if line.strip() and not line.strip().startswith("#")]
        if bash_code_lines and all(any(line.strip().startswith(cmd) for cmd in always_permitted_bash) for line in bash_code_lines):
            logging.info(colored("✅ Code execution permitted automatically (safe command list)", "green"))
            return True

    # --- Handle Default Mode (single LLM with manual confirmation) ---
    if not args.monte_carlo:
        return await enhanced_permission_prompt(
            args, 
            tool_name="code_execution",
            action_description="Execute code block",
            input_event=input_event,
            input_lock=input_lock,
            shared_input=shared_input,
            get_user_input_func=get_user_input_with_bindings
        )

    # --- Guard-based Logic ---
    llm_config = g.get_llm_config()
    guard_council = [key for key, data in llm_config.items() if data.get('selected') and data.get('guard', 0) > 0 for _ in range(data['guard'])]

    # 1. If no guards are configured, fall back to manual confirmation.
    if not guard_council:
        if args.auto:
            logging.info(colored("✅ Code execution permitted automatically (--auto flag, no guards configured).", "green"))
            return True
        logging.warning(colored("🛡️", "red", attrs=["bold"]) + colored(" No Guard models configured. Manual confirmation required.", "yellow"))
        return await enhanced_permission_prompt(
            args, 
            tool_name="code_execution",
            action_description="Execute code block (no guards configured)",
            input_event=input_event,
            input_lock=input_lock,
            shared_input=shared_input,
            get_user_input_func=get_user_input_with_bindings
        )

    # Display the initial guard council.
    guard_display = [f"☁️ {colored(m, 'green')}" if ':' not in m else f"🏠 {colored(m, 'green')}" for m in guard_council]
    logging.info(colored("🛡️", "red", attrs=["bold"]) + f"  {len(guard_council)} Guards verifying code: {', '.join(guard_display)}")

    async def get_verdict(model_key: str):
        """Gets a single verdict from a model. Returns (verdict, analysis, model_key) on success, or None on failure."""
        execution_guard_chat = Chat(
            instruction_message=f"""You are a Code Execution Guardian. Your primary goal is to prevent unsafe or incomplete code execution. Analyze the code for safety and completeness.
- SAFE operations: Reading files (ls, cat), simple data retrieval (curl to public APIs), basic system info (ps, uname), file modifications with clear comments.
- UNSAFE operations: File deletions (rm -rf), system modifications (sudo), unrestricted network access, running unknown executables.
- INCOMPLETE code: Placeholders like `YOUR_API_KEY`, `TODO`, or unimplemented functions.

**Process:**
1. **Brief Analysis:** Follow the final guidance below
2. **Single Word Verdict:** End your response with exactly one word: `yes`, `no`, or `unfinished`.

**Although these following classes are custom and generally considered trusted. You MUST verify each line for intentional and careful usage.**
{utils_manager.get_util_names()} 

**Final guidance**
1. Observe if the goal of the code is clarified and reasonable while any persistenly modifying operations like deletions have been backed up sensibly first. 
2. Express your grasp on the system mindfully before responding with a final word verdict.
""",
            debug_title=f"🛡️  Guard Vote ({model_key})"
        )
        analysis_prompt = f"Analyze this code for safe execution and completeness:\n{code_to_execute}"
        execution_guard_chat.add_message(Role.USER, analysis_prompt)
        
        try:
            guard_context = {'log_prefix': '🛡️  '}
            response = await LlmRouter.generate_completion(
                execution_guard_chat,
                [model_key],
                force_preferred_model=True,
                decision_patterns={"guard": r'\b(yes|no|unfinished)\b'},
                branch_context=guard_context
            )
            verdict_matches = list(re.finditer(r'\b(yes|no|unfinished)\b', response.lower(), re.DOTALL | re.MULTILINE))
            verdict = verdict_matches[-1].group(1) if verdict_matches else "no"
            analysis = response.strip()
            return verdict, analysis, model_key
        except Exception as e:
            logging.warning(colored(f"⚠️ Guard {model_key} failed: {str(e)}", "yellow"))
            return None

    # --- Resilient, Multi-stage Guard Verification ---
    MIN_VOTES = len(guard_council)
    successful_results = []
    models_attempted = set()
    
    # Stage 1: Attempt with the primary guard council.
    tasks = [get_verdict(model) for model in guard_council]
    models_attempted.update(guard_council)
    batch_results = await asyncio.gather(*tasks)
    successful_results.extend([res for res in batch_results if res is not None])
    
    # Track which guard models failed
    failed_guards = []
    for i, model in enumerate(guard_council):
        if batch_results[i] is None:
            failed_guards.append(model)

    # Stage 2: If needed, fetch iterative fallbacks until minimum votes are met.
    if len(successful_results) < MIN_VOTES:
        guard_fallback_iterator.update_fallback_models(llm_config)
    
    while len(successful_results) < MIN_VOTES:
        # Fetch one fallback model at a time to replace failed models
        fallback_model = guard_fallback_iterator.get_next_fallback(exclude=list(models_attempted))
        fallback_models = [fallback_model] if fallback_model else []

        if not fallback_models:
            logging.warning(colored(f"🛡️  Could not gather {MIN_VOTES} guard responses. All available guard and fallback models exhausted.", "yellow"))
            break # Exit the loop if no more models are available.

        # Enhanced logging showing failed model → replacement
        if failed_guards:
            failed_model = failed_guards.pop(0) if failed_guards else "unknown guard"
            replacement = fallback_models[0] if fallback_models else "none"
            usage_stats = guard_fallback_iterator.get_usage_stats()
            usage_count = usage_stats.get(replacement, 0)
            
            # Record the failure
            guard_fallback_iterator.record_failure(failed_model, "execution timeout/error", replacement)
            
            logging.info(colored("🔄 Guard replacement: ", "cyan") + 
                        colorize_model_name(failed_model) + 
                        colored(" → ", "cyan") + 
                        colorize_model_name(replacement) + 
                        colored(f" ({usage_count})", "cyan"))
        
        tasks = [get_verdict(model) for model in fallback_models]
        models_attempted.update(fallback_models)
        batch_results = await asyncio.gather(*tasks)
        successful_results.extend([res for res in batch_results if res is not None])

    # --- Final Verdict Processing ---
    if len(successful_results) < MIN_VOTES:
        logging.error(colored(f"❌ Could not get minimum {MIN_VOTES} guard responses (got {len(successful_results)}).", "red"))
        user_input, _ = await get_user_input_with_bindings(
            args, None, colored("Execute code anyway? (y/n): ", "cyan"),
            input_event=input_event, input_lock=input_lock, shared_input=shared_input
        )
        return user_input.lower() == 'y'

    # Analyze guard responses and determine the final outcome.
    response_counts = Counter(res[0] for res in successful_results)
    total_responses = len(successful_results)
    
    verdict_details = []
    for verdict in ['yes', 'no', 'unfinished']:
        count = response_counts.get(verdict, 0)
        if count > 0:
            models_with_verdict = [colorize_model_name(model) for v, _, model in successful_results if v == verdict]
            verdict_details.append(f"{verdict.capitalize()} ({count}/{total_responses} guards: {', '.join(models_with_verdict)})")
    logging.info(colored("🛡️", "red", attrs=["bold"]) + colored(f"  Guard Consensus Summary: {'; '.join(verdict_details)}", "light_blue"))
    
    if response_counts.get('no', 0) > 0:
        final_verdict = 'no'
    elif response_counts.get('unfinished', 0) > 0:
        final_verdict = 'unfinished'
    else:
        final_verdict = 'yes'

    def get_analysis_for_verdict(verdict_type):
        return "\n".join([f"- {analysis}" for v, analysis, _ in successful_results if v == verdict_type])

    if final_verdict == 'yes':
        logging.info(colored(f"✅ Code execution cleared by ({response_counts.get('yes', 0)}/{total_responses}) guards.", "green"))
        return True
    elif final_verdict == 'unfinished':
        analysis = get_analysis_for_verdict('unfinished')
        logging.warning(colored(f"⚠️ Code revision requested by ({response_counts.get('unfinished', 0)}/{total_responses}) guards. Reasoning:\n", "yellow") + colored(analysis, "light_magenta"))
        args.message.insert(0, f"The code was deemed safe but unfinished. Please add comments and complete the implementation. Code review:\n{analysis}")
        logging.info(colored("💬 Auto-prompting with refinement instructions...", "blue"))
        return False
    else: # 'no'
        analysis = get_analysis_for_verdict('no')
        logging.error(colored(f"❌ Code execution blocked by ({response_counts.get('no', 0)}/{total_responses}) guards. Reasoning:\n", "red") + colored(analysis, "magenta"))
        user_response, _ = await get_user_input_with_bindings(
            args, None, colored("Execute code anyway? (y=execute, n=abort, c=refine): ", "cyan"),
            input_event=input_event, input_lock=input_lock, shared_input=shared_input
        )
        
        if user_response.lower() in ['y', 'yes', 'execute']:
            logging.warning(colored("✅ Code execution permitted by user override", "yellow"))
            return True
        elif user_response.lower() in ['c', 'refine']:
            args.message.insert(0, f"The code was rejected by security guards. Please refine the implementation. Code review:\n{analysis}")
            logging.info(colored("💬 Auto-prompting with refinement instructions...", "blue"))
            return False
        else:  # 'n', 'no', 'abort', or any other input
            logging.error(colored("✖️ Execution cancelled.", "red"))
            return False

async def select_best_branch(
    context_chat: Chat,
    assistant_responses: List[str],
    branch_model_map: Optional[Dict[int, str]] = None,
) -> Tuple[int, str]:
    """
    Uses a council of LLM judges to select the best response via a sophisticated tournament system.

    1.  **Win Condition:** A branch wins if it gains a 2+ vote lead.
    2.  **Reversed Temperature:** Starts at 0.6 and increases by 0.05 each round to break deadlocks.
    3.  **Conditional Knockout:** If no winner and >2 candidates remain, a single, unambiguous
        loser (not tied for last) is eliminated.
    4.  **Head-to-Head:** If only 2 candidates remain, they vote until a winner emerges.
    5.  **Performance Logging:** Each vote is timed.

    Returns:
        Tuple[int, str]: (selected_index, final_vote_summary_string)
    """
    # --- 1. Setup: Identify Evaluators and Candidates ---
    llm_config = g.get_llm_config()
    evaluator_council = [key for key, data in llm_config.items() if data.get('selected') and data.get('eval', 0) > 0 for _ in range(data['eval'])]

    if not evaluator_council:
        # Fallback logic for when no evaluators are configured
        evaluation_fallback_iterator.update_fallback_models(llm_config)
        fallback_evaluator = evaluation_fallback_iterator.get_next_fallback()
        if fallback_evaluator:
            evaluator_council.append(fallback_evaluator)
            logging.info(colored("No specific evaluator set. Using fallback: ", "cyan") + colorize_model_name(fallback_evaluator))
        elif g.SELECTED_LLMS:
            evaluator_council.append(g.SELECTED_LLMS[0])
            logging.info(colored("No fallback models available. Using primary LLM to judge: ", "yellow") + colorize_model_name(g.SELECTED_LLMS[0]))
        else:
            logging.error(colored("⚠️ No evaluators available to judge branches. Defaulting to first branch.", "red"))
            return 0, "No evaluator available."

    num_alternatives = len(assistant_responses)
    if num_alternatives <= 1:
        return 0, "Only one branch generated."


    async def get_vote(model_key: str, prompt_to_use: str, temperature: float, index_map: Dict[int, int]):
        """Asks a single model to vote, times it, and returns the original branch index."""
        judge_chat = Chat(instruction_message="You are an impartial judge...", debug_title=f"⚖️  Branch Judge ({model_key})")
        judge_chat.add_message(Role.USER, prompt_to_use)
        
        start_time = time.time()
        try:
            vote_context = {'start_time': start_time, 'log_printed': False, 'store_message': False, 'log_prefix': '⚖️  ', 'index_map': index_map}
            response = await LlmRouter.generate_completion(
                judge_chat, [model_key], force_preferred_model=True, temperature=temperature,
                decision_patterns={"eval": r'Selected index:\s*(\d+)'}, branch_context=vote_context
            )
            match = re.search(r'Selected index:\s*(\d+)', response)
            elapsed_time = time.time() - start_time

            if match:
                selected_index = int(match.group(1))
                if selected_index in index_map:
                    original_index = index_map[selected_index]
                    # LlmRouter's decision pattern already logs the result, so we just return
                    return original_index, model_key, None

            logging.warning(colored(f"⚖️  ⚠️  Could not parse vote from {colorize_model_name(model_key)}: '...{response[-20:]}' ({elapsed_time:.1f}s)", "yellow"))
            return None, model_key, "invalid response format"
        except Exception as e:
            elapsed_time = time.time() - start_time
            logging.error(colored(f"⚖️  ❌ Evaluator {colorize_model_name(model_key)} failed ({elapsed_time:.1f}s): {e}", "red"))
            return None, model_key, str(e)

    # --- 2. Main Voting Loop ---
    current_round = 1
    current_temperature = 0.6  # Start with a more deterministic temperature
    vote_counts = Counter()
    candidate_indices = list(range(num_alternatives))

    while True:
        if len(candidate_indices) == 1:
            winner_index = candidate_indices[0]
            logging.info(colored(f"🏆 Winner by elimination: Branch {winner_index} is the last one standing.", "green"))
            return winner_index, ""

        # A. Prepare prompt for the current round's active candidates
        index_map = {i: original_idx for i, original_idx in enumerate(candidate_indices)}
        prompt = f"VOTING ROUND {current_round}: Please choose from these {len(candidate_indices)} options.\n"
        for i, original_idx in enumerate(candidate_indices):
            prompt += f"\n--- INDEX {i} (Original Branch {original_idx}) ---\n{assistant_responses[original_idx]}"
        prompt += "\n\n**Your task:** ... end your response with `Selected index: [number]` from the list above."
        
        logging.info(colored(f"⚖️  Round {current_round} voting on {len(candidate_indices)} candidates (Temp: {current_temperature:.2f}): ", "blue", attrs=["bold"]) + colorize_model_list(evaluator_council))
        
        # B. Gather and accumulate votes
        tasks = [get_vote(model, prompt, current_temperature, index_map) for model in evaluator_council]
        results = await asyncio.gather(*tasks)
        valid_votes = [res for res in results if res[0] is not None]
        vote_counts.update(Counter(vote[0] for vote in valid_votes))

        # C. Log cumulative scores for all original branches
        if vote_counts:
            vote_details = []
            # Create list of (index, count) tuples and sort by count descending
            sorted_branches = sorted(
                [(index, vote_counts.get(index, 0)) for index in range(num_alternatives)], 
                key=lambda x: x[1], 
                reverse=True
            )
            for index, count in sorted_branches:
                model_name = colorize_model_name(branch_model_map.get(index, 'Unknown'))
                status = "" if index in candidate_indices else colored(" (eliminated)", "dark_grey")
                entry_text = f"Branch {index} ({model_name}): {count} vote(s){status}"
                vote_details.append(entry_text)
            logging.info(f"⚖️  🗳️  Cumulative Score (Round {current_round}): {' | '.join(vote_details)}")

        # D. Check for a decisive winner among *current* candidates
        filtered_counts = Counter({i: vote_counts[i] for i in candidate_indices})
        most_common_current = filtered_counts.most_common()
        
        if most_common_current:
            winner_index, winner_count = most_common_current[0]
            if len(most_common_current) == 1 or winner_count >= most_common_current[1][1] + 2:
                margin = winner_count - (most_common_current[1][1] if len(most_common_current) > 1 else 0)
                
                logging.info(colored(f"🏆 Decisive winner found: Branch {winner_index} (", "light_green") + colorize_model_name(branch_model_map.get(winner_index, 'Unknown')) + colored(f") won with a margin of {margin}. Next: ⚖️  ➡️  🛡️", "light_green"))
                return winner_index, ""

        # E. If no winner, apply conditional knockout rule
        # This block only runs if there are more than 2 candidates left.
        if len(candidate_indices) > 2:
            min_score = min(filtered_counts.values())
            # Find all candidates tied for the lowest score
            losers = [idx for idx, score in filtered_counts.items() if score == min_score]
            
            # ONLY eliminate if there is a single, unambiguous loser.
            if len(losers) == 1:
                eliminated_branch_index = losers[0]
                candidate_indices.remove(eliminated_branch_index)
                eliminated_model_name = colorize_model_name(branch_model_map.get(eliminated_branch_index, 'Unknown'))
                logging.info(colored(f"⚖️  Knockout: Branch {eliminated_branch_index} (", "yellow") + eliminated_model_name + colored(f") eliminated ({min_score} votes).", "yellow"))
            else:
                logging.info(colored(f"⚖️  Deadlock for last place ({min_score} votes). No candidate will be eliminated this round.", "cyan"))

        # F. Prepare for the next round
        current_round += 1
        current_temperature = round(current_temperature + 0.05, 2)


def _generate_tool_documentation() -> str:
    """
    Dynamically discovers all tools from the tools directory and formats
    their documentation for the system prompt.
    """
    docs_string = "3.  **ACT**: Decide on a tool and execute it to pass your current step. You can make use of the following tools:\n\n"
    
    # Import and discover tools from the tools directory
    tools_dir = Path(__file__).parent / "tools"
    tool_classes = {}
    
    if tools_dir.exists():
        for file in tools_dir.glob("*.py"):
            if file.name.startswith("_") or file.stem == "__init__":
                continue
            
            try:
                # Import the tool module
                module_name = f"tools.{file.stem}"
                module = importlib.import_module(module_name)
                
                # Find the tool class (should match the filename)
                tool_class_name = file.stem
                if hasattr(module, tool_class_name):
                    tool_class = getattr(module, tool_class_name)
                    if hasattr(tool_class, 'get_delim') and hasattr(tool_class, 'get_tool_info'):
                        tool_classes[tool_class.get_delim()] = tool_class
            except Exception as e:
                logging.warning(f"Could not load tool from {file}: {e}")
    
    # Sort tools to ensure a consistent order in the prompt
    sorted_tool_names = sorted(tool_classes.keys())

    for tool_name in sorted_tool_names:
        tool_class = tool_classes[tool_name]
        try:
            info = tool_class.get_tool_info()
            # Use four spaces for indentation in the final prompt
            docs_string += (
                f"    - **`{info['name']}`**: {info['description']}\n"
                f"      *Example*: `{info['example']}`\n\n"
            )
        except Exception as e:
            logging.warning(f"Could not generate documentation for tool '{tool_name}': {e}")
    
    # Store tool_classes globally for later use. This is a critical side-effect
    # that makes the tools available to the rest of the application.
    g.tool_classes = tool_classes
    return docs_string


async def construct_engineered_prompt() -> Chat:
    """
    Constructs the initial Chat object with a dynamically generated system prompt.
    This prompt includes the agent's core principles and a list of available tools.
    """
    base_instructions = '''**SYSTEM: Agent Protocol & Capabilities**

You are a sophisticated AI agent operating within a command-line interface (CLI) and a python notebook. Your primary directive is to understand user requests, formulate a plan, and execute code to achieve the goal.

### Guiding Principles (Your Core Logic)
You must follow this loop for every task.
1.  **OBSERVE & DECOMPOSE**: Analyze the user's request and break it down into the required steps. Start with: `<think>...</think>`.
2.  **PLAN & TRACK**: Display/update your <todos>...</todos> to align your next step(s) with the stated long term goal(s).'''
    
    # Dynamically generate the documentation for all available tools
    # This also populates g.tool_classes as a side effect.
    tool_docs = _generate_tool_documentation()
    
    step_4 = "4.  **ITERATE**: After executing a tool, return to Step 1 **OBSERVE & DECOMPOSE** to reflect on your observations and plan your next move."
    
    # Combine all parts into the final instruction message
    default_inst = f"{base_instructions}\n{tool_docs}\n{step_4}"
    
    # Create and configure the new Chat object
    context_chat = Chat(debug_title="Main Context Chat")
    context_chat.set_instruction_message(default_inst)
    
    return context_chat


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


def handle_multiline_input() -> str:
    """Handles multiline input from the user."""
    print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
    lines = []
    while True:
        try:
            line = input()
            if line == "--f":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

async def get_user_input_with_bindings(
    args: argparse.Namespace,
    context_chat: Chat,
    prompt: str = colored(" Enter your request: ", 'blue', attrs=["bold"]),
    input_override: str = None,
    force_input: bool = False,
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str]]:
    """Interactively captures user input and processes special keybindings.

    This function serves as the primary Read-Eval-Print Loop (REPL). It
    returns the user's text prompt. If a command that generates data (like
    taking a screenshot) is used, it returns that data alongside an empty prompt,
    allowing the main loop to handle it and re-prompt for text.

    Args:
        args (argparse.Namespace): The application's runtime arguments, modified
            in place by commands.
        context_chat (Chat): The current conversation context, modified in place
            by commands.
        prompt (str, optional): The prompt string to display.
        input_override (str, optional): A string to use instead of interactive input.
        force_input (bool, optional): If True, forces a direct `input()` call.

    Returns:
        Tuple[str, List[str]]: A tuple containing:
            - The user's text input string.
            - A list of base64-encoded image strings, if any were captured.
    """
    # --- First Call Logic ---
    if get_user_input_with_bindings.is_first_call:
        get_user_input_with_bindings.is_first_call = False

    while True:
        if prompt == "":
            user_input = ""
        elif input_override:
            user_input = input_override
        elif force_input:
            if input_event and input_lock and shared_input is not None:
                await input_event.wait()
                input_event.clear()
                with input_lock:
                    user_input = shared_input["value"]
            else:
                user_input = input(prompt)
        else:
            try:
                # --- FIX: Use the thread-safe event/lock mechanism ---
                if input_event and input_lock and shared_input is not None:
                    # In TUI mode, wait for the event that signals input is ready
                    agent_logger = logging.getLogger("agent_main")
                    agent_logger.info("Agent waiting for input event...")
                    await input_event.wait()
                    agent_logger.info("Input event received! Processing...")
                    input_event.clear() # Reset the event for the next input
                    with input_lock:
                        user_input = shared_input["value"]
                        agent_logger.info(f"Retrieved user input: '{user_input}'")
                else:
                    # Original CLI behavior
                    if context_chat:
                        user_input = input(f"\n[Tokens: {math.ceil(len(context_chat.__str__())*3/4)} | Messages: {len(context_chat.messages)}] " + prompt)
                    else:
                        user_input = input(prompt)
            except KeyboardInterrupt:
                logging.warning(colored("\n# cli-agent: Exiting due to Ctrl+C...", "yellow"))
                exit()
        
        # --- Command Handling ---
        if user_input in ["-img", "--img", "-screenshot", "--screenshot"] or args.image:
            logging.info(colored("# cli-agent: Taking screenshot.", "green"))
            args.image = False
            base64_images = await handle_screenshot_capture()
            # Return the captured image data. The empty string signals no text prompt was entered.
            return "", base64_images
        elif user_input == "-m" or user_input == "--m":
            # Return the multiline input with an empty image list.
            return handle_multiline_input(), []
        
        if user_input == "-r" or user_input == "--r":
            if not context_chat or len(context_chat.messages) < 2:
                logging.error(colored("# cli-agent: No chat history found, cannot regenerate.", "red"))
                continue
            logging.info(colored("# cli-agent: Regenerating last response.", "green"))
            context_chat.messages.pop()
            user_input = ""
        elif user_input == "-l" or user_input == "--local":
            args.local = not args.local
            g.FORCE_LOCAL = args.local
            logging.info(colored(f"# cli-agent: Local mode toggled {'on' if args.local else 'off'}.", "green"))
            continue
        elif user_input == "-llm" or user_input == "--llm":
            logging.info(colored("# cli-agent: Opening LLM selection.", "green"))
            await llm_selection(args, preselected_llms=g.SELECTED_LLMS)
            continue
        elif user_input == "-a" or user_input == "--auto":
            args.auto = not args.auto
            
            llm_config = g.get_llm_config()
            guard_count = sum(
                data.get('guard', 0) 
                for data in llm_config.values() 
                if data.get('selected')
            )
            
            status_msg = f"# cli-agent: Automatic execution toggled {'on' if args.auto else 'off'}"
            if args.auto and guard_count > 0:
                status_msg += f" ({guard_count} guards active)"
            elif args.auto and guard_count == 0:
                status_msg += " (no guards, will execute without confirmation)"
            elif not args.auto and guard_count > 0:
                status_msg += f" (ignoring {guard_count} available guards)"
            
            logging.info(colored(f"{status_msg}.", "green"))
            continue
        elif user_input == "-f" or user_input == "--fast":
            args.fast = not args.fast
            logging.info(colored(f"# cli-agent: Fast mode toggled {'on' if args.fast else 'off'} (deprecated).", "green"))
            continue
        elif user_input == "-v" or user_input == "--v":
            args.voice = not args.voice
            logging.info(colored(f"# cli-agent: Voice mode toggled {'on' if args.voice else 'off'}.", "green"))
            continue
        elif user_input == "--full_output":
            args.full_output_mode = not args.full_output_mode
            g.SUMMARY_MODE = not args.full_output_mode
            logging.info(colored(f"# cli-agent: Full output mode toggled {'on' if args.full_output_mode else 'off'}.", "green"))
            continue
        elif user_input == "-p" or user_input == "--p":
            logging.info(colored("# cli-agent: Printing chat history.", "green"))
            os.system('clear')
            print(colored("Chat history:", "green"))
            if context_chat:
                context_chat.print_chat()
            else:
                print(colored("No chat history available.", "yellow"))
            continue
        elif user_input in ["-o", "--o", "-online", "--online"]:
            args.online = not args.online
            logging.info(colored(f"# cli-agent: Online mode toggled {'on' if args.online else 'off'}.", "green"))
            continue
        elif user_input in ["-e", "--e", "--exit"]:
            logging.info(colored("# cli-agent: Exiting...", "green"))
            exit(0)
        elif user_input in ["-h", "--h", "--help"]:
            print_startup_summary(args)
            CMD_WIDTH = 20
            print(colored("\n--- Other Commands ---", "yellow"))
            print(f"  {colored('-r, --regenerate'.ljust(CMD_WIDTH), 'white')}Regenerate last response")
            print(f"  {colored('-img, --image'.ljust(CMD_WIDTH), 'white')}Take screenshot")
            print(f"  {colored('-p, --p'.ljust(CMD_WIDTH), 'white')}Print chat history")
            print(f"  {colored('-m, --m'.ljust(CMD_WIDTH), 'white')}Enter multiline input mode")
            print(f"  {colored('-e, --exit'.ljust(CMD_WIDTH), 'white')}Exit CLI-Agent")
            print(f"  {colored('-h, --help'.ljust(CMD_WIDTH), 'white')}Show this help")
            print(f"  {colored('-s, --save'.ljust(CMD_WIDTH), 'white')}Save current LLM selection as default")
            print(f"  {colored('--full_output'.ljust(CMD_WIDTH), 'white')}Show full model output instead of summary")
            continue
        elif user_input in ["-s", "--save"]:
            await save_current_llm_config_as_default()
            continue
        elif user_input in ["-clear", "--clear"]:
            context_chat = None
            context_chat = await construct_engineered_prompt()
            print("# cli-agent: Chat cleared.")
            continue
        elif user_input in ["-compact", "--compact"]:
            if context_chat and context_chat.messages:
                # Clear current chat and start fresh
                print("# cli-agent: Conversation compacting instructed.")
                g.AGENT_IS_COMPACTING = True
                return f"Let's pause for a moment, please write a summary of our progress inside a markdown file at: {g.CLIAGENT_PERSISTENT_STORAGE_PATH }/MEMORY/replace_with_descriptive_title.md", []
                
            continue
        # If the code reaches here, it's a normal text prompt.
        return user_input, []

# Initialize the stateful flag for the first call.
get_user_input_with_bindings.is_first_call = True

async def handle_screenshot_capture() -> List[str]:
    """
    Handles the screenshot capture process.

    Returns:
        List[str]: The base64 encoded image data or an empty list if failed.
    """
    base64_images: List[str] = []
    screenshots_paths: List[str] = []
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        base64_images = []
        try:
            logging.info(colored("Attempting screenshot with Spectacle (region selection)...", "green"))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_filename = temp_file.name
            subprocess.run(['spectacle', '-rbno', temp_filename], check=True)
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                with open(temp_filename, 'rb') as image_file:
                    png_data = image_file.read()
                base64_img = base64.b64encode(png_data).decode('utf-8')
                base64_images = [base64_img]
                pyperclip.copy(base64_img)
                logging.info(colored("Screenshot captured successfully. (+Copied to clipboard)", "green"))
            else:
                logging.warning(colored(f"No screenshot was captured (attempt {attempt}).", "yellow"))
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        except subprocess.CalledProcessError:
            logging.error(colored(f"Spectacle command failed on attempt {attempt}. Is it installed?", "red"))
        except Exception as e:
            logging.exception(f"Unexpected error during screenshot capture attempt {attempt}: {e}")
        if base64_images:
            break
        if attempt < max_attempts and not base64_images:
            logging.warning(colored("Retrying screenshot capture...", "yellow"))
            await asyncio.sleep(2)
    if base64_images:
        images_dir = os.path.join(g.AGENTS_SANDBOX_DIR, "screenshots")
        os.makedirs(images_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, base64_img in enumerate(base64_images):
            try:
                img_data = base64.b64decode(base64_img)
                img_path = os.path.join(images_dir, f"screenshot_{timestamp}_{i+1}.png")
                screenshots_paths.append(img_path)
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
                logging.info(colored(f"Screenshot saved to {img_path}", "green"))
            except Exception as e:
                logging.error(colored(f"Error saving screenshot {i+1}: {e}", "red"))
    else:
        logging.error(colored("No images were captured.", "red"))
    if not base64_images:
        logging.warning(colored("# cli-agent: No screenshot was captured after multiple attempts.", "yellow"))
        return []
    
    return base64_images

def preprocess_consecutive_sudo_commands(code: str) -> str:
    """Combine consecutive shell commands with sudo to reduce password prompts."""
    lines = code.strip().split('\n')
    processed_lines = []
    shell_commands = []
    for line in lines:
        line = line.strip()
        if line.startswith('!'):
            shell_commands.append(line[1:])
        else:
            if shell_commands:
                combined = ' && '.join(shell_commands)
                processed_lines.append(f'!{combined}')
                shell_commands = []
            processed_lines.append(line)
    if shell_commands:
        combined = ' && '.join(shell_commands)
        processed_lines.append(f'!{combined}')
    return '\n'.join(processed_lines)

def extract_paths(user_input: str) -> Tuple[List[str], List[str]]:
    """
    Extracts local file paths, folder paths, and online URLs from a string.
    """
    token_pattern = re.compile(r'(["\'])(.+?)\1|(\S+)')
    trailing_chars_to_strip = '.,;:"\'!?)>]}'
    local_paths = set()
    online_paths = set()
    for match in token_pattern.finditer(user_input):
        candidate = match.group(2) or match.group(3)
        if not candidate:
            continue
        cleaned_candidate = candidate.rstrip(trailing_chars_to_strip)
        if cleaned_candidate.startswith(('http://', 'https://')):
            online_paths.add(cleaned_candidate)
        elif cleaned_candidate.startswith(('/', './', '~/', 'file://')) or re.match(r'^[a-zA-Z]:[\\/]', cleaned_candidate):
            local_paths.add(cleaned_candidate)
    return list(local_paths), list(online_paths)

# The main() function is now the primary entry point for all logic
def initialize_notebook_lazy():
    """
    Lazy initialization of the ComputationalNotebook.
    This is called on the first use of bash or python tools.
    """
    import logging
    from termcolor import colored
    
    if hasattr(g, 'notebook') and g.notebook is not None:
        return g.notebook  # Already initialized
    
    tui_callback = getattr(g, 'tui_callback', None)
    stdout_callback = getattr(g, 'stdout_callback', None)
    stderr_callback = getattr(g, 'stderr_callback', None)
    
    if not stdout_callback or not stderr_callback:
        raise RuntimeError("Notebook callbacks not available - main initialization may have failed")
    
    if tui_callback: 
        tui_callback("  - Initializing computational notebook...", "system")
    else: 
        logging.info(colored("  - Initializing computational notebook (tmux + python)...", "blue"))
    
    notebook = ComputationalNotebook(stdout_callback=stdout_callback, stderr_callback=stderr_callback)
    g.notebook = notebook
    
    if tui_callback: 
        tui_callback("✔ Notebook ready.", "system_ok")
    else: 
        logging.info(colored("✔ Computational notebook ready.", "green"))
    
    return notebook

async def generate_branch(model_key: str, branch_index: int, lock: asyncio.Lock, context_chat: Chat, painter: TextStreamPainter, temperature: float, base64_images: list, llm_config: dict, branch_model_map: dict, num_branches: int):
    """Generates a single response branch, handling model fallbacks."""
    try:
        async with lock:
            logging.debug(f"🚀 Branch {branch_index+1} starting with primary model: {colorize_model_name(model_key)}")
        
        branch_model_map[branch_index] = model_key
        response_buffer_list = [""]
        branch_update_callback = create_interruption_callback(
            response_buffer=response_buffer_list,
            painter=painter,
            lock=lock
        )
        
        branch_fallback_iterator.update_fallback_models(llm_config)
        
        exclude_list = [model_key]
        # Get all available fallback models instead of limiting to 2
        iterative_fallbacks = branch_fallback_iterator.get_multiple_fallbacks(
            count=10,  # Generous limit to get all available fallbacks
            exclude=exclude_list
        )
        
        if not iterative_fallbacks:
            other_branch_models = [
                key for key, data in llm_config.items()
                if (data.get('selected') and 
                    key not in LlmRouter().failed_models and
                    key != model_key and
                    data.get('beams', 0) > 0)
            ]
            iterative_fallbacks = other_branch_models  # Use all available models
            
        async with lock:
            if iterative_fallbacks:
                usage_stats = branch_fallback_iterator.get_usage_stats()
                usage_info = ", ".join([f"{m}({usage_stats.get(m, 0)})" for m in iterative_fallbacks])
                logging.debug(f"🔄 Branch {branch_index+1} branch fallbacks: {usage_info}")
        
        models_to_try = [model_key] + iterative_fallbacks
        
        # Pass all models to LLM router at once for proper fallback handling
        try:
            context_chat.debug_title = f"MCT Branch {branch_index+1}/{num_branches}"
            
            async with lock:
                logging.debug(f"🎯 Branch {branch_index+1} primary model: {colorize_model_name(model_key)} with {len(iterative_fallbacks)} fallbacks")
            
            branch_start_time = time.time()
            current_branch_context = {
                'branch_number': branch_index,
                'store_message': True,
                'log_printed': False,
                'start_time': branch_start_time
            }
            async with lock:
                print("🌿 ", end="", flush=True)

            await LlmRouter.generate_completion(
                chat=context_chat,
                preferred_models=models_to_try,  # Pass ALL models for proper fallback
                force_preferred_model=True,
                temperature=temperature,
                base64_images=base64_images,
                    generation_stream_callback=branch_update_callback,
                    strengths=g.LLM_STRENGTHS,
                    thinking_budget=None,
                    exclude_reasoning_tokens=True,
                    branch_context=current_branch_context
                )
            async with lock:
                elapsed_time = time.time() - current_branch_context.get('start_time', time.time())
                timing_str = f" ({elapsed_time:.1f}s)"
                if current_branch_context.get('log_printed', False):
                    print(colored(f"{timing_str}", "green"))
                else:
                    print(colored(f"🔄 Branch {branch_index+1} completed{timing_str} ✅", "green"))
            branch_model_map[branch_index] = model_key  # Record the primary model attempted
            return response_buffer_list[0]
        except StreamInterruptedException as e:
            async with lock:
                elapsed_time = time.time() - current_branch_context.get('start_time', time.time())
                timing_str = f" ({elapsed_time:.1f}s)"
                if current_branch_context.get('log_printed', False):
                    print(colored(timing_str, "green"))
                else:
                    print(colored(f"🔄 Branch {branch_index+1} interrupted{timing_str} ✅", "green"))
            branch_model_map[branch_index] = model_key
            return e.response
        except Exception as e:
            async with lock:
                all_tried = colorize_model_list(models_to_try)
                logging.error(colored(f"❌ Branch {branch_index+1} failed after trying all models (", "red") + all_tried + colored(f"): {str(e)[:100]}", "red"))
            return None
    
    except Exception as unexpected_error:
        async with lock:
            logging.error(colored(f"❌ Branch {branch_index+1} (", "red") + colorize_model_name(model_key) + colored(f") unexpected failure: {str(unexpected_error)[:200]}", "red"))
        return None

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
                sandbox=False,
                gui=False,
                c=False,
                llm=None,
                fast=False,
                min=False,
                image=False,
                full_output_mode=False,
                exit=False,
                debug_chats=False
            )
            
            # Setup logging properly for TUI mode (output goes through print/stdout)
            setup_logging(args.debug, args.log_file, tui_mode=True)
            
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
        utils_manager = UtilsManager()

        if os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip():
            args.local = True
        if args.voice:
            args.auto = True
            if not tui_callback: logging.info(colored("# cli-agent: Voice mode enabled, automatically enabling auto execution mode", "green"))
        if args.sandbox:
            g.USE_SANDBOX = True
        
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
            web_server = WebServer()
            g.web_server = web_server
            web_server.start()
        
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
        
        if not tui_callback:
            logging.info(colored("  - Starting background model discovery...", "blue"))
        else:
            logging.info(colored("  - Starting background model discovery...", "blue"))

        g.start_background_model_discovery()
        
        # Store the callbacks globally for lazy notebook initialization
        g.stdout_callback = stdout_callback
        g.stderr_callback = stderr_callback
        g.tui_callback = tui_callback  # Store tui_callback as well for consistent logging
        
        if tui_callback: tui_callback("  - Loading utilities and playbooks...", "system")
        playbook_manager = PlaybookManager(vector_db=utils_manager.vector_db)
        
        if tui_callback: tui_callback("✔ Initialization complete.", "system_ok")
        else: logging.info(colored("Agent ready (notebook will initialize on first use).", "blue"))

        if context_chat is None:
            context_chat = await construct_engineered_prompt()
            
            def get_recent_files_output():
                # This function simulates the output of the bash command for the bootstrap message.
                directory = os.getcwd()
                output_lines = [f"Working Directory: {directory}\n"]
                try:
                    # Find only files, not directories
                    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
                    # Sort by modification time, newest first
                    all_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
                    output_lines.append("5 last modified files:")
                    if not all_files:
                        output_lines.append("-> No files found in this directory.")
                    else:
                        # Get the top 5
                        for filename in all_files[:5]:
                            output_lines.append(f"- {filename}")
                except FileNotFoundError:
                    output_lines.append(f"Error: The directory '{directory}' does not exist.")
                return "\n".join(output_lines)


        if '-m' in sys.argv or '--message' in sys.argv:
            if not args.message:
                multiline_input = handle_multiline_input()
                if multiline_input.strip():
                    args.message.append(multiline_input)

        if not tui_callback:
            logging.info(colored("Ready.", "magenta"))
            swap_to_simple_logging()

        user_interrupt = False
        if not (args.c or args.regenerate):
            context_chat.add_message(Role.USER, "Please list the 5 most recently modified files in this cwd")
            initial_bootstrap_message = f"""<think>
The user wants me to list the 5 most recently modified files in this current working directory.
To successfully list them I can utilise a combination of the bash utilities ls, grep, head, awk and echo.
</think><todos>
- [ ] Use bash to list the 5 most recently modified filed in the cwd.
</todos><bash>
echo "Working Directory: $(pwd)"
echo ""
echo "5 last modified files:"
# List files sorted by modification time, filter for regular files, take the top 5, and format the output.
# The '||' part handles the case where no files are found.
ls -lt | grep "^-" | head -n 5 | awk '{{print "- " $NF}}' || echo "-> No files found in this directory."
</bash><context>
{get_recent_files_output()}
</context><think>
I have completed the task of listing the 5 most recently modified files in the current working directory. I will now update the todos and respond to the user indicating that the task is complete and ask if they need anything else.
</think><todos>
- [x] Use bash to list the 5 most recently modified filed in the cwd.
</todos>I have successfully listed the 5 most recently modified filed in the cwd, is there anything else I can assist you with?"""
            context_chat.add_message(Role.ASSISTANT, initial_bootstrap_message)
        
        while True:
            LlmRouter().failed_models.clear()
            user_input: Optional[str] = None
            
            g.LLM_STRENGTHS = []
            
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

            if args.regenerate:
                user_input = ""
                args.regenerate = False
            elif args.voice:
                user_input, _, wake_word_used = get_listen_microphone()(private_remote_wake_detection=args.private_remote_wake_detection)
            elif args.message:
                user_input = args.message.pop(0)
                print(colored(f"💬 Processing message: {user_input}" + (f" (⏳ {len(args.message)} more queued)" if args.message else ""), 'blue', attrs=['bold']), flush=True)
            elif g.AGENT_IS_COMPACTING:
                g.AGENT_IS_COMPACTING = False
                context_chat = await construct_engineered_prompt()
                context_chat.add_message(Role.USER, "Hi can you please check your memory for the latest summary?")
                context_chat.add_message(Role.ASSISTANT, """<think>
The user wants me to search my memory for a the latest summary.
I need to create a todo and call the summary tool with the order key set to latest.
</think><todos>
- [] Check my memory
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
                    local_paths, _ = extract_paths(user_input)
                    for path in local_paths:
                        expanded_path = os.path.expanduser(path)
                        if os.path.exists(expanded_path) and (os.path.isfile(expanded_path) or os.path.isdir(expanded_path)):
                            view_result_str = ViewFiles.run(paths=[expanded_path])
                            view_result = json.loads(view_result_str)
                            if view_result.get("result"):
                                user_input += f"\n\n# Content of: {path}\n{view_result['result']}"
                except (ImportError, FileNotFoundError, subprocess.CalledProcessError, Exception):
                    pass
                user_interrupt = False

            if LlmRouter.has_unconfirmed_data():
                LlmRouter.confirm_finetuning_data()

            action_counter, assistant_response = 0, ""
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
                                default_fallbacks = ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17', 'gemini-2.5-flash-preview-05-20', 'gemma3n:e2b']
                                g.DEFAULT_FALLBACK_MODELS = default_fallbacks
                                models_for_tasks = [default_fallbacks[0]]
                                logging.info(colored("🔧 No models configured - using default fallbacks: " + ", ".join(default_fallbacks), "yellow"))
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
                            tasks = [generate_branch(model_key, i, print_lock, context_chat, text_stream_painter, temperature, base64_images, llm_config, branch_model_map, num_branches) for i, model_key in enumerate(models_for_tasks)]
                            branch_results = await asyncio.gather(*tasks)
                            response_branches = [res for res in branch_results if res and res.strip()]
                        else:
                            response_buffer_list = [""]
                            interruption_callback = create_interruption_callback(response_buffer_list, text_stream_painter)
                            current_todo = f"<current_todo>\n{todos().get_next_unchecked_task()}\n</current_todo>"
                            for assistant_prefix in ["", "<", current_todo, "<think", "<thinking>"]:
                                if (assistant_prefix):
                                    print(colored(f"Model did not generate text, attempting generation with prefix: {assistant_prefix}", "yellow"))
                                if ("" == assistant_prefix and context_chat.messages[-1][1].endswith("</context>")):
                                    assistant_prefix = "<think>"
                                try:
                                    # In default mode, use all available models for fallback
                                    fallback_models = getattr(g, 'DEFAULT_FALLBACK_MODELS', models_for_tasks)
                                    await LlmRouter.generate_completion(
                                        chat=context_chat,
                                        preferred_models=fallback_models,
                                        force_preferred_model=True,
                                        temperature=temperature,
                                        base64_images=base64_images,
                                        generation_stream_callback=interruption_callback,
                                        assistant_prefix=assistant_prefix
                                    )
                                    generated_text = response_buffer_list[0]
                                except StreamInterruptedException as e: # This exception is called by the tool delimiting mechanism at the end of any xml block
                                    generated_text = e.response
                                
                                if (generated_text and generated_text != assistant_prefix):
                                    response_branches.append(generated_text)
                                    break
                        
                        base64_images = []

                    except KeyboardInterrupt as e:
                        print(colored(f"A KeyboardInterrupt was raised, e: {e}"))
                        break
                    except Exception as e:
                        print(colored(f"A Exception was raised, e: {e}"))
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
                        print(colored("The agent is spamming its last response instead of progressing. Aborting...", "red"))
                        stall_counter, last_action_signature = 0, None
                        break

                    context_chat.add_message(Role.ASSISTANT, assistant_response)
                    
                    # Display the response in summary mode (since streaming doesn't show it)
                    if g.SUMMARY_MODE and assistant_response.strip():
                        print(assistant_response)
                    
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
                        # Keep the voice functionality
                        if args.voice:
                            verbal_text = re.sub(r'```[^`]*```', '', assistant_response)
                            if verbal_text.strip(): # Only attempt TTS if there's actual text to speak
                                try:
                                    logging.info(colored(f"Attempting TTS for: '{verbal_text[:50]}...' (length: {len(verbal_text)})", "cyan"))
                                    text_to_speech(text=verbal_text)
                                    logging.info(colored("TTS call completed.", "cyan"))
                                except Exception as tts_e:
                                    logging.error(colored(f"TTS failed: {tts_e}", "red"), exc_info=True)
                            else:
                                logging.info(colored("No verbal text to speak after stripping code blocks.", "yellow"))
                        
                        # --- Enhanced Path-Aware Permission Checking ---
                        # Check permissions using path detection for tools with file operations
                        should_execute = False
                        if len(tool_tags_found) == 1 and any(k in tool_tags_found[0].__module__ for k in ["todos", "think", "searchweb", "search"]):
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
                                        input_event=input_event,
                                        input_lock=input_lock,
                                        shared_input=shared_input,
                                        get_user_input_func=get_user_input_with_bindings
                                    )
                                    
                                    if not permission_granted:
                                        should_execute = False
                                        break  # If any tool is denied, deny all

                        if should_execute:
                            try:
                                # Reset tool context for clean execution
                                tool_context.reset()
                                tool_results = []
                                for tool_class in tool_tags_found:
                                    tool_name = tool_class.get_delim()
                                    blocks = get_extract_blocks()(assistant_response, [tool_name])
                                    
                                    if not blocks:
                                        continue

                                    # --- FIX: Execute only the first block for each tool to prevent duplicate execution ---
                                    first_block = blocks[0]
                                    if len(blocks) > 1:
                                        logging.warning(colored(f"⚠️  Duplicate '{tool_name}' blocks detected. Only the first was executed.", "yellow"))
                                    
                                    # All tools now share the same simple run signature.
                                    tool_result = tool_class.run(first_block)
                                    
                                    # Wrap tools returned values as tool_ouput
                                    if tool_name not in ["todos"] and tool_result:
                                        tool_results.append(tool_result)
                                
                                returned_context = ""
                                for context in tool_results:
                                    context = re.sub(r'\x1b\[[0-9;]*m', '', context) #  remove ANSI escape codes (also known as terminal escape sequences)
                                    if len(context) > readfile.CHUNK_LIMIT:
                                        context = context[:readfile.CHUNK_LIMIT//4] + "\n...output truncated...\n" + context[-readfile.CHUNK_LIMIT//4:]
                                    returned_context += f"<context>\n{context}\n</context>"

                                if (len(returned_context) > 0):
                                    context_chat.add_message(Role.ASSISTANT, returned_context)
                                    
                                    # Check for images from tool context and add them to the chat
                                    if tool_context.has_images():
                                        context_chat.base64_images.extend(tool_context.get_images())
                                        logging.info(colored(f"📸 Added {len(tool_context.get_images())} image(s) to chat context for vision analysis", "blue"))
                                        tool_context.clear_images()  # Clear images after adding them
                                
                                assistant_response = ""
                                action_counter += 1
                                continue # Continue the agent loop for the next action
                            except Exception:
                                error_output = f"<context>\n{traceback.format_exc()}\n</context>"
                                context_chat.add_message(Role.ASSISTANT, error_output)
                                assistant_response = ""
                                continue # Continue the agent loop for the next action
                        else:
                            cancellation_notice = "<context>\nCode execution cancelled by user or guard.\n</context>"
                            context_chat.add_message(Role.ASSISTANT, cancellation_notice)
                            assistant_response = ""
                            break # Ask for user input
                    else:
                        # No tools found, this is a conversational response.
                        break # Ask for user input
                except KeyboardInterrupt:
                    user_interrupt = True
                    break # Ask for user input
                except Exception as e:
                    if "ctrl+c" not in str(e).lower():
                        logging.critical(colored(f"An unexpected error occurred in the agent loop: {e}", "red"), exc_info=args.debug)
                    user_interrupt = True
                    break # Ask for user input
            
            else:
                print("# cli-agent: No conversation to compact.")

            if args.exit and not args.message:
                exit(0)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.warning(colored("CLI-Agent was interrupted. Shutting down gracefully...", "yellow") )
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            logging.critical(colored(f"CLI-Agent encountered a fatal error: {e}", "red"), exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow") )
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            print(colored(f"\nCLI-Agent encountered a fatal error during startup: {e}", "red"))
            traceback.print_exc()