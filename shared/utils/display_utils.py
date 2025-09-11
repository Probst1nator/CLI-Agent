"""
Display and UI utilities for CLI-Agent.

This module contains UI/display functions extracted from main.py to provide
reusable formatting and colorization functionality across the codebase.
"""

import argparse
import logging
from typing import List
from termcolor import colored

# --- FIX: The import from main.py has been removed from the top level ---

def colorize_model_name(model_name: str, tui_mode: bool = False) -> str:
    """
    Applies color to a model name based on its provider.
    
    Args:
        model_name: The model name to colorize
        tui_mode: Whether to use TUI markup format instead of termcolor
        
    Returns:
        Colorized model name string
    """
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
    """
    Colorizes a list of model names and joins them into a string.
    
    Args:
        model_list: List of model names to colorize
        tui_mode: Whether to use TUI markup format
        
    Returns:
        Comma-separated string of colorized model names
    """
    return ', '.join(colorize_model_name(m, tui_mode) for m in model_list)


def print_startup_summary(args: argparse.Namespace, tui_mode: bool = False) -> None:
    """
    Prints a comprehensive summary of the current agent configuration using the logger.
    
    Args:
        args: Parsed command line arguments
        tui_mode: Whether to use TUI markup format
    """
    # --- FIX: Moved import locally to break circular dependency ---
    from main import get_provider_emoji_and_name
    
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
    auto_color = 'green' if args.auto else 'red'
    
    # Force Local mode status
    force_local = getattr(g, 'FORCE_LOCAL', False) or args.local
    local_status = get_status_str(force_local)
    
    # Voice mode status  
    voice_status = get_status_str(args.voice)
    
    # Online voice status
    online_voice_status = get_status_str(args.online_voice, 'Cloud TTS', 'Local TTS')
    
    # Monte Carlo mode detection
    has_monte_carlo = any(
        llm_config.get(key, {}).get('beams', 0) > 0 or 
        llm_config.get(key, {}).get('eval', 0) > 0 or 
        llm_config.get(key, {}).get('guard', 0) > 0 
        for key in selected_models
    ) or args.monte_carlo
    monte_carlo_status = get_status_str(has_monte_carlo)
    
    # Display configuration summary with proper alignment
    if tui_mode:
        logging.info("Auto-execution:".ljust(CMD_WIDTH) + f" [{auto_color}]{auto_status}[/{auto_color}]")
        logging.info("Force Local:".ljust(CMD_WIDTH) + f" {local_status}")
        logging.info("Voice Mode:".ljust(CMD_WIDTH) + f" {voice_status}")
        logging.info("TTS Mode:".ljust(CMD_WIDTH) + f" {online_voice_status}")
        logging.info("Monte Carlo:".ljust(CMD_WIDTH) + f" {monte_carlo_status}")
    else:
        logging.info("Auto-execution:".ljust(CMD_WIDTH) + " " + colored(auto_status, auto_color))
        logging.info("Force Local:".ljust(CMD_WIDTH) + f" {local_status}")
        logging.info("Voice Mode:".ljust(CMD_WIDTH) + f" {voice_status}")
        logging.info("TTS Mode:".ljust(CMD_WIDTH) + f" {online_voice_status}")
        logging.info("Monte Carlo:".ljust(CMD_WIDTH) + f" {monte_carlo_status}")
    
    # Display model configuration if any models are selected
    if selected_models:
        models_header = "[yellow]--- Model Configuration ---[/yellow]" if tui_mode else colored("--- Model Configuration ---", "yellow")
        logging.info(models_header)
        
        # Calculate total branches based on 'beams'
        total_branches = sum(llm_config.get(key, {}).get('beams', 0) for key in branch_models)
        
        if branch_models:
            count_color = get_count_color(total_branches)
            label = f"  🌿 {total_branches} Branch Generators:".ljust(DESC_WIDTH)
            models_str = colorize_model_list(branch_models, tui_mode)
            if tui_mode:
                logging.info(f"[{count_color}]{label}[/{count_color}] {models_str}")
            else:
                logging.info(colored(label, count_color) + " " + models_str)
        
        if eval_models:
            count = len(eval_models)
            count_color = get_count_color(count)
            label = f"  ⚖️  {count} Branch Evaluators:".ljust(DESC_WIDTH)
            models_str = colorize_model_list(eval_models, tui_mode)
            if tui_mode:
                logging.info(f"[{count_color}]{label}[/{count_color}] {models_str}")
            else:
                logging.info(colored(label, count_color) + " " + models_str)
        
        if guard_models:
            count = len(guard_models)
            count_color = get_count_color(count)
            label = f"  🛡️  {count} Execution Guards:".ljust(DESC_WIDTH)
            models_str = colorize_model_list(guard_models, tui_mode)
            if tui_mode:
                logging.info(f"[{count_color}]{label}[/{count_color}] {models_str}")
            else:
                logging.info(colored(label, count_color) + " " + models_str)
        
        # Always show fallback models for visibility, even if none are configured
        count = len(fallback_models)
        count_color = "light_green" if count > 0 else "red"
        label = f"  🔄 {count} Fallback Models:".ljust(DESC_WIDTH)
        models_str = colorize_model_list(fallback_models, tui_mode) if fallback_models else "None"
        if tui_mode:
            logging.info(f"[{count_color}]{label}[/{count_color}] {models_str}")
        else:
            logging.info(colored(label, count_color) + " " + models_str)
    
    # Display final status line
    status_line = "[yellow]--- Ready for User Input ---[/yellow]" if tui_mode else colored("--- Ready for User Input ---", "yellow")
    logging.info(status_line)

def show_unified_bindings_display() -> None:
    """
    Shows a simple display of available commands.
    """
    # Import command definitions from the refactored module
    from agent.core.command_system import _COMMAND_DEFINITIONS
    
    print(colored("\n--- Available Commands ---", "yellow"))
    CMD_WIDTH = 28
    for cmd_info in _COMMAND_DEFINITIONS:
        aliases = ', '.join(cmd_info['aliases'])
        line = f"  {colored(aliases.ljust(CMD_WIDTH), 'white')}{cmd_info['description']}"
        print(line)