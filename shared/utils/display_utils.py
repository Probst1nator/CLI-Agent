"""
Display and UI utilities for CLI-Agent.

This module contains UI/display functions extracted from main.py to provide
reusable formatting and colorization functionality across the codebase.
"""

import argparse
import logging

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


def colorize_model_list(model_list: list[str], tui_mode: bool = False) -> str:
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
    from core.globals import g  # Lazy import to avoid circular dependency at top level

    header = "[yellow]--- Agent Configuration ---[/yellow]" if tui_mode else colored("--- Agent Configuration ---", "yellow")
    logging.info(header)

    # Get detailed LLM configuration
    llm_config = g.get_llm_config()
    # Use g.SELECTED_LLMS to preserve user selection order instead of iterating through config
    selected_models = g.SELECTED_LLMS if g.SELECTED_LLMS else [key for key, data in llm_config.items() if data.get('selected')]

    # All selected models are now treated equally (no specialized roles)
    fallback_models = selected_models

    # Auto Execution status
    if args.yolo:
        auto_status = 'ON'
    else:
        auto_status = 'OFF'
    auto_color = 'green' if args.yolo else 'red'

    # Force Local mode status
    force_local = getattr(g, 'FORCE_LOCAL', False) or args.local
    local_status = get_status_str(force_local)

    # Voice mode status
    voice_status = get_status_str(args.voice)

    # Online voice status
    online_voice_status = get_status_str(args.online_voice, 'Cloud TTS', 'Local TTS')


    # Display configuration summary with proper alignment
    if tui_mode:
        logging.info("Auto-execution:".ljust(CMD_WIDTH) + f" [{auto_color}]{auto_status}[/{auto_color}]")
        logging.info("Force Local:".ljust(CMD_WIDTH) + f" {local_status}")
        logging.info("Voice Mode:".ljust(CMD_WIDTH) + f" {voice_status}")
        logging.info("TTS Mode:".ljust(CMD_WIDTH) + f" {online_voice_status}")
    else:
        logging.info("Auto-execution:".ljust(CMD_WIDTH) + " " + colored(auto_status, auto_color))
        logging.info("Force Local:".ljust(CMD_WIDTH) + f" {local_status}")
        logging.info("Voice Mode:".ljust(CMD_WIDTH) + f" {voice_status}")
        logging.info("TTS Mode:".ljust(CMD_WIDTH) + f" {online_voice_status}")

    # Display model configuration if any models are selected
    if selected_models:
        models_header = "[yellow]--- Model Configuration ---[/yellow]" if tui_mode else colored("--- Model Configuration ---", "yellow")
        logging.info(models_header)

        # Display selected models
        count = len(selected_models)
        count_color = "light_green" if count > 0 else "red"
        label = f"  {count} Selected Models:".ljust(DESC_WIDTH)
        models_str = colorize_model_list(selected_models, tui_mode) if selected_models else "None"
        if tui_mode:
            logging.info(f"[{count_color}]{label}[/{count_color}] {models_str}")
        else:
            logging.info(colored(label, count_color) + " " + models_str)

        # Display role-specific LLMs (Override, Autocomplete, Router, Planner)
        override_llm = getattr(g, 'override_llm', None) or 'None'
        override_label = "  Override LLM:".ljust(DESC_WIDTH)
        if tui_mode:
            logging.info(f"[red]{override_label}[/red] {override_llm}")
        else:
            logging.info(colored(override_label, "red") + " " + override_llm)

        autocomplete_llm = getattr(g, 'autocomplete_llm', None) or 'None'
        autocomplete_label = "  Autocomplete LLM:".ljust(DESC_WIDTH)
        if tui_mode:
            logging.info(f"[cyan]{autocomplete_label}[/cyan] {autocomplete_llm}")
        else:
            logging.info(colored(autocomplete_label, "cyan") + " " + autocomplete_llm)

        router_llm = getattr(g, 'router_llm', None) or 'None'
        router_label = "  Router LLM:".ljust(DESC_WIDTH)
        if tui_mode:
            logging.info(f"[cyan]{router_label}[/cyan] {router_llm}")
        else:
            logging.info(colored(router_label, "cyan") + " " + router_llm)

        planner_llm = getattr(g, 'planner_llm', None) or 'None'
        planner_label = "  Planner LLM:".ljust(DESC_WIDTH)
        if tui_mode:
            logging.info(f"[cyan]{planner_label}[/cyan] {planner_llm}")
        else:
            logging.info(colored(planner_label, "cyan") + " " + planner_llm)

    # Display final status line
    status_line = "[yellow]--- Ready for User Input ---[/yellow]" if tui_mode else colored("--- Ready for User Input ---", "yellow")
    logging.info(status_line)

def show_unified_bindings_display(gui_mode: bool = False) -> None:
    """
    Shows a colorful display of available commands.

    Args:
        gui_mode: If True, hides commands that are not relevant in GUI mode (like /llm)
    """
    # Import command definitions from the refactored module
    from agent.core.command_system import _COMMAND_DEFINITIONS

    print(colored("\n--- Available Commands ---", "yellow", attrs=["bold"]))
    CMD_WIDTH = 28
    for cmd_info in _COMMAND_DEFINITIONS:
        # Skip /llm command in GUI mode since GUI has its own LLM selector
        if gui_mode and '/llm' in cmd_info['aliases']:
            continue

        aliases = ', '.join(cmd_info['aliases'])
        # Color the aliases in green and descriptions in white for better readability
        aliases_colored = colored(aliases.ljust(CMD_WIDTH), 'green', attrs=['bold'])
        description_colored = colored(cmd_info['description'], 'white')
        line = f"  {aliases_colored}{description_colored}"
        print(line)