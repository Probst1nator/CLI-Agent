"""
Configuration utilities for CLI-Agent.

This module extracts LLM and Monte Carlo configuration functions from main.py
into a reusable component that maintains the exact same logic.
"""

import logging
from typing import List
from termcolor import colored

from core.globals import g


def get_provider_emoji_and_name(model_key: str) -> tuple[str, str]:
    """
    Get emoji and provider name for a model key.

    This function is imported from main.py to maintain exact compatibility.
    For the refactored version, we import it from main.py to avoid duplication.
    """
    # Import from main to maintain exact logic
    from main import get_provider_emoji_and_name as main_get_provider_emoji_and_name
    return main_get_provider_emoji_and_name(model_key)


def colorize_model_list(model_list: List[str], tui_mode: bool = False) -> str:
    """
    Colorize a list of model names.

    This function is imported from main.py to maintain exact compatibility.
    """
    # Import from main to maintain exact logic
    from main import colorize_model_list as main_colorize_model_list
    return main_colorize_model_list(model_list, tui_mode)


def apply_default_llm_config() -> None:
    """
    Loads the cached LLM configuration from disk and applies it as the default
    for the current session. This makes LLM selections persistent.

    Extracted from main.py with exact same logic.
    """
    llm_config = g.get_llm_config()
    if not llm_config:
        return # No saved config to apply

    # Preserve chronological selection order by using g.SELECTED_LLMS if available
    if g.SELECTED_LLMS:
        selected_llms = [key for key in g.SELECTED_LLMS if llm_config.get(key, {}).get('selected')]
    else:
        selected_llms = [key for key, data in llm_config.items() if data.get('selected')]


    # Only apply if there are actual selections, to not override command line flags unintentionally
    if selected_llms:
        g.SELECTED_LLMS = selected_llms
        # Count unique models by provider
        unique_models = list(set(selected_llms))  # Deduplicate
        provider_counts = {}
        for model in unique_models:
            emoji, provider_name = get_provider_emoji_and_name(model)
            provider_key = f"{emoji} {provider_name}"
            provider_counts[provider_key] = provider_counts.get(provider_key, 0) + 1

        provider_summary = ", ".join([f"{provider}: {count}" for provider, count in provider_counts.items()])
        logging.info(f"📋 Loaded latest LLM configuration. Total: {colored(str(len(unique_models)), 'green')} models ({provider_summary})")


