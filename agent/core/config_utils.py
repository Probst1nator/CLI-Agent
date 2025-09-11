"""
Configuration utilities for CLI-Agent.

This module extracts LLM and Monte Carlo configuration functions from main.py
into a reusable component that maintains the exact same logic.
"""

import json
import logging
import os
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


def apply_monte_carlo_config() -> None:
    """
    Loads or creates a Monte Carlo configuration with branching, evaluation, and guarding.
    If no config exists, creates a default with the first 3 available models.
    
    Extracted from main.py with exact same logic.
    """
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