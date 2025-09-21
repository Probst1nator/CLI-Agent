"""
Lazy import optimization for CLI-Agent.

This module demonstrates how to optimize startup time by moving heavy
imports inside the functions that use them, rather than at module level.
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

# Only import types for static analysis, not at runtime
if TYPE_CHECKING:
    from prompt_toolkit import PromptSession


def get_prompt_session() -> 'PromptSession':
    """
    Get prompt session with lazy imports.

    This avoids loading prompt_toolkit at startup unless interactive
    input is actually needed.
    """
    # Import only when needed - saves ~100ms startup time
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    import os

    # Create session with lazy-loaded components
    history = FileHistory(os.path.expanduser('~/.cli_agent_history'))
    return PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory()
    )


def get_ml_pipeline(task: str) -> Optional[Any]:
    """
    Get ML pipeline with lazy transformers import.

    This is a good example of lazy loading heavy ML libraries
    that are only needed for specific features.
    """
    try:
        # Import only when ML functionality is actually used
        from transformers import pipeline
        return pipeline(task)
    except ImportError:
        print("Transformers not available for ML pipeline")
        return None
    except Exception as e:
        print(f"Error creating ML pipeline: {e}")
        return None

