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


def create_ascii_header(text: str) -> str:
    """
    Create ASCII art header with lazy pyfiglet import.
    
    This avoids loading the heavy pyfiglet module unless the header
    is actually needed.
    """
    try:
        from pyfiglet import figlet_format
        return figlet_format(text, font="slant")
    except ImportError:
        # Graceful fallback if pyfiglet not available
        return f"\n=== {text} ===\n"


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


def process_data_with_pandas(data: list) -> Optional[Any]:
    """
    Process data with lazy pandas import.
    
    Pandas is a heavy library that should only be loaded when
    data processing is actually needed.
    """
    try:
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame(data)
        return df.describe()
    except ImportError:
        print("Pandas/Numpy not available for data processing")
        return None


def lazy_audio_processing() -> Optional[Any]:
    """
    Audio processing with lazy imports.
    
    Audio libraries are often heavy and should only be loaded
    when audio functionality is used.
    """
    try:
        # Import audio libraries only when needed
        import librosa
        import soundfile as sf
        
        return {"librosa": librosa, "soundfile": sf}
    except ImportError:
        print("Audio libraries not available")
        return None


# Demonstration of import time optimization
class LazyImportOptimizer:
    """
    Class demonstrating lazy import patterns for CLI-Agent.
    
    This shows how to structure code to minimize startup time while
    maintaining full functionality when needed.
    """
    
    def __init__(self):
        # Only store lightweight config, no heavy imports
        self._prompt_session = None
        self._ml_models = {}
        
    def get_interactive_session(self):
        """Get interactive session with lazy loading."""
        if self._prompt_session is None:
            self._prompt_session = get_prompt_session()
        return self._prompt_session
        
    def get_ml_model(self, task: str):
        """Get ML model with caching and lazy loading."""
        if task not in self._ml_models:
            self._ml_models[task] = get_ml_pipeline(task)
        return self._ml_models[task]
        
    def process_visualization(self, data):
        """Process visualization with lazy matplotlib import."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data)
            return plt
        except ImportError:
            print("Visualization libraries not available")
            return None


# Example usage in startup optimization
STARTUP_OPTIMIZATIONS = {
    "prompt_toolkit": "Import only when interactive input needed (~100ms saved)",
    "pyfiglet": "Import only when ASCII art needed (~50ms saved)", 
    "transformers": "Import only when ML features used (~500ms saved)",
    "pandas/numpy": "Import only when data processing needed (~200ms saved)",
    "matplotlib/seaborn": "Import only when visualization needed (~300ms saved)",
    "librosa/audio": "Import only when audio processing needed (~400ms saved)"
}

def demonstrate_startup_savings():
    """Demonstrate potential startup time savings."""
    print("Lazy Import Optimization Benefits:")
    total_savings = 0
    
    for library, savings_info in STARTUP_OPTIMIZATIONS.items():
        print(f"  • {library}: {savings_info}")
        # Extract milliseconds from description for rough total
        if "ms saved" in savings_info:
            ms = int(savings_info.split("(")[1].split("ms")[0].replace("~", ""))
            total_savings += ms
    
    print(f"\nTotal potential startup time savings: ~{total_savings}ms ({total_savings/1000:.1f}s)")
    print("This makes CLI-Agent much more responsive for simple operations!")