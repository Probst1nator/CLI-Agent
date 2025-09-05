"""
Modern agent infrastructure

Core components for the CLI-Agent system organized by functionality.
"""

# Agent core components
from .notebook.computational_notebook import ComputationalNotebook
from .playbook.playbook_manager import PlaybookManager
from .utils_manager.utils_manager import UtilsManager
from .text_painter.stream_painter import TextStreamPainter
from .llm_selection.llm_selector import LlmSelector
from .utils_manager.util_base import UtilBase

__all__ = [
    'ComputationalNotebook',
    'PlaybookManager', 
    'UtilsManager',
    'TextStreamPainter',
    'LlmSelector',
    'UtilBase'
]