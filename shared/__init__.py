"""
Shared utilities for CLI-Agent tools

This module provides common utilities and helper functions that can be
used across all CLI-Agent tools and projects.
"""

# Import from new shared location and legacy py_methods for compatibility
try:
    from .dia_helper import get_dia_model
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py_methods'))
        from dia_helper import get_dia_model
    except ImportError:
        get_dia_model = None

try:
    from .common_utils import extract_blocks
except ImportError:
    try:
        from utils import extract_blocks
    except ImportError:
        extract_blocks = None

# Import from tools/file_copier for AIFixPath
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'file_copier'))
    from ai_path_finder import AIFixPath
except ImportError:
    AIFixPath = None

# Import path resolver
try:
    from .path_resolver import PathResolver, setup_cli_agent_imports
except ImportError:
    PathResolver = None
    setup_cli_agent_imports = None

# Define what gets imported with "from shared import *"
__all__ = ['get_dia_model', 'AIFixPath', 'extract_blocks', 'PathResolver', 'setup_cli_agent_imports']