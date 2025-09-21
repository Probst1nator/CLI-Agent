"""
Shared utilities for all CLI-Agent tools

Organized by functionality for clean imports.
"""

# Web utilities
try:
    from .web.web_server import WebServer
except ImportError:
    WebServer = None

# Search utilities
try:
    from .search.brave_search import BraveSearchAPI
except ImportError:
    BraveSearchAPI = None

# Python utilities
try:
    from .python.python_utils import *
except ImportError:
    pass

__all__ = ['WebServer', 'BraveSearchAPI']