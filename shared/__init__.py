"""
Shared utilities for CLI-Agent tools

This module provides common utilities and helper functions that can be
used across all CLI-Agent tools and projects.
"""

# Core shared utilities
from .audio import *
from .common_utils import extract_blocks
from .dia_helper import get_dia_model
from .path_resolver import PathResolver

# Import from structured utilities
from .utils import BraveSearchAPI, WebServer

# Define what gets imported with "from shared import *"
__all__ = ['get_dia_model', 'extract_blocks', 'PathResolver', 'WebServer', 'BraveSearchAPI']