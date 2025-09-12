# utils/__init__.py

# --- Simple, Deterministic Tools ---
from .updatefile import UpdateFile 
from .viewfiles import ViewFiles
# RemoveFile module doesn't exist - keeping commented for now
# from .removefile import RemoveFile
# TodosUtil temporarily disabled to fix startup hang
# TODO: Fix circular import issue with deprecated todos module
import os  # Keep os import for project_root calculation
# import importlib.util
# _todos_spec = importlib.util.spec_from_file_location(
#     "todos", 
#     os.path.join(os.path.dirname(__file__), ".deprecated", "todos.py")
# )
# _todos_module = importlib.util.module_from_spec(_todos_spec)
# _todos_spec.loader.exec_module(_todos_module)
# TodosUtil = _todos_module.TodosUtil
TodosUtil = None  # Placeholder
from .web_fetch import WebFetchUtil

# --- Other Utilities ---
# Make sure other tools like SearchWeb, GenerateImage, etc.,
# are also updated to return structured JSON.

import sys
# Add project root to sys.path for absolute imports  
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.searchweb import searchweb as SearchWeb
from .viewimage import ViewImage

# --- Example/Reference ---
