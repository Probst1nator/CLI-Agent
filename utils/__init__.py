# utils/__init__.py

# --- Simple, Deterministic Tools ---
from .editfile import EditFile 
from .removefile import RemoveFile
from .viewfiles import ViewFiles
from .todos import TodosUtil
from .web_fetch import WebFetchUtil

# --- Other Utilities ---
# Make sure other tools like SearchWeb, GenerateImage, etc.,
# are also updated to return structured JSON.

from .searchweb import SearchWeb
from .generateimage import GenerateImage
from .viewimage import ViewImage

# --- Example/Reference ---
