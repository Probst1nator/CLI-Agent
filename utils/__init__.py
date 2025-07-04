# utils/__init__.py

# --- Simple, Deterministic Tools ---
from .authorfile import AuthorFile 
from .todos import TodosUtil

# --- Other Utilities ---
# Make sure other tools like SearchWeb, GenerateImage, etc.,
# are also updated to return structured JSON.

from .searchweb import SearchWeb
from .generateimage import GenerateImage
from .viewimage import ViewImage

# --- Example/Reference ---
from ._example_util import ExampleUtil
