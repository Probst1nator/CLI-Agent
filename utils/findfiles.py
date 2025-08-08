import os
import fnmatch
import json
from typing import Optional, Set, Dict, Any

from py_classes.cls_util_base import UtilBase

class FindFiles(UtilBase):
    """
    A quick and strong utility to find files and folders by name,
    and to find files by their content. It intelligently ignores
    common non-source directories like .git, __pycache__, etc.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["search files", "find file", "locate file", "search content", "file system", "directory search", "wildcard search", "config file", "dockerfile", "requirements.txt", "model files", "weights", "checkpoint", "log files", "error logs", "dependency", "library", "module", "script", "codebase analysis"],
            "use_cases": [
                "Find all python files in the current project.",
                "Search for the term 'API_KEY' in all configuration files.",
                "Locate the 'main.py' file.",
                "Find all files modified in the last 24 hours containing 'error'.",
                "Find all Dockerfiles in the repository.",
                "Search for model weight files (.safetensors, .bin) in the project.",
                "Locate configuration files that contain 'ollama' settings.",
                "Find all requirements.txt files and their dependencies."
            ],
            "arguments": {
                "search_string": "The pattern or text to search for. Supports wildcards like '*.py'.",
                "location": "The directory path to start the search from. Defaults to the current directory."
            }
        }

    # Directories to ignore during the search to improve speed and relevance.
    _IGNORE_DIRS = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        'dist', 'build', 'env', '.env', '.vscode', '.idea'
    }
    
    # File extensions to skip during content search to avoid reading large binaries.
    _BINARY_EXTENSIONS = {
        '.exe', '.dll', '.so', '.o', '.a', '.lib', '.jar', '.war', '.ear',
        '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico',
        '.mp3', '.wav', '.flac', '.ogg', '.aac',
        '.mp4', '.avi', '.mov', '.mkv', '.webm',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.db', '.sqlite', '.sqlite3', '.db3',
        '.pyc', '.pyo'
    }

    @staticmethod
    def _is_binary_file(file_path: str) -> bool:
        """
        Checks if a file is likely a binary by its extension or by sniffing
        the first few bytes for a null character.
        """
        # First, check by extension for a quick pass
        _, ext = os.path.splitext(file_path)
        if ext.lower() in FindFiles._BINARY_EXTENSIONS:
            return True
        
        # If extension is not definitive, sniff the file content
        try:
            with open(file_path, 'rb') as f:
                # Reading the first 1024 bytes is usually enough
                chunk = f.read(1024)
                if b'\x00' in chunk:
                    return True
        except (IOError, OSError):
            # If we can't read the file, treat it as inaccessible/binary
            return True
        return False

    @staticmethod
    def run(search_string: str, location: Optional[str] = ".") -> str:
        """
        Searches for files and folders by name, and files by content.
        The search is case-insensitive and supports wildcards (*, ?) for names.

        Args:
            search_string (str): The string to search for. Can be a pattern for names
                                 (e.g., '*.py', 'test_*') or a literal string for content.
            location (Optional[str]): The absolute or relative path to start searching from.
                                      Defaults to the current working directory.

        Returns:
            str: A JSON string with a 'result' key containing the list of found paths,
                 or an 'error' key on failure.
        """
        try:
            start_path = os.path.abspath(location)
            if not os.path.isdir(start_path):
                return json.dumps({
                    "error": f"Search location does not exist or is not a directory: '{start_path}'"
                }, indent=2)

            found_paths: Set[str] = set()
            lower_search_string = search_string.lower()
            
            # Using os.walk for a comprehensive traversal
            for root, dirs, files in os.walk(start_path, topdown=True):
                # Prune the directories to search, improving performance
                dirs[:] = [d for d in dirs if d not in FindFiles._IGNORE_DIRS]

                # 1. Search for matching directory names
                for dirname in dirs:
                    if fnmatch.fnmatch(dirname.lower(), lower_search_string):
                        dir_path = os.path.join(root, dirname)
                        found_paths.add(os.path.abspath(dir_path))

                # 2. Search for matching file names and file content
                for filename in files:
                    full_path = os.path.join(root, filename)
                    
                    # Search by file name first
                    if fnmatch.fnmatch(filename.lower(), lower_search_string):
                        found_paths.add(os.path.abspath(full_path))
                        continue  # Already found, no need to check content

                    # Search by content if not found by name
                    if FindFiles._is_binary_file(full_path):
                        continue

                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            # Read line by line to be memory efficient
                            for line in f:
                                if lower_search_string in line.lower():
                                    found_paths.add(os.path.abspath(full_path))
                                    break # File matched, move to the next file
                    except (IOError, OSError):
                        # Ignore files that can't be read
                        pass
            
            results_list = sorted(list(found_paths))
            
            if not results_list:
                message = f"No files or folders found matching '{search_string}' in '{start_path}'."
            else:
                message = f"Found {len(results_list)} item(s) matching '{search_string}' in '{start_path}'."
            
            result = {
                "result": {
                    "status": "Success",
                    "message": message,
                    "search_string": search_string,
                    "location": start_path,
                    "matches_found": len(results_list),
                    "paths": results_list
                }
            }
            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred during search: {str(e)}"}, indent=2)