# tools/search.py
"""
This file implements the 'search' tool. It allows the agent to search for a
pattern within file contents, filenames, and directory names in a specified path.
"""
import os
import re

class search:
    MAX_RESULTS = 100 # To prevent overwhelming the context window

    @staticmethod
    def get_delim() -> str:
        return 'search'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "search",
            "description": f"Recursively searches for a regex pattern in file contents and names, starting from a given path. Defaults to the current directory. Stops after {search.MAX_RESULTS} matches.",
            "example": "<search><pattern>my_function</pattern><path>./src</path></search>"
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML-like input to extract a search pattern and an optional path.
        Performs a recursive search and returns a formatted list of matches.
        """
        try:
            pattern_match = re.search(r'<pattern>(.*?)</pattern>', content, re.DOTALL)
            path_match = re.search(r'<path>(.*?)</path>', content, re.DOTALL)

            if not pattern_match:
                return "Error: Invalid format. Input must contain a <pattern> tag."

            pattern = pattern_match.group(1).strip()
            search_path = path_match.group(1).strip() if path_match else '.'

            if not pattern:
                return "Error: Search pattern cannot be empty."

            try:
                regex = re.compile(pattern)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"

            full_path = os.path.abspath(search_path)

            if not os.path.exists(full_path):
                return f"Error: Path does not exist: '{full_path}'"

            matches = []
            
            # If the path is a file, just search that one file
            if os.path.isfile(full_path):
                search._search_in_file(full_path, regex, matches)
            
            # If the path is a directory, walk through it
            elif os.path.isdir(full_path):
                for root, dirs, files in os.walk(full_path, topdown=True):
                    # Prune common large/binary directories to speed up search
                    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.venv', 'build', 'dist']]
                    
                    if len(matches) >= search.MAX_RESULTS: break

                    # Search in directory names
                    for d in dirs:
                        if len(matches) >= search.MAX_RESULTS: break
                        if regex.search(d):
                            matches.append(f"DIR:  {os.path.join(root, d)}")
                    
                    # Search in file names and contents
                    for f in files:
                        if len(matches) >= search.MAX_RESULTS: break
                        file_path = os.path.join(root, f)
                        search._search_in_file(file_path, regex, matches)
            else:
                return f"Error: Path is not a file or directory: '{full_path}'"

            if not matches:
                return "No matches found."
            
            result_str = "\n".join(matches)
            if len(matches) >= search.MAX_RESULTS:
                result_str += f"\n... (Search stopped after reaching {search.MAX_RESULTS} results limit)"
                
            return result_str

        except Exception as e:
            return f"Error executing search tool: {str(e)}"
            
    @staticmethod
    def _search_in_file(file_path: str, regex: re.Pattern, matches: list):
        """Helper function to search a single file's name and content."""
        if len(matches) >= search.MAX_RESULTS:
            return

        # Check filename
        if regex.search(os.path.basename(file_path)):
            matches.append(f"FILE: {file_path}")
            if len(matches) >= search.MAX_RESULTS: return
        
        # Check content, avoiding large binary files
        try:
            if os.path.getsize(file_path) > 2 * 1024 * 1024: # Skip files > 2MB
                return
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    if len(matches) >= search.MAX_RESULTS: return
                    if regex.search(line):
                        matches.append(f"LINE: {file_path}:{i}: {line.strip()}")
        except (OSError, UnicodeDecodeError):
            pass # Ignore read errors for binary files, permissions issues etc.