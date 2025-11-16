# tools/grepsearch.py
"""
Tool to search for patterns within file contents, filenames, and directory names.
"""
import os
import re

class grepsearch:
    MAX_RESULTS = 100  # Prevent overwhelming the context window

    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'grepsearch'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "grepsearch",
            "description": (
                "Search for regex patterns in file contents, filenames, and directories. "
                "Respects common ignore patterns (node_modules, .git, etc.). "
                "Optional <mode> (all/files/directories) and <max_results>."
            ),
            "example": (
                "<grepsearch><pattern>my_function</pattern><path>./src</path></grepsearch>\n"
                "<grepsearch><pattern>TODO</pattern><path>.</path><max_results>50</max_results></grepsearch>"
            )
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML input to extract a search pattern, path, and mode.
        Performs a recursive search and returns a formatted list of matches.
        """
        try:
            pattern_match = re.search(r'<pattern>(.*?)</pattern>', content, re.DOTALL)
            path_match = re.search(r'<path>(.*?)</path>', content, re.DOTALL)
            mode_match = re.search(r'<mode>(.*?)</mode>', content, re.DOTALL)
            max_results_match = re.search(r'<max_results>(.*?)</max_results>', content, re.DOTALL)

            if not pattern_match:
                return "Error: Invalid format. Input must contain a <pattern> tag."

            pattern = pattern_match.group(1).strip()
            search_path = path_match.group(1).strip() if path_match else '.'
            search_mode = mode_match.group(1).strip().lower() if mode_match else 'all'
            max_results = int(max_results_match.group(1).strip()) if max_results_match else grepsearch.MAX_RESULTS

            if search_mode not in ['all', 'files', 'directories']:
                return f"Error: Invalid mode '{search_mode}'. Must be one of 'all', 'files', 'directories'."

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

            if os.path.isfile(full_path):
                if search_mode in ('all', 'files'):
                    grepsearch._search_in_file(full_path, regex, matches, max_results)

            elif os.path.isdir(full_path):
                for root, dirs, files in os.walk(full_path, topdown=True):
                    # Filter out common ignored directories
                    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.venv', 'build', 'dist', '.pytest_cache']]

                    if len(matches) >= max_results:
                        break

                    if search_mode in ('all', 'directories'):
                        for d in dirs:
                            if len(matches) >= max_results:
                                break
                            if regex.search(d):
                                matches.append(f"DIR:  {os.path.join(root, d)}")

                    if search_mode in ('all', 'files'):
                        for f in files:
                            if len(matches) >= max_results:
                                break
                            file_path = os.path.join(root, f)
                            grepsearch._search_in_file(file_path, regex, matches, max_results)
            else:
                return f"Error: Path is not a file or directory: '{full_path}'"

            if not matches:
                return "No matches found."

            result_str = "\n".join(matches)
            if len(matches) >= max_results:
                result_str += f"\n... (Search stopped after reaching {max_results} results limit)"

            return result_str

        except Exception as e:
            return f"Error executing grepsearch tool: {str(e)}"

    @staticmethod
    def _search_in_file(file_path: str, regex: re.Pattern, matches: list, max_results: int):
        """Search within a file for pattern matches."""
        if len(matches) >= max_results:
            return

        # Check filename
        if regex.search(os.path.basename(file_path)):
            matches.append(f"FILE: {file_path}")
            if len(matches) >= max_results:
                return

        # Search file contents
        try:
            # Skip large files
            if os.path.getsize(file_path) > 2 * 1024 * 1024:
                return

            with open(file_path, encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    if len(matches) >= max_results:
                        return
                    if regex.search(line):
                        matches.append(f"LINE: {file_path}:{i}: {line.strip()}")
        except (OSError, UnicodeDecodeError):
            pass
