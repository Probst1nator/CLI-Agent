# tools/grepsearch.py
"""
This file implements the 'grepsearch' tool. It allows the agent to search for a
pattern within file contents, filenames, and directory names in a specified path.
It automatically respects .gitignore rules to provide more relevant results.
"""
import os
import re

try:
    import pathspec
except ImportError:
    pathspec = None

class grepsearch:
    MAX_RESULTS = 100 # To prevent overwhelming the context window

    @staticmethod
    def get_delim() -> str:
        return 'grepsearch'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "grepsearch",
            "description": (
                f"Recursively searches for a regex pattern in file/directory names and file contents, starting from a given path. "
                f"Defaults to the current directory. Automatically ignores files and directories specified in .gitignore files. "
                f"Stops after {grepsearch.MAX_RESULTS} matches."
            ),
            "example": (
                "<grepsearch>"
                "<pattern>my_function</pattern>"
                "<path>./src</path>"
                "<mode>files</mode>" # Optional: 'all' (default), 'files', or 'directories'
                "</grepsearch>"
            )
        }

    @staticmethod
    def _find_project_root_and_spec(start_path: str) -> 'Tuple[object | None, str | None]':
        """
        Finds the project root (.git directory) and parses all .gitignore files
        from the start_path up to that root.
        Returns a tuple of (PathSpec object, project_root_path).
        """
        if not pathspec:
            return None, None

        gitignore_patterns = []
        project_root = None
        current_path = os.path.abspath(start_path)

        # Walk up to the root of the filesystem
        while True:
            # Check for .git directory to identify the project root
            if os.path.isdir(os.path.join(current_path, '.git')):
                project_root = current_path

            gitignore_path = os.path.join(current_path, '.gitignore')
            if os.path.isfile(gitignore_path):
                with open(gitignore_path, encoding='utf-8') as f:
                    # Prepend patterns from higher-level .gitignores
                    gitignore_patterns = f.readlines() + gitignore_patterns

            parent_path = os.path.dirname(current_path)
            if parent_path == current_path: # Reached filesystem root
                break

            current_path = parent_path
            if project_root: # Stop once we're outside the project root
                break

        if gitignore_patterns:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_patterns)
            return spec, project_root

        return None, project_root

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML-like input to extract a search pattern, path, and mode.
        Performs a recursive search and returns a formatted list of matches.
        """
        try:
            pattern_match = re.search(r'<pattern>(.*?)</pattern>', content, re.DOTALL)
            path_match = re.search(r'<path>(.*?)</path>', content, re.DOTALL)
            mode_match = re.search(r'<mode>(.*?)</mode>', content, re.DOTALL)

            if not pattern_match:
                return "Error: Invalid format. Input must contain a <pattern> tag."

            pattern = pattern_match.group(1).strip()
            search_path = path_match.group(1).strip() if path_match else '.'
            search_mode = mode_match.group(1).strip().lower() if mode_match else 'all'

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

            spec, project_root = grepsearch._find_project_root_and_spec(full_path)
            if pathspec is None:
                print("Warning: 'pathspec' library not found. Running grepsearch without .gitignore filtering.")

            matches = []

            if os.path.isfile(full_path):
                is_ignored = False
                if spec and project_root:
                    rel_path = os.path.relpath(full_path, project_root)
                    if spec.match_file(rel_path):
                        is_ignored = True

                if is_ignored:
                    return "No matches found (file is ignored by .gitignore)."

                if search_mode in ('all', 'files'):
                    grepsearch._search_in_file(full_path, regex, matches)

            elif os.path.isdir(full_path):
                for root, dirs, files in os.walk(full_path, topdown=True):
                    if spec and project_root:
                        # Filter directories and files using .gitignore spec
                        # We must use paths relative to the project root for matching
                        paths_to_check_dirs = [os.path.relpath(os.path.join(root, d), project_root) for d in dirs]
                        paths_to_check_files = [os.path.relpath(os.path.join(root, f), project_root) for f in files]

                        ignored_dirs = {os.path.basename(p) for p in spec.match_files(paths_to_check_dirs)}
                        ignored_files = {os.path.basename(p) for p in spec.match_files(paths_to_check_files)}

                        dirs[:] = [d for d in dirs if d not in ignored_dirs and d not in ['.git']]
                        files = [f for f in files if f not in ignored_files]
                    else:
                        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.venv', 'build', 'dist']]

                    if len(matches) >= grepsearch.MAX_RESULTS: break

                    if search_mode in ('all', 'directories'):
                        for d in dirs:
                            if len(matches) >= grepsearch.MAX_RESULTS: break
                            if regex.search(d):
                                matches.append(f"DIR:  {os.path.join(root, d)}")

                    if search_mode in ('all', 'files'):
                        for f in files:
                            if len(matches) >= grepsearch.MAX_RESULTS: break
                            file_path = os.path.join(root, f)
                            grepsearch._search_in_file(file_path, regex, matches)
            else:
                return f"Error: Path is not a file or directory: '{full_path}'"

            if not matches:
                return "No matches found."

            result_str = "\n".join(matches)
            if len(matches) >= grepsearch.MAX_RESULTS:
                result_str += f"\n... (Search stopped after reaching {grepsearch.MAX_RESULTS} results limit)"

            return result_str

        except Exception as e:
            return f"Error executing grepsearch tool: {str(e)}"

    @staticmethod
    def _search_in_file(file_path: str, regex: re.Pattern, matches: list):
        if len(matches) >= grepsearch.MAX_RESULTS:
            return

        # Check filename (only search content if mode is 'all' or 'files')
        if regex.search(os.path.basename(file_path)):
            matches.append(f"FILE: {file_path}")
            if len(matches) >= grepsearch.MAX_RESULTS: return

        try:
            if os.path.getsize(file_path) > 2 * 1024 * 1024:
                return
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    if len(matches) >= grepsearch.MAX_RESULTS: return
                    if regex.search(line):
                        matches.append(f"LINE: {file_path}:{i}: {line.strip()}")
        except (OSError, UnicodeDecodeError):
            pass