# tools/readdirectory.py
"""
This file implements the 'readdirectory' tool. It recursively lists
the contents of a directory in a tree structure, showing all files
and directories without reading their contents.
"""
import os
import re

from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)

# Attempt to import pathspec for .gitignore handling
try:
    import pathspec
except ImportError:
    pathspec = None


class readdirectory:
    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool."""
        return 'readdirectory'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "readdirectory",
            "description": "Recursively lists directory contents in a tree structure. Shows all files and subdirectories. Respects .gitignore rules. Optional max_depth parameter to limit recursion depth. Does NOT read file contents.",
            "example": "<readdirectory><path>./documents</path><max_depth>2</max_depth></readdirectory>"
        }

    @staticmethod
    def _find_project_root_and_spec(start_path: str) -> tuple:
        """
        Finds the project root (.git) and parses .gitignore files.
        Returns (pathspec, project_root) or (None, None).
        """
        if not pathspec:
            return None, None

        gitignore_patterns = []
        project_root = None
        current_path = os.path.abspath(start_path)

        while True:
            if os.path.isdir(os.path.join(current_path, '.git')):
                project_root = current_path

            gitignore_path = os.path.join(current_path, '.gitignore')
            if os.path.isfile(gitignore_path):
                try:
                    with open(gitignore_path, encoding='utf-8') as f:
                        gitignore_patterns = f.readlines() + gitignore_patterns
                except Exception as e:
                    logger.warning(f"Could not read .gitignore at {gitignore_path}: {e}")

            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                break

            current_path = parent_path
            if project_root and not current_path.startswith(project_root):
                break

        if gitignore_patterns:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_patterns)
            return spec, project_root

        return None, project_root

    @staticmethod
    def run(content: str) -> str:
        """
        Executes the directory scan and returns a tree structure.
        """
        try:
            # Parse path parameter
            path_match = re.search(r'<path>(.*?)</path>', content, re.DOTALL)
            search_path = path_match.group(1).strip() if path_match else '.'

            # Parse optional max_depth parameter
            max_depth = None
            max_depth_match = re.search(r'<max_depth>(\d+)</max_depth>', content)
            if max_depth_match:
                max_depth = int(max_depth_match.group(1))

            full_path = os.path.abspath(search_path)
            if not os.path.isdir(full_path):
                return f"Error: Path does not exist or is not a directory: '{full_path}'"

            logger.info(f"Listing directory: {full_path} (max_depth={max_depth})")

            spec, project_root = readdirectory._find_project_root_and_spec(full_path)

            # Collect all paths
            all_entries = []
            total_files = 0
            total_dirs = 0

            for root, dirs, files in os.walk(full_path, topdown=True):
                # Calculate current depth relative to the starting path
                current_depth = root[len(full_path):].count(os.sep)

                # If max_depth is set and we've reached it, don't descend further
                if max_depth is not None and current_depth >= max_depth:
                    dirs[:] = []  # Clear dirs list to prevent further recursion
                    continue

                # Prune ignored directories
                if spec and project_root:
                    rel_root = os.path.relpath(root, project_root) if project_root else root
                    paths_to_check_dirs = [os.path.join(rel_root, d).replace('\\', '/') for d in dirs]
                    ignored_dirs = {os.path.basename(p) for p in spec.match_files(paths_to_check_dirs)}
                    dirs[:] = [d for d in dirs if d not in ignored_dirs and d != '.git']
                else:
                    # Default exclusions when no .gitignore
                    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.venv', '.vscode']]

                # Add directories
                for d in sorted(dirs):
                    dir_path = os.path.join(root, d)
                    rel_path = os.path.relpath(dir_path, full_path)
                    depth = rel_path.count(os.sep)
                    all_entries.append((rel_path, depth, True))  # True = is_dir
                    total_dirs += 1

                # Add files
                for filename in sorted(files):
                    file_path = os.path.join(root, filename)

                    # Check if file is ignored by .gitignore
                    is_ignored = False
                    if spec and project_root:
                        rel_path = os.path.relpath(file_path, project_root).replace('\\', '/')
                        if spec.match_file(rel_path):
                            is_ignored = True

                    if is_ignored:
                        continue

                    rel_path = os.path.relpath(file_path, full_path)
                    depth = rel_path.count(os.sep)
                    all_entries.append((rel_path, depth, False))  # False = is_file
                    total_files += 1

            # Format the output
            output = [f"Directory listing of '{full_path}'"]
            if max_depth is not None:
                output.append(f"Max depth: {max_depth}")
            output.append(f"Total: {total_dirs} directories, {total_files} files")
            output.append("")
            output.append("=" * 80)

            # Build tree structure
            if not all_entries:
                output.append("(empty directory)")
            else:
                # Sort entries by path for consistent output
                all_entries.sort(key=lambda x: x[0])

                for rel_path, depth, is_dir in all_entries:
                    # Calculate indentation
                    indent = "  " * depth
                    marker = "└─ " if depth > 0 else ""

                    # Add directory marker
                    if is_dir:
                        display_name = os.path.basename(rel_path) + "/"
                    else:
                        display_name = os.path.basename(rel_path)

                    output.append(f"{indent}{marker}{display_name}")

            logger.info(f"Listed {total_files} files and {total_dirs} directories")
            return "\n".join(output)

        except Exception as e:
            error_msg = f"Error executing readdirectory tool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
