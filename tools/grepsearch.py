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
                "Search for regex patterns in file contents, filenames, and directories. "
                "Respects .gitignore rules. Most common: <pattern>function_name</pattern><path>./src</path>. "
                f"Use <help>grepsearch</help> for modes and regex syntax."
            ),
            "example": (
                "<grepsearch><pattern>my_function</pattern><path>./src</path></grepsearch>\n"
                "<grepsearch><pattern>class.*Config</pattern><path>.</path><mode>files</mode></grepsearch>"
            )
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        GREPSEARCH TOOL - EXTENDED HELP                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  <grepsearch>
    <pattern>your_regex_pattern</pattern>
    <path>./directory</path>
    <mode>all</mode>
  </grepsearch>

PURPOSE:
Recursively search for regex patterns in:
- File contents (showing matching lines with line numbers)
- File names
- Directory names

Automatically respects .gitignore rules to filter out irrelevant files.

PARAMETERS:

  <pattern> (REQUIRED)
    - Regular expression pattern to search for
    - Supports full Python regex syntax
    - Case-sensitive by default
    - Examples:
      <pattern>def my_function</pattern>
      <pattern>class.*Config</pattern>
      <pattern>TODO|FIXME|HACK</pattern>

  <path> (OPTIONAL, default: ".")
    - Starting directory or file path
    - Can be absolute or relative
    - Defaults to current directory
    - Examples:
      <path>./src</path>
      <path>/home/user/project</path>
      <path>./specific_file.py</path>

  <mode> (OPTIONAL, default: "all")
    - Search scope filter
    - Three modes available:

    1. 'all' (default) - Search everything
       - File contents (shows matching lines)
       - File names
       - Directory names

    2. 'files' - Search only files
       - File contents and file names
       - Skips directory name matching

    3. 'directories' - Search only directory names
       - Only matches directory names
       - Ignores file contents and names

REGEX PATTERN EXAMPLES:

1. EXACT MATCH:
   <pattern>my_function</pattern>
   Matches: "my_function", "def my_function()", "call my_function()"

2. WORD BOUNDARY:
   <pattern>\\bconfig\\b</pattern>
   Matches: "config", "get config", "config ="
   Not: "configuration", "reconfig"

3. CASE INSENSITIVE:
   <pattern>(?i)error</pattern>
   Matches: "error", "Error", "ERROR", "ErRoR"

4. MULTIPLE OPTIONS:
   <pattern>TODO|FIXME|HACK</pattern>
   Matches any of: "TODO", "FIXME", "HACK"

5. CLASS/FUNCTION DEFINITIONS:
   <pattern>(class|def)\\s+\\w+</pattern>
   Matches: "class MyClass", "def my_function"

6. IMPORT STATEMENTS:
   <pattern>^import|^from.*import</pattern>
   Matches: "import os", "from typing import List"

7. STRING LITERALS:
   <pattern>"[^"]*database[^"]*"</pattern>
   Matches: "database_url", "connect to database"

GITIGNORE BEHAVIOR:

Automatically filters out files/directories matching .gitignore patterns:
- Reads all .gitignore files from search path up to project root
- Identifies project root by presence of .git directory
- Applies standard gitignore matching rules

Common ignored patterns:
  - node_modules/
  - __pycache__/
  - .venv/, venv/
  - build/, dist/
  - *.pyc, *.log
  - .git/

If pathspec library not installed:
  - Falls back to basic filtering (node_modules, __pycache__, etc.)
  - Warning message displayed

RESULT FORMAT:

Three types of matches:

1. DIRECTORY MATCH:
   DIR:  /path/to/matching_directory

2. FILENAME MATCH:
   FILE: /path/to/matching_file.py

3. CONTENT MATCH:
   LINE: /path/to/file.py:42: matching line content here

Results limited to {grepsearch.MAX_RESULTS} matches to prevent overwhelming output.

SEARCH MODES IN DETAIL:

MODE: 'all' (Most comprehensive)
  <grepsearch>
    <pattern>config</pattern>
    <path>./src</path>
    <mode>all</mode>
  </grepsearch>

  Returns:
  - Directories named "config" or containing "config"
  - Files named "config.py", "my_config.json", etc.
  - Lines in files containing "config"

MODE: 'files' (Code search)
  <grepsearch>
    <pattern>def parse_</pattern>
    <path>./src</path>
    <mode>files</mode>
  </grepsearch>

  Returns:
  - Files with "parse_" in name
  - Lines in files with function definitions like "def parse_json()"
  - Ignores directory names

MODE: 'directories' (Structure exploration)
  <grepsearch>
    <pattern>test</pattern>
    <path>.</path>
    <mode>directories</mode>
  </grepsearch>

  Returns:
  - Directories named "tests", "test_utils", etc.
  - Does NOT search file contents

COMMON USE CASES:

1. Find function definitions:
   <grepsearch>
     <pattern>^def \\w+</pattern>
     <path>./src</path>
     <mode>files</mode>
   </grepsearch>

2. Find TODO comments:
   <grepsearch>
     <pattern>#\\s*(TODO|FIXME|HACK)</pattern>
     <path>.</path>
     <mode>files</mode>
   </grepsearch>

3. Find import statements:
   <grepsearch>
     <pattern>^from .* import|^import</pattern>
     <path>./src</path>
     <mode>files</mode>
   </grepsearch>

4. Find configuration files:
   <grepsearch>
     <pattern>config\\.(json|yaml|yml|toml)</pattern>
     <path>.</path>
     <mode>files</mode>
   </grepsearch>

5. Find error handling:
   <grepsearch>
     <pattern>except.*Error|raise.*Error</pattern>
     <path>./src</path>
     <mode>files</mode>
   </grepsearch>

6. Find test directories:
   <grepsearch>
     <pattern>test</pattern>
     <path>.</path>
     <mode>directories</mode>
   </grepsearch>

7. Find API endpoints:
   <grepsearch>
     <pattern>@app\\.(get|post|put|delete)</pattern>
     <path>./api</path>
     <mode>files</mode>
   </grepsearch>

TIPS AND BEST PRACTICES:

1. START SPECIFIC, THEN BROADEN:
   ✓ Start with: <pattern>def my_specific_function</pattern>
   ✓ If no results: <pattern>my_specific</pattern>
   ✓ Still nothing: <pattern>specific</pattern>

2. USE WORD BOUNDARIES:
   ✓ <pattern>\\buser\\b</pattern> matches "user"
   ✗ <pattern>user</pattern> matches "username", "user_id", "superuser"

3. ESCAPE SPECIAL CHARACTERS:
   Regex special chars: . * + ? [ ] ( ) {{ }} ^ $ | \\
   ✓ <pattern>file\\.txt</pattern> matches "file.txt"
   ✗ <pattern>file.txt</pattern> matches "fileXtxt", "file.txt"

4. LIMIT SEARCH SCOPE:
   ✓ <path>./src/components</path> - Specific directory
   ✗ <path>.</path> - Entire project (slower)

5. USE APPROPRIATE MODE:
   - Looking for code? Use mode='files'
   - Exploring structure? Use mode='directories'
   - Not sure? Use mode='all' (default)

6. LEVERAGE GITIGNORE:
   - Searches are automatically filtered
   - No need to exclude common directories manually
   - Results focus on relevant source code

7. COMBINE WITH OTHER TOOLS:
   Workflow: grepsearch → readfile → editfile

   Step 1: Find files
   <grepsearch><pattern>old_api_function</pattern><path>./src</path></grepsearch>

   Step 2: Read context
   <readfile><filepath>./src/api.py</filepath><lines>40-60</lines></readfile>

   Step 3: Make changes
   <editfile><filepath>./src/api.py</filepath><search>old_api</search><replace>new_api</replace></editfile>

PERFORMANCE NOTES:

- Stops after {grepsearch.MAX_RESULTS} matches to prevent overwhelming output
- Skips binary files automatically
- Skips files larger than 2MB
- Uses streaming for large files
- Respects .gitignore for faster searches

LIMITATIONS:

- Maximum {grepsearch.MAX_RESULTS} results per search
- No multi-line regex patterns (searches line-by-line)
- No negative lookahead/lookbehind in patterns
- Files >2MB are skipped
- Binary files are skipped
- Cannot modify search results directly

ERROR MESSAGES:

"Invalid regex pattern"
  → Check regex syntax, escape special characters
  → Test pattern with simpler version

"Path does not exist"
  → Verify path is correct
  → Check for typos in directory name

"No matches found"
  → Try broader pattern
  → Check if files are .gitignored
  → Verify you're searching the right directory

"Search stopped after reaching {grepsearch.MAX_RESULTS} results"
  → Refine pattern to be more specific
  → Narrow search path
  → Use more restrictive mode

REGEX QUICK REFERENCE:

ANCHORS:
  ^     - Start of line
  $     - End of line
  \\b    - Word boundary

QUANTIFIERS:
  *     - 0 or more
  +     - 1 or more
  ?     - 0 or 1
  {{n}}   - Exactly n times
  {{n,m}} - Between n and m times

CHARACTER CLASSES:
  .     - Any character
  \\d    - Digit [0-9]
  \\w    - Word character [a-zA-Z0-9_]
  \\s    - Whitespace
  [abc] - Any of a, b, or c
  [^abc]- None of a, b, or c

GROUPS:
  (...)  - Capturing group
  (?:...)- Non-capturing group
  |      - OR operator

MODIFIERS:
  (?i)   - Case insensitive
  (?m)   - Multi-line mode

RELATED TOOLS:
  - <help>searchfile</help> - BM25-ranked content search (better for relevance)
  - <help>readfile</help> - Read file contents with line numbers
  - <help>editfile</help> - Modify file contents
  - <help>readdirectory</help> - View directory structure

╚══════════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def _find_project_root_and_spec(start_path: str) -> 'tuple[object | None, str | None]':
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