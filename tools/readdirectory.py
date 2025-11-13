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
            "description": (
                "View directory structure as a tree showing all files and subdirectories. "
                "Respects .gitignore. Most common: <path>./src</path>. "
                "Use <help>readdirectory</help> for max_depth and filtering options."
            ),
            "example": (
                "<readdirectory><path>./src</path></readdirectory>\n"
                "<readdirectory><path>./documents</path><max_depth>2</max_depth></readdirectory>"
            )
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      READDIRECTORY TOOL - EXTENDED HELP                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  <readdirectory>
    <path>./directory</path>
    <max_depth>3</max_depth>
  </readdirectory>

PURPOSE:
Explore project structure by displaying directory contents in a visual tree format.
Shows files and subdirectories without reading file contents. Automatically
filters out ignored files using .gitignore rules.

PARAMETERS:

  <path> (OPTIONAL, default: ".")
    - Directory path to list
    - Can be absolute or relative
    - Defaults to current directory
    - Examples:
      <path>./src</path>
      <path>/home/user/project</path>
      <path>../other_project</path>

  <max_depth> (OPTIONAL, default: unlimited)
    - Maximum recursion depth
    - Positive integer
    - Depth 1 = only immediate children
    - Depth 2 = children and grandchildren
    - Unlimited if not specified
    - Examples:
      <max_depth>1</max_depth>  # Shallow overview
      <max_depth>3</max_depth>  # Moderate depth
      <max_depth>5</max_depth>  # Deep exploration

OUTPUT FORMAT:

Tree structure with indentation:

Directory listing of '/home/user/project/src'
Total: 5 directories, 12 files

================================================================================
components/
  └─ Button.tsx
  └─ Input.tsx
  └─ Layout.tsx
utils/
  └─ api.ts
  └─ helpers.ts
index.ts
App.tsx

Directories: Shown with trailing "/"
Files: Shown without special marker
Indentation: "  " (2 spaces) per level
Tree markers: "└─ " for children

DEPTH EXAMPLES:

1. DEPTH 1 (Immediate children only):
   <readdirectory>
     <path>./src</path>
     <max_depth>1</max_depth>
   </readdirectory>

   Result:
   components/
   utils/
   index.ts
   App.tsx

2. DEPTH 2 (One level deeper):
   <readdirectory>
     <path>./src</path>
     <max_depth>2</max_depth>
   </readdirectory>

   Result:
   components/
     └─ Button.tsx
     └─ Input.tsx
   utils/
     └─ api.ts
   index.ts

3. UNLIMITED (Full tree):
   <readdirectory>
     <path>./src</path>
   </readdirectory>

   Shows complete directory structure with all nested levels.

GITIGNORE BEHAVIOR:

Automatically filters files/directories based on .gitignore:
- Reads all .gitignore files from path up to project root
- Identifies project root by .git directory
- Applies standard gitignore matching rules

Always excluded (regardless of .gitignore):
  - .git/

Default exclusions (when no .gitignore found):
  - node_modules/
  - __pycache__/
  - .venv/
  - .vscode/

Result:
  - Cleaner output focused on source code
  - Faster execution (skips large dependency directories)
  - More relevant for understanding project structure

COMMON USE CASES:

1. Quick project overview:
   <readdirectory>
     <path>.</path>
     <max_depth>2</max_depth>
   </readdirectory>

   Perfect for understanding high-level project organization.

2. Explore specific component:
   <readdirectory>
     <path>./src/components</path>
     <max_depth>3</max_depth>
   </readdirectory>

   Focus on specific part of codebase.

3. Find configuration files:
   <readdirectory>
     <path>.</path>
     <max_depth>1</max_depth>
   </readdirectory>

   See root-level config files and directories.

4. Check test structure:
   <readdirectory>
     <path>./tests</path>
   </readdirectory>

   View complete test organization.

5. Compare directory structures:
   <readdirectory>
     <path>./src</path>
     <max_depth>2</max_depth>
   </readdirectory>
   <readdirectory>
     <path>./dist</path>
     <max_depth>2</max_depth>
   </readdirectory>

   Compare source and build outputs.

6. Document project layout:
   <readdirectory>
     <path>.</path>
     <max_depth>3</max_depth>
   </readdirectory>

   Create documentation showing project structure.

7. Verify build output:
   <readdirectory>
     <path>./build</path>
     <max_depth>2</max_depth>
   </readdirectory>

   Check what was generated during build.

TIPS AND BEST PRACTICES:

1. START SHALLOW, GO DEEPER:
   ✓ First: <max_depth>1</max_depth> - Get overview
   ✓ Then: <max_depth>2</max_depth> - More detail
   ✓ Finally: No limit - Full exploration

2. USE SPECIFIC PATHS:
   ✓ <path>./src/components</path> - Focused
   ✗ <path>.</path> - Can be overwhelming

3. LIMIT DEPTH FOR LARGE PROJECTS:
   ✓ <max_depth>3</max_depth> - Manageable output
   ✗ No limit - Potential information overload

4. COMBINE WITH OTHER TOOLS:
   Workflow: readdirectory → readfile → editfile

   Step 1: See structure
   <readdirectory><path>./src</path><max_depth>2</max_depth></readdirectory>

   Step 2: Read specific file
   <readfile><filepath>./src/components/Button.tsx</filepath></readfile>

   Step 3: Make changes
   <editfile><filepath>./src/components/Button.tsx</filepath>
             <search>old</search><replace>new</replace></editfile>

5. CHECK IGNORED FILES:
   If you expect files that aren't showing:
   - Check .gitignore in parent directories
   - Verify files aren't in excluded directories
   - Consider if pathspec library is installed

6. USE FOR DOCUMENTATION:
   Output is already formatted for markdown:
   - Copy tree structure to README
   - Document project organization
   - Explain directory conventions

PERFORMANCE NOTES:

- Fast for small to medium projects (<1000 files)
- Depth limiting improves performance on large projects
- .gitignore filtering speeds up searches
- Skips reading file contents (only lists names)

Empty directories:
  - Shown in output if they exist
  - May not appear if filtered by .gitignore

Large projects:
  - Consider using max_depth to limit output
  - Focus on specific subdirectories
  - Use grepsearch for targeted file finding

UNDERSTANDING OUTPUT:

COUNT LINE:
  "Total: 5 directories, 12 files"
  - Only counts visible (non-ignored) items
  - Excludes .git and ignored directories
  - Accurate count of project structure

TREE MARKERS:
  "└─ " - Child item (file or directory)
  "  " - Indentation (2 spaces per level)
  "/" - Directory indicator

SORTING:
  - Directories listed before files at same level
  - Alphabetical within each category
  - Consistent across all platforms

COMPARISON WITH OTHER TOOLS:

vs grepsearch:
  - readdirectory: Shows structure, no content search
  - grepsearch: Searches file contents with regex

vs readfile:
  - readdirectory: Lists files, doesn't read them
  - readfile: Reads file contents

vs searchfile:
  - readdirectory: Shows all files in tree
  - searchfile: Finds files matching query with BM25

Use readdirectory when:
  ✓ Exploring new codebase
  ✓ Understanding project organization
  ✓ Finding where files are located
  ✓ Documenting structure

Use other tools when:
  ✗ Need to search file contents (use grepsearch/searchfile)
  ✗ Need to read files (use readfile)
  ✗ Need to find specific patterns (use grepsearch)

LIMITATIONS:

- Does NOT read file contents (only lists names)
- Cannot filter by file type or extension
- Cannot search within filenames (use grepsearch for that)
- No file size or modification date information
- Maximum reasonable output: ~1000 files
- Very deep nesting may be truncated in display

ERROR MESSAGES:

"Path does not exist or is not a directory"
  → Verify path is correct
  → Check for typos
  → Ensure directory exists

"Could not read .gitignore"
  → .gitignore file has permission issues
  → Non-critical - continues without that .gitignore

EXAMPLE WORKFLOWS:

1. EXPLORING NEW PROJECT:
   <readdirectory><path>.</path><max_depth>2</max_depth></readdirectory>
   # Get high-level overview

   <readdirectory><path>./src</path></readdirectory>
   # Deep dive into source code

   <readfile><filepath>./README.md</filepath></readfile>
   # Read project documentation

2. FINDING SPECIFIC FILES:
   <readdirectory><path>./src</path><max_depth>2</max_depth></readdirectory>
   # See structure

   <grepsearch><pattern>config</pattern><path>./src</path></grepsearch>
   # Find config-related files

   <readfile><filepath>./src/config/app.config.ts</filepath></readfile>
   # Read the config file

3. DOCUMENTING STRUCTURE:
   <readdirectory><path>.</path><max_depth>3</max_depth></readdirectory>
   # Generate tree for README.md

   Copy output to document project layout

RELATED TOOLS:
  - <help>grepsearch</help> - Search for patterns in files
  - <help>searchfile</help> - BM25-ranked content search
  - <help>readfile</help> - Read file contents
  - <help>writefile</help> - Create new files

╚══════════════════════════════════════════════════════════════════════════════╝
"""

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
