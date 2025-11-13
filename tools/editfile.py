# tools/editfile.py
"""
This file implements the 'editfile' tool. It allows the agent to perform a
find-and-replace operation on a file, resolving paths relative to the current
working directory.
"""
import os
import re
from re import Match


class editfile:
    @staticmethod
    def get_delim() -> str:
        return 'editfile'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "editfile",
            "description": (
                "Find and replace text in files. Supports global replace or line-specific edits. "
                "Use <search><replace> for find/replace, or <line><replace> for line editing. "
                "Use <help>editfile</help> for advanced options and examples."
            ),
            "example": (
                "<editfile><filepath>./app.js</filepath><search>old_var</search><replace>new_var</replace></editfile>\n"
                "<editfile><filepath>./main.py</filepath><line>15</line><replace>print('updated')</replace></editfile>"
            )
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          EDITFILE TOOL - EXTENDED HELP                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  <editfile><filepath>path/to/file</filepath><search>old</search><replace>new</replace></editfile>
  <editfile><filepath>path/to/file</filepath><line>10</line><replace>new content</replace></editfile>

TWO EDITING MODES:

1. GLOBAL REPLACE MODE
   - Finds and replaces ALL occurrences of a string
   - Uses exact string matching
   - Requires <search> and <replace> tags

2. LINE-SPECIFIC MODE
   - Replaces content of a specific line number
   - Requires <line> and <replace> tags
   - Line numbers start at 1

PARAMETERS:
  <filepath>  - Path to file (required, relative to current directory)
  <search>    - String to find (for global replace mode)
  <replace>   - Replacement string (required in both modes)
  <line>      - Line number to edit (for line-specific mode)

GLOBAL REPLACE EXAMPLES:

1. Basic find and replace:
   <editfile>
     <filepath>./src/config.js</filepath>
     <search>const PORT = 3000;</search>
     <replace>const PORT = 8080;</replace>
   </editfile>

2. Replace variable name:
   <editfile>
     <filepath>./app.py</filepath>
     <search>old_variable</search>
     <replace>new_variable</replace>
   </editfile>

3. Replace multiline content:
   <editfile>
     <filepath>./README.md</filepath>
     <search>## Old Section
This is old content</search>
     <replace>## New Section
This is updated content</replace>
   </editfile>

4. Replace with empty string (deletion):
   <editfile>
     <filepath>./style.css</filepath>
     <search>.deprecated-class {
  color: red;
}</search>
     <replace></replace>
   </editfile>

LINE-SPECIFIC EXAMPLES:

1. Edit a specific line:
   <editfile>
     <filepath>./main.py</filepath>
     <line>15</line>
     <replace>    print("Updated line")</replace>
   </editfile>

2. Replace function definition:
   <editfile>
     <filepath>./utils.py</filepath>
     <line>42</line>
     <replace>def new_function_name(param1, param2):</replace>
   </editfile>

3. Update configuration value:
   <editfile>
     <filepath>./config.yaml</filepath>
     <line>8</line>
     <replace>debug: false</replace>
   </editfile>

IMPORTANT BEHAVIORS:

1. GLOBAL REPLACE:
   - Replaces ALL occurrences (not just first)
   - Fails if search string not found
   - Exact string matching (case-sensitive)
   - Preserves file encoding (UTF-8)

2. LINE EDITING:
   - Replaces entire line content
   - Preserves line ending (\\n)
   - Fails if line number exceeds file length
   - Shows old and new content in success message

3. PATH RESOLUTION:
   - Paths resolved relative to current working directory
   - Absolute paths supported
   - Symbolic links are followed

4. PERMISSIONS:
   - Respects write access controls
   - May be restricted by --allow-write-path configuration

TIPS AND BEST PRACTICES:

1. USE SPECIFIC SEARCH STRINGS
   - Include context to make searches unique
   - Avoid generic strings that might match unintended locations
   - Example: Use "const API_URL = 'old';" instead of just "'old'"

2. CHECK BEFORE EDITING
   - Use searchfile to verify what will be changed
   - Use readfile to view content before editing
   - Workflow: search → read → edit

3. LINE NUMBERS
   - Use readfile with <lines> to see line numbers
   - Example: <readfile><filepath>./app.py</filepath><lines>10-20</lines></readfile>

4. PRESERVE INDENTATION
   - Include leading whitespace in <replace> content
   - Example: <replace>    return value</replace> (4 spaces)

5. MULTILINE EDITS
   - Global replace supports multiline content
   - Use line-specific mode for single line changes
   - Consider insertline or deletelines for structural changes

WORKFLOW EXAMPLES:

1. Safe editing workflow:
   Step 1 - Find the line:
   <searchfile><query>def old_function</query><path>./utils.py</path></searchfile>

   Step 2 - Read context:
   <readfile><filepath>./utils.py</filepath><lines>40-50</lines></readfile>

   Step 3 - Edit:
   <editfile><filepath>./utils.py</filepath><search>def old_function()</search><replace>def new_function()</replace></editfile>

2. Batch variable renaming:
   <editfile><filepath>./app.py</filepath><search>old_var</search><replace>new_var</replace></editfile>
   <editfile><filepath>./utils.py</filepath><search>old_var</search><replace>new_var</replace></editfile>
   <editfile><filepath>./config.py</filepath><search>old_var</search><replace>new_var</replace></editfile>

LIMITATIONS:
  - No regex support (exact string matching only)
  - No undo functionality
  - Cannot edit binary files
  - Global replace is all-or-nothing (can't replace just first occurrence)
  - Cannot create new files (file must exist)

ERROR MESSAGES:

"String to find was not found"
  → Check spelling, whitespace, and case sensitivity
  → Use searchfile to verify the exact string

"Line exceeds file length"
  → Use insertline to add new lines instead
  → Check file has expected number of lines

"Write access denied"
  → Check file permissions
  → May be restricted by security policy

RELATED TOOLS:
  - <help>insertline</help> - Add new lines to files
  - <help>deletelines</help> - Remove lines from files
  - <help>searchfile</help> - Find content before editing
  - <help>readfile</help> - View content with line numbers

╚══════════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML-like input to find and replace text in a file, or edit specific lines.
        Supports:
        1. search/replace: replaces all occurrences of the find string
        2. Line-specific: replaces content of a specific line number
        """
        try:
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)
            line_match = re.search(r'<line>(.*?)</line>', content, re.DOTALL)
            replace_match = re.search(r'<replace>(.*?)</replace>', content, re.DOTALL)

            if not filepath_match:
                return "Error: Invalid format. Input must contain <filepath> tag."

            # Check for line-specific editing mode
            if line_match and replace_match:
                return editfile._edit_specific_line(filepath_match, line_match, replace_match)
            elif search_match and replace_match:
                return editfile._find_and_replace(filepath_match, search_match, replace_match)
            else:
                return "Error: Invalid format. Input must contain either (<find> and <replace>) OR (<line> and <content>) tags."

        except Exception as e:
            return f"Error executing editfile tool: {str(e)}"

    @staticmethod
    def _find_and_replace(filepath_match: Match[str], find_match: Match[str], replace_match: Match[str]) -> str:
        """Handle traditional find-and-replace editing."""
        filepath = filepath_match.group(1).strip()
        find_str = find_match.group(1)
        replace_str = replace_match.group(1)

        if not filepath:
            return "Error: The <filepath> tag cannot be empty."

        # Resolve the path relative to the current working directory for safety and predictability.
        full_path = os.path.abspath(filepath)

        # Check if write is allowed via --allow-write-path
        from core.globals import g
        allowed_write_paths = getattr(g, 'ALLOWED_WRITE_PATHS', [])
        if allowed_write_paths and full_path not in allowed_write_paths:
            return f"Error: Write access denied. File '{full_path}' is not in the allowed write paths list. Allowed paths: {allowed_write_paths}"

        if not os.path.isfile(full_path):
            return f"Error: File not found at '{full_path}' (resolved from '{filepath}')"

        with open(full_path, encoding='utf-8') as f:
            original_content = f.read()

        if find_str not in original_content:
            return f"Error: The string to find was not found in the file '{full_path}'. No changes were made."

        # Note: str.replace() replaces all occurrences. To replace only the first,
        # use: new_content = original_content.replace(find_str, replace_str, 1)
        new_content = original_content.replace(find_str, replace_str)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return f"Successfully edited file: {full_path}"

    @staticmethod
    def _edit_specific_line(filepath_match: Match[str], line_match: Match[str], content_match: Match[str]) -> str:
        """Handle line-specific editing."""
        filepath = filepath_match.group(1).strip()
        line_str = line_match.group(1).strip()
        new_content = content_match.group(1)

        if not filepath:
            return "Error: The <filepath> tag cannot be empty."

        try:
            line_number = int(line_str)
            if line_number < 1:
                return "Error: Line number must be 1 or greater."
        except ValueError:
            return f"Error: Invalid line number '{line_str}'. Must be an integer."

        # Resolve the path relative to the current working directory
        full_path = os.path.abspath(filepath)

        # Check if write is allowed via --allow-write-path
        from core.globals import g
        allowed_write_paths = getattr(g, 'ALLOWED_WRITE_PATHS', [])
        if allowed_write_paths and full_path not in allowed_write_paths:
            return f"Error: Write access denied. File '{full_path}' is not in the allowed write paths list. Allowed paths: {allowed_write_paths}"

        if not os.path.isfile(full_path):
            return f"Error: File not found at '{full_path}' (resolved from '{filepath}')"

        with open(full_path, encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        if line_number > total_lines:
            return f"Error: Line {line_number} exceeds file length ({total_lines} lines). Use insertline tool to add new lines."

        # Replace the specific line (convert to 0-based indexing)
        old_line = lines[line_number - 1].rstrip('\n\r')
        lines[line_number - 1] = new_content + '\n'

        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return f"Successfully edited line {line_number} in {full_path}:\nOld: {old_line}\nNew: {new_content}"