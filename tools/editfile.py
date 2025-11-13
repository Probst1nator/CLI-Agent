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
        return """EDITFILE - EXTENDED HELP

USAGE:
  <editfile><filepath>path/file</filepath><search>old</search><replace>new</replace></editfile>
  <editfile><filepath>path/file</filepath><line>10</line><replace>content</replace></editfile>

MODES:
  Global Replace: <search><replace> - Replace ALL occurrences (exact match)
  Line-Specific: <line><replace> - Replace entire line N (1-based)

PARAMETERS:
  <filepath> - File path (required)
  <search>   - String to find (global mode)
  <replace>  - Replacement string (both modes)
  <line>     - Line number (line mode, 1-based)

EXAMPLES:
  Global: <editfile><filepath>./app.py</filepath><search>old_var</search><replace>new_var</replace></editfile>
  Line:   <editfile><filepath>./main.py</filepath><line>15</line><replace>print("updated")</replace></editfile>
  Multi:  <editfile><filepath>./README.md</filepath><search>## Old
Content</search><replace>## New
Content</replace></editfile>

KEY BEHAVIORS:
  • Global mode replaces ALL matches (not just first)
  • Case-sensitive exact matching
  • Line mode replaces entire line content
  • Preserves indentation when you include it in <replace>
  • UTF-8 encoding, preserves line endings
  • No undo - changes immediate

WORKFLOW:
  1. Find: <searchfile><query>pattern</query><path>file</path></searchfile>
  2. Verify: <readfile><filepath>file</filepath><lines>40-50</lines></readfile>
  3. Edit: <editfile><filepath>file</filepath><search>old</search><replace>new</replace></editfile>

TIPS:
  • Use specific search strings with context to avoid unintended matches
  • Check line numbers with readfile before line-specific edits
  • Include indentation in <replace> (e.g., "    return value")
  • For structural changes use insertline/deletelines instead

COMMON ERRORS:
  "String not found" → Verify exact match with searchfile (check case/whitespace)
  "Line exceeds length" → Use insertline to add lines
  "Write access denied" → Check permissions or security policy

LIMITATIONS:
  • Exact string matching only (no regex)
  • Cannot create new files
  • Global replace is all-or-nothing
  • No undo functionality

RELATED: <help>insertline</help> <help>deletelines</help> <help>readfile</help>
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