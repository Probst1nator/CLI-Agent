# tools/deletelines.py
"""
This file implements the 'deletelines' tool. It allows the agent to delete
specific lines or ranges of lines from a file, resolving paths relative to
the current working directory.
"""
import os
import re


class deletelines:
    @staticmethod
    def get_delim() -> str:
        return 'deletelines'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "deletelines",
            "description": (
                "Delete specific lines or ranges from files. "
                "Use <line>N</line> for single line or <start>N</start><end>M</end> for ranges. "
                "Use <help>deletelines</help> for pattern-based deletion and examples."
            ),
            "example": (
                "<deletelines><filepath>./main.py</filepath><line>15</line></deletelines>\n"
                "<deletelines><filepath>./app.py</filepath><start>5</start><end>8</end></deletelines>"
            )
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DELETELINES TOOL - EXTENDED HELP                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  <deletelines><filepath>path/to/file</filepath><line>10</line></deletelines>
  <deletelines><filepath>path/to/file</filepath><start>5</start><end>15</end></deletelines>

TWO DELETION MODES:

1. SINGLE LINE DELETION
   - Delete one specific line
   - Uses <line> tag
   - Line numbers start at 1

2. RANGE DELETION
   - Delete multiple consecutive lines
   - Uses <start> and <end> tags
   - Inclusive range (both start and end are deleted)

PARAMETERS:
  <filepath>  - Path to file (required, relative to current directory)
  <line>      - Single line number to delete
  <start>     - First line of range to delete
  <end>       - Last line of range to delete (inclusive)

SINGLE LINE EXAMPLES:

1. Delete a specific line:
   <deletelines>
     <filepath>./src/main.py</filepath>
     <line>15</line>
   </deletelines>

2. Remove debug statement:
   <deletelines>
     <filepath>./app.js</filepath>
     <line>42</line>
   </deletelines>

3. Delete import statement:
   <deletelines>
     <filepath>./utils.py</filepath>
     <line>3</line>
   </deletelines>

RANGE DELETION EXAMPLES:

1. Delete multiple lines:
   <deletelines>
     <filepath>./config.py</filepath>
     <start>10</start>
     <end>15</end>
   </deletelines>

2. Remove entire function:
   <deletelines>
     <filepath>./helpers.js</filepath>
     <start>25</start>
     <end>40</end>
   </deletelines>

3. Delete comment block:
   <deletelines>
     <filepath>./README.md</filepath>
     <start>50</start>
     <end>60</end>
   </deletelines>

IMPORTANT BEHAVIORS:

1. LINE NUMBERING:
   - Lines are numbered starting from 1 (not 0)
   - Use readfile with <lines> to see line numbers first
   - Example: <readfile><filepath>./app.py</filepath><lines>1-50</lines></readfile>

2. RANGE INCLUSION:
   - Both start and end lines are deleted (inclusive range)
   - <start>5</start><end>8</end> deletes lines 5, 6, 7, AND 8

3. DELETED CONTENT REPORTING:
   - Shows the content that was deleted
   - Helps verify correct lines were removed
   - Includes line numbers for reference

4. FILE MODIFICATION:
   - Original file is modified in-place
   - No backup is created automatically
   - Changes are immediate and permanent

WORKFLOW RECOMMENDATIONS:

1. SAFE DELETION WORKFLOW:
   Step 1 - Find the lines to delete:
   <searchfile><query>deprecated function</query><path>./utils.py</path></searchfile>

   Step 2 - Read context with line numbers:
   <readfile><filepath>./utils.py</filepath><lines>40-60</lines></readfile>

   Step 3 - Delete the lines:
   <deletelines><filepath>./utils.py</filepath><start>45</start><end>52</end></deletelines>

2. DELETE MULTIPLE SECTIONS:
   Delete from bottom to top to preserve line numbers:
   <deletelines><filepath>./app.py</filepath><start>100</start><end>110</end></deletelines>
   <deletelines><filepath>./app.py</filepath><start>50</start><end>60</end></deletelines>
   <deletelines><filepath>./app.py</filepath><start>20</start><end>25</end></deletelines>

COMMON USE CASES:

1. Remove debugging code:
   <deletelines><filepath>./debug.py</filepath><line>42</line></deletelines>

2. Delete deprecated functions:
   <deletelines><filepath>./api.py</filepath><start>150</start><end>175</end></deletelines>

3. Remove unused imports:
   <deletelines><filepath>./main.py</filepath><line>3</line></deletelines>
   <deletelines><filepath>./main.py</filepath><line>5</line></deletelines>

4. Clean up comments:
   <deletelines><filepath>./code.js</filepath><start>10</start><end>15</end></deletelines>

5. Remove configuration sections:
   <deletelines><filepath>./config.yaml</filepath><start>25</start><end>30</end></deletelines>

TIPS AND BEST PRACTICES:

1. CHECK BEFORE DELETING:
   - Always read the file first to verify line numbers
   - Use searchfile to locate the content
   - Double-check the range before deletion

2. DELETE FROM BOTTOM TO TOP:
   - When deleting multiple sections, start from the bottom
   - This preserves line numbers for earlier sections
   - Prevents line number shifting issues

3. VERIFY LINE NUMBERS:
   - Line numbers can shift after insertions/edits
   - Re-read the file if you've made other changes
   - The tool shows deleted content for verification

4. CONSIDER ALTERNATIVES:
   - For replacing code, use editfile instead
   - For adding code, use insertline
   - For complex refactoring, use multiple tools

5. BACKUP IMPORTANT FILES:
   - Deletion is permanent
   - Consider using version control (git)
   - No undo functionality available

ERROR MESSAGES:

"Line number exceeds file length"
  → File has fewer lines than specified
  → Re-read file to check current line count

"Start line must be less than or equal to end line"
  → Range is backwards (start > end)
  → Swap the values

"Write access denied"
  → Check file permissions
  → May be restricted by security policy

LIMITATIONS:
  - No pattern-based deletion (must specify exact line numbers)
  - No undo functionality
  - Cannot delete from multiple files in one call
  - Range must be consecutive lines

RELATED TOOLS:
  - <help>insertline</help> - Add new lines to files
  - <help>editfile</help> - Modify existing line content
  - <help>readfile</help> - View file with line numbers
  - <help>searchfile</help> - Find lines to delete

╚══════════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML-like input to delete specific lines or line ranges from a file.
        Supports both single line deletion and range deletion.
        """
        try:
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            line_match = re.search(r'<line>(.*?)</line>', content, re.DOTALL)
            start_match = re.search(r'<start>(.*?)</start>', content, re.DOTALL)
            end_match = re.search(r'<end>(.*?)</end>', content, re.DOTALL)

            if not filepath_match:
                return "Error: Invalid format. Input must contain <filepath> tag."

            filepath = filepath_match.group(1).strip()

            if not filepath:
                return "Error: The <filepath> tag cannot be empty."

            # Determine deletion mode
            if line_match:
                # Single line deletion
                try:
                    line_number = int(line_match.group(1).strip())
                    if line_number < 1:
                        return "Error: Line number must be 1 or greater."
                    start_line = end_line = line_number
                except ValueError:
                    return "Error: Invalid line number. Must be an integer."

            elif start_match and end_match:
                # Range deletion
                try:
                    start_line = int(start_match.group(1).strip())
                    end_line = int(end_match.group(1).strip())

                    if start_line < 1 or end_line < 1:
                        return "Error: Line numbers must be 1 or greater."

                    if start_line > end_line:
                        return "Error: Start line must be less than or equal to end line."

                except ValueError:
                    return "Error: Invalid line numbers. Must be integers."

            else:
                return "Error: Invalid format. Input must contain either <line> for single line deletion or both <start> and <end> for range deletion."

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

            # Validate line numbers against file length
            if start_line > total_lines:
                return f"Error: Start line {start_line} exceeds file length ({total_lines} lines)."

            if end_line > total_lines:
                return f"Error: End line {end_line} exceeds file length ({total_lines} lines)."

            # Store deleted content for reporting
            deleted_lines = []
            for i in range(start_line - 1, end_line):  # Convert to 0-based indexing
                if i < len(lines):
                    deleted_lines.append(f"{i+1:4d}→{lines[i].rstrip()}")

            # Delete the specified lines (convert to 0-based indexing)
            del lines[start_line - 1:end_line]

            # Write the updated content back to the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            # Create result message
            if start_line == end_line:
                result = f"Successfully deleted line {start_line} from {full_path}"
            else:
                result = f"Successfully deleted lines {start_line}-{end_line} from {full_path}"

            if deleted_lines:
                result += ":\nDeleted content:\n" + "\n".join(deleted_lines)

            return result

        except Exception as e:
            return f"Error executing deletelines tool: {str(e)}"