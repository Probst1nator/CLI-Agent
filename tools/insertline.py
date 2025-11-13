from typing import Optional

# tools/insertline.py
"""
This file implements the 'insertline' tool. It allows the agent to insert
new lines at specific positions in a file, resolving paths relative to the
current working directory.
"""
import os
import re


class insertline:
    @staticmethod
    def get_delim() -> str:
        return 'insertline'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "insertline",
            "description": (
                "Insert lines at specific positions. Use <after>N</after> or <before>N</before> "
                "for positioning. Defaults to end of file. "
                "Use <help>insertline</help> for positioning modes and examples."
            ),
            "example": (
                "<insertline><filepath>./src/main.py</filepath><after>10</after><content>print('inserted')</content></insertline>\n"
                "<insertline><filepath>./app.py</filepath><before>1</before><content># Header comment</content></insertline>"
            )
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        INSERTLINE TOOL - EXTENDED HELP                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  <insertline><filepath>path/to/file</filepath><after>10</after><content>new line</content></insertline>
  <insertline><filepath>path/to/file</filepath><before>5</before><content>new line</content></insertline>

THREE INSERTION MODES:

1. AFTER MODE (Insert after specific line)
   - Uses <after> tag with line number
   - Line numbers are 1-based (line 1 is first line)
   - <after>0</after> inserts at beginning of file
   - <after>N</after> inserts after line N

2. BEFORE MODE (Insert before specific line)
   - Uses <before> tag with line number
   - Line numbers are 1-based
   - <before>1</before> inserts at beginning
   - <before>N</before> inserts before line N

3. DEFAULT MODE (Insert at end)
   - Omit both <after> and <before> tags
   - Inserts at end of file
   - Useful for appending content

PARAMETERS:
  <filepath>  - Path to file (required, relative to current directory)
  <content>   - Text to insert (required, can be multiline)
  <after>     - Line number to insert after (optional)
  <before>    - Line number to insert before (optional)

POSITIONING EXAMPLES:

1. Insert at beginning:
   <insertline>
     <filepath>./main.py</filepath>
     <after>0</after>
     <content>#!/usr/bin/env python3</content>
   </insertline>

   OR:

   <insertline>
     <filepath>./main.py</filepath>
     <before>1</before>
     <content>#!/usr/bin/env python3</content>
   </insertline>

2. Insert after specific line:
   <insertline>
     <filepath>./utils.py</filepath>
     <after>15</after>
     <content>    # New helper function</content>
   </insertline>

3. Insert before specific line:
   <insertline>
     <filepath>./config.py</filepath>
     <before>20</before>
     <content>DEBUG_MODE = True</content>
   </insertline>

4. Insert at end (default):
   <insertline>
     <filepath>./README.md</filepath>
     <content>## New Section</content>
   </insertline>

MULTILINE CONTENT:

Insert multiple lines at once:
<insertline>
  <filepath>./app.py</filepath>
  <after>10</after>
  <content>def new_function():
    \"\"\"New function documentation.\"\"\"
    return True</content>
</insertline>

Important: The tool automatically adds a newline if not present.

INDENTATION TIPS:

1. PRESERVE INDENTATION:
   Include leading whitespace in <content>:
   <insertline>
     <filepath>./class.py</filepath>
     <after>25</after>
     <content>    def method(self):  # 4 spaces
        pass                   # 8 spaces</content>
   </insertline>

2. MATCH SURROUNDING CODE:
   Read context first to match indentation:
   <readfile><filepath>./app.py</filepath><lines>20-30</lines></readfile>
   Then insert with matching indentation.

3. PYTHON INDENTATION:
   - Use 4 spaces per level (PEP 8)
   - Avoid mixing tabs and spaces
   - Include proper indentation in <content>

4. JAVASCRIPT/CSS INDENTATION:
   - Common: 2 spaces per level
   - Match project style
   - Check existing code first

IMPORTANT BEHAVIORS:

1. LINE NUMBERING:
   - Lines are numbered starting from 1
   - <after>0</after> is special: inserts at beginning
   - Use readfile with <lines> to see line numbers

2. NEWLINE HANDLING:
   - Automatically adds newline if missing
   - Preserves existing newlines in content
   - File always ends with proper line ending

3. FILE MODIFICATION:
   - Original file is modified in-place
   - No backup is created automatically
   - Changes are immediate and permanent

4. PERMISSIONS:
   - Respects write access controls
   - May be restricted by security policy

WORKFLOW RECOMMENDATIONS:

1. SAFE INSERTION WORKFLOW:
   Step 1 - Find insertion point:
   <searchfile><query>class MyClass</query><path>./app.py</path></searchfile>

   Step 2 - Read context with line numbers:
   <readfile><filepath>./app.py</filepath><lines>20-40</lines></readfile>

   Step 3 - Insert at correct position:
   <insertline><filepath>./app.py</filepath><after>30</after><content>    def new_method(self):
        pass</content></insertline>

2. ADD MULTIPLE LINES:
   Insert from bottom to top to preserve line numbers:
   <insertline><filepath>./utils.py</filepath><after>100</after><content>line 1</content></insertline>
   <insertline><filepath>./utils.py</filepath><after>50</after><content>line 2</content></insertline>
   <insertline><filepath>./utils.py</filepath><after>20</after><content>line 3</content></insertline>

COMMON USE CASES:

1. Add import statements:
   <insertline>
     <filepath>./main.py</filepath>
     <after>0</after>
     <content>import datetime</content>
   </insertline>

2. Insert function definition:
   <insertline>
     <filepath>./helpers.py</filepath>
     <after>45</after>
     <content>def validate_input(data):
    return bool(data)</content>
   </insertline>

3. Add comment blocks:
   <insertline>
     <filepath>./config.py</filepath>
     <before>10</before>
     <content># Configuration section
# Updated: 2024-01-15</content>
   </insertline>

4. Append to file:
   <insertline>
     <filepath>./log.txt</filepath>
     <content>[2024-01-15] New log entry</content>
   </insertline>

5. Add class methods:
   <insertline>
     <filepath>./models.py</filepath>
     <after>50</after>
     <content>    def to_dict(self):
        \"\"\"Convert object to dictionary.\"\"\"
        return self.__dict__</content>
   </insertline>

TIPS AND BEST PRACTICES:

1. CHECK LINE NUMBERS FIRST:
   - Always read the file to verify line numbers
   - Line numbers can shift after edits
   - Use searchfile to locate insertion points

2. PRESERVE INDENTATION:
   - Include exact whitespace in content
   - Match surrounding code style
   - Test with small insertions first

3. INSERT FROM BOTTOM TO TOP:
   - When adding multiple lines, start from bottom
   - Prevents line number shifting
   - More reliable for batch insertions

4. USE MULTILINE FOR COMPLEX ADDITIONS:
   - Include entire blocks in single insertion
   - Maintains proper structure
   - Easier to verify correctness

5. VERIFY AFTER INSERTION:
   - Read the modified section to confirm
   - Check indentation and formatting
   - Ensure no syntax errors introduced

6. BACKUP IMPORTANT FILES:
   - Insertions are permanent
   - Consider using version control
   - No undo functionality available

ERROR MESSAGES:

"Line number exceeds file length"
  → File has fewer lines than specified
  → Re-read file to check current line count
  → Omit <after>/<before> to append at end

"Invalid line number"
  → Must be an integer
  → Check for typos in line number

"Write access denied"
  → Check file permissions
  → May be restricted by security policy

LIMITATIONS:
  - No automatic indentation detection
  - No undo functionality
  - Cannot insert into multiple files in one call
  - Must specify exact line numbers

RELATED TOOLS:
  - <help>editfile</help> - Modify existing line content
  - <help>deletelines</help> - Remove lines from files
  - <help>readfile</help> - View file with line numbers
  - <help>searchfile</help> - Find insertion points

╚══════════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML-like input to insert new lines at specific positions in a file.
        Supports insertion at the beginning, end, or after/before specific line numbers.
        """
        try:
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)
            after_match = re.search(r'<after>(.*?)</after>', content, re.DOTALL)
            before_match = re.search(r'<before>(.*?)</before>', content, re.DOTALL)

            if not filepath_match or not content_match:
                return "Error: Invalid format. Input must contain <filepath> and <content> tags."

            filepath = filepath_match.group(1).strip()
            new_content = content_match.group(1)

            if not filepath:
                return "Error: The <filepath> tag cannot be empty."

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
            insert_position: Optional[int] = None
            position_description = ""

            # Determine insertion position
            if after_match:
                try:
                    after_line = int(after_match.group(1).strip())
                    if after_line < 0:
                        return "Error: 'after' line number must be 0 or greater. Use 0 to insert at beginning."
                    if after_line > total_lines:
                        return f"Error: 'after' line {after_line} exceeds file length ({total_lines} lines)."
                    insert_position = after_line  # Insert after this line (0-based: after line N means at index N)
                    position_description = f"after line {after_line}"
                except ValueError:
                    return "Error: Invalid 'after' line number. Must be an integer."

            elif before_match:
                try:
                    before_line = int(before_match.group(1).strip())
                    if before_line < 1:
                        return "Error: 'before' line number must be 1 or greater."
                    if before_line > total_lines + 1:
                        return f"Error: 'before' line {before_line} exceeds file length ({total_lines} lines) + 1."
                    insert_position = before_line - 1  # Insert before this line (0-based: before line N means at index N-1)
                    position_description = f"before line {before_line}"
                except ValueError:
                    return "Error: Invalid 'before' line number. Must be an integer."

            else:
                # Default: insert at the end of the file
                insert_position = total_lines
                position_description = "at end of file"

            # Insert the new content
            new_line = new_content if new_content.endswith('\n') else new_content + '\n'
            lines.insert(insert_position, new_line)

            # Write the updated content back to the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return f"Successfully inserted line {position_description} in {full_path}:\n{new_content.rstrip()}"

        except Exception as e:
            return f"Error executing insertline tool: {str(e)}"