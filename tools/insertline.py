# tools/insertline.py
"""
This file implements the 'insertline' tool. It allows the agent to insert
new lines at specific positions in a file, resolving paths relative to the
current working directory.
"""
import os
import re
from typing import Optional

class insertline:
    @staticmethod
    def get_delim() -> str:
        return 'insertline'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "insertline",
            "description": "Inserts new lines at specific positions in a file. Can insert at the beginning, end, or after a specific line number. Paths are resolved relative to the current working directory. Requires `<filepath>` and `<content>` tags, with optional `<after>` or `<before>` tags for positioning.",
            "example": "<insertline><filepath>./src/main.py</filepath><after>10</after><content>print('inserted line')</content></insertline> or <insertline><filepath>./src/main.py</filepath><before>1</before><content># New header comment</content></insertline>"
        }

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

            with open(full_path, 'r', encoding='utf-8') as f:
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