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
            "description": "Deletes specific lines or ranges of lines from a file. Supports single line deletion or range deletion. Paths are resolved relative to the current working directory. Requires `<filepath>` and either `<line>` for single line or `<start>` and `<end>` for ranges.",
            "example": "<deletelines><filepath>./src/main.py</filepath><line>15</line></deletelines> or <deletelines><filepath>./src/main.py</filepath><start>5</start><end>8</end></deletelines>"
        }

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