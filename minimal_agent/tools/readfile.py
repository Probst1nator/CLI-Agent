# tools/readfile.py
"""
Tool to read file contents with line number support.
"""
import os
import re


class readfile:
    """Read the contents of a specified file."""
    CHUNK_LIMIT = 20000  # Character limit to prevent overwhelming context

    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'readfile'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "readfile",
            "description": (
                "Read text files with line numbers. Supports line ranges for large files. "
                "Optional <encoding> for non-UTF8 files (utf-8, latin-1, ascii, etc.)."
            ),
            "example": (
                "<readfile>./src/main.py</readfile>\n"
                "<readfile><filepath>./app.js</filepath><lines>50-75</lines></readfile>\n"
                "<readfile><filepath>./legacy.txt</filepath><encoding>latin-1</encoding></readfile>"
            )
        }

    @staticmethod
    def _parse_line_range(line_spec: str) -> tuple:
        """Parse line specification into (start_line, end_line).

        Supports:
        - Single line: "15" -> (15, 15)
        - Range: "10-20" -> (10, 20)
        - Open range: "10-" -> (10, None)
        - From start: "-20" -> (1, 20)
        """
        line_spec = line_spec.strip()

        if '-' in line_spec:
            parts = line_spec.split('-', 1)
            start_str, end_str = parts[0].strip(), parts[1].strip()

            start_line = int(start_str) if start_str else 1
            end_line = int(end_str) if end_str else None

            return (start_line, end_line)
        else:
            # Single line
            line_num = int(line_spec)
            return (line_num, line_num)

    @staticmethod
    def _extract_lines(content: str, start_line: int, end_line: int = None) -> str:
        """Extract specific lines from file content."""
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Convert to 0-based indexing
        start_idx = max(0, start_line - 1)

        if end_line is None:
            end_idx = total_lines
        else:
            end_idx = min(total_lines, end_line)

        if start_idx >= total_lines:
            return f"Error: Start line {start_line} exceeds file length ({total_lines} lines)"

        selected_lines = lines[start_idx:end_idx]

        # Add line numbers
        result_lines = []
        for i, line in enumerate(selected_lines, start=start_line):
            result_lines.append(f"{i:4d}→{line}" if line.endswith('\n') else f"{i:4d}→{line}\n")

        header = f"Lines {start_line}-{end_line if end_line else total_lines} ({total_lines} total lines):\n"
        return header + ''.join(result_lines)

    @staticmethod
    def run(content: str) -> str:
        """
        Reads the content of the file specified in the input string.
        Supports both simple filepath format and XML format with line ranges.

        Args:
            content: A string containing the path to the file or XML with filepath and lines.

        Returns:
            The content of the file as a string, or an error message.
        """
        try:
            # Check if input is XML format with filepath and lines tags
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            lines_match = re.search(r'<lines>(.*?)</lines>', content, re.DOTALL)
            encoding_match = re.search(r'<encoding>(.*?)</encoding>', content, re.DOTALL)

            if filepath_match:
                # XML format
                filepath = filepath_match.group(1).strip()
                line_spec = lines_match.group(1).strip() if lines_match else None
                file_encoding = encoding_match.group(1).strip() if encoding_match else 'utf-8'
            else:
                # Simple format - just filepath
                filepath = content.strip()
                line_spec = None
                file_encoding = 'utf-8'

            if not filepath:
                return "Error: No filepath provided."

            # Resolve the path relative to the current working directory
            full_path = os.path.abspath(filepath)

            if not os.path.isfile(full_path):
                return f"Error: File not found at '{full_path}'"

            # Attempt to read the file with specified encoding
            try:
                with open(full_path, encoding=file_encoding, errors='replace') as f:
                    file_content = f.read()
            except (UnicodeDecodeError, LookupError) as e:
                # Try fallback encodings
                if file_encoding != 'utf-8':
                    try:
                        with open(full_path, encoding='utf-8', errors='replace') as f:
                            file_content = f.read()
                    except Exception:
                        return f"Error: Could not read file with encoding '{file_encoding}' or UTF-8 fallback. Reason: {e}"
                else:
                    # Try latin-1 as last resort
                    try:
                        with open(full_path, encoding='latin-1') as f:
                            file_content = f.read()
                    except Exception as read_error:
                        return f"Error: Could not read file '{full_path}'. Tried UTF-8 and Latin-1. Reason: {read_error}"
            except Exception as read_error:
                return f"Error: Could not read file '{full_path}'. Reason: {read_error}"

            # Handle line-indexed reading if line_spec is provided
            if line_spec:
                try:
                    start_line, end_line = readfile._parse_line_range(line_spec)
                    line_content = readfile._extract_lines(file_content, start_line, end_line)
                    return line_content
                except ValueError:
                    return f"Error: Invalid line specification '{line_spec}'. Use formats like '15', '10-20', '10-', or '-20'."
                except Exception as e:
                    return f"Error extracting lines: {str(e)}"

            # Add line numbers to full content
            lines = file_content.splitlines(keepends=True)
            numbered_lines = []
            for i, line in enumerate(lines, start=1):
                numbered_lines.append(f"{i:4d}→{line}" if line.endswith('\n') else f"{i:4d}→{line}\n")

            return ''.join(numbered_lines)

        except Exception as e:
            return f"An unexpected error occurred in the readfile tool: {str(e)}"
