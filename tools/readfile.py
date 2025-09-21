# tools/readfile.py
"""
This file implements the 'readfile' tool. It allows the agent to read the
content of a file from the local file system. To prevent overwhelming the
context window, it truncates large files, showing the beginning and end.
Now supports reading images and registering them as base64 data for vision models.
If the file is not found at the specified path, it will attempt a recursive
search from the expected parent directory.
"""
import os
import base64
import mimetypes
import sys
import json
import hashlib

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from core.tool_context import tool_context
    TOOL_CONTEXT_AVAILABLE = True
except ImportError:
    TOOL_CONTEXT_AVAILABLE = False

# Import TextResourceManager for large file handling
try:
    from shared.utils.text_resource_manager import TextResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False

class readfile:
    """
    A tool to read the contents of a specified file.
    Paths are resolved relative to the current working directory.
    Large files are automatically truncated to fit within a reasonable context size.
    """
    # Define a character limit to prevent excessively large outputs.
    # This helps manage the LLM's context window and reduces token usage.
    CHUNK_LIMIT = 20000
    CONTENT_THRESHOLD = 20000  # Threshold for using resource manager

    # State file to track truncated reads
    STATE_FILE = os.path.expanduser("~/.readfile_state.json")

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'readfile'

    @staticmethod
    def _get_file_hash(filepath: str) -> str:
        """Generate a unique hash for a file based on its absolute path and modification time."""
        abs_path = os.path.abspath(filepath)
        try:
            mtime = os.path.getmtime(abs_path)
            content = f"{abs_path}:{mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except OSError:
            # If we can't get mtime, use just the path
            return hashlib.md5(abs_path.encode()).hexdigest()

    @staticmethod
    def _load_truncation_state() -> dict:
        """Load the state of previously truncated files."""
        try:
            if os.path.exists(readfile.STATE_FILE):
                with open(readfile.STATE_FILE, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    @staticmethod
    def _save_truncation_state(state: dict):
        """Save the state of truncated files."""
        try:
            with open(readfile.STATE_FILE, 'w') as f:
                json.dump(state, f)
        except IOError:
            # Silently fail if we can't save state
            pass

    @staticmethod
    def _mark_file_truncated(filepath: str):
        """Mark a file as having been truncated in a previous read."""
        file_hash = readfile._get_file_hash(filepath)
        state = readfile._load_truncation_state()
        state[file_hash] = {
            'path': os.path.abspath(filepath),
            'truncated': True
        }
        readfile._save_truncation_state(state)

    @staticmethod
    def _was_file_truncated(filepath: str) -> bool:
        """Check if a file was previously truncated."""
        file_hash = readfile._get_file_hash(filepath)
        state = readfile._load_truncation_state()
        return file_hash in state and state[file_hash].get('truncated', False)

    @staticmethod
    def _clear_truncation_state(filepath: str):
        """Clear the truncation state for a file after showing full content."""
        file_hash = readfile._get_file_hash(filepath)
        state = readfile._load_truncation_state()
        if file_hash in state:
            del state[file_hash]
            readfile._save_truncation_state(state)

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
            # Read from start_line to end of file
            end_idx = total_lines
        else:
            end_idx = min(total_lines, end_line)

        if start_idx >= total_lines:
            return f"Error: Start line {start_line} exceeds file length ({total_lines} lines)"

        selected_lines = lines[start_idx:end_idx]

        # Add line numbers for clarity
        result_lines = []
        for i, line in enumerate(selected_lines, start=start_line):
            result_lines.append(f"{i:4d}→{line}" if line.endswith('\n') else f"{i:4d}→{line}\n")

        header = f"Lines {start_line}-{end_line if end_line else total_lines} ({total_lines} total lines):\n"
        return header + ''.join(result_lines)

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "readfile",
            "description": (
                "Supports text and image files. "
                "Large text files are automatically handled by the system - if content exceeds limits, "
                "a resource_id is provided and you can use the `searchtext` tool to inspect the content."
            ),
            "example": (
                "<readfile>./src/main.py</readfile>\n"
                "<readfile><filepath>./app.js</filepath><lines>50-75</lines></readfile>\n"
                "<readfile>./screenshot.png</readfile>\n"
            )
        }

    @staticmethod
    def is_image_file(filepath: str) -> bool:
        """Check if the file is an image based on its extension, MIME type, and magic bytes."""
        # Common image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.ico', '.tiff', '.tif'}

        # Check file extension
        _, ext = os.path.splitext(filepath.lower())
        if ext in image_extensions:
            return True

        # Also check MIME type
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type and mime_type.startswith('image/'):
            return True

        # For files without extensions or unclear MIME types, check magic bytes
        try:
            with open(filepath, 'rb') as f:
                header = f.read(20)  # Read first 20 bytes

            # Check for common image magic bytes
            if header.startswith(b'\x89PNG'):  # PNG
                return True
            elif header.startswith(b'\xff\xd8\xff'):  # JPEG
                return True
            elif header.startswith(b'GIF8'):  # GIF
                return True
            elif header.startswith(b'BM'):  # BMP
                return True
            elif header.startswith(b'RIFF') and b'WEBP' in header[:20]:  # WebP
                return True
            elif header.startswith(b'\x00\x00\x01\x00') or header.startswith(b'\x00\x00\x02\x00'):  # ICO
                return True
            elif header.startswith((b'II*\x00', b'MM\x00*')):  # TIFF
                return True

        except Exception:
            # If we can't read the file, fall back to extension/MIME type only
            pass

        return False

    @staticmethod
    def process_image_file(filepath: str) -> str:
        """
        Process an image file by encoding it as base64 and registering it with the tool context.

        Args:
            filepath: Path to the image file

        Returns:
            A description message about the processed image
        """
        try:
            # Get file info
            file_size = os.path.getsize(filepath)
            mime_type, _ = mimetypes.guess_type(filepath)

            # Read and encode the image
            with open(filepath, 'rb') as image_file:
                image_data = image_file.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')

            # If MIME type is unknown, try to detect from magic bytes
            if not mime_type:
                header = image_data[:20] if len(image_data) >= 20 else image_data
                if header.startswith(b'\x89PNG'):
                    mime_type = 'image/png'
                elif header.startswith(b'\xff\xd8\xff'):
                    mime_type = 'image/jpeg'
                elif header.startswith(b'GIF8'):
                    mime_type = 'image/gif'
                elif header.startswith(b'BM'):
                    mime_type = 'image/bmp'
                elif header.startswith(b'RIFF') and b'WEBP' in header:
                    mime_type = 'image/webp'

            # Create description
            filename = os.path.basename(filepath)
            size_kb = file_size / 1024
            description = f"Image: {filename} ({mime_type or 'unknown type'}, {size_kb:.1f} KB)"

            # Register with tool context if available
            if TOOL_CONTEXT_AVAILABLE:
                tool_context.add_base64_image(base64_data, description)
                return f"📸 Image loaded: {description}\n[Image has been processed and is now available for analysis]"
            else:
                return f"📸 Image detected: {description}\n[Warning: Tool context not available - image analysis may be limited]"

        except Exception as e:
            return f"Error processing image file '{filepath}': {str(e)}"

    @staticmethod
    def run(content: str) -> str:
        """
        Reads the content of the file specified in the input string.
        Supports both simple filepath format and XML format with line ranges.
        If file is not found, performs a recursive search from the parent directory.

        Args:
            content: A string containing the path to the file or XML with filepath and lines.

        Returns:
            The content of the file as a string, or an error message if the file
            cannot be read. Returns a truncated version for large files.
        """
        try:
            import re

            # Check if input is XML format with filepath and lines tags
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            lines_match = re.search(r'<lines>(.*?)</lines>', content, re.DOTALL)

            if filepath_match:
                # XML format
                filepath = filepath_match.group(1).strip()
                line_spec = lines_match.group(1).strip() if lines_match else None
            else:
                # Simple format - just filepath
                filepath = content.strip()
                line_spec = None

            if not filepath:
                return "Error: No filepath provided."

            # Resolve the path relative to the current working directory for consistency and security
            full_path = os.path.abspath(filepath)
            search_message = ""

            # Check if the file exists. If not, perform a fallback recursive search.
            if not os.path.isfile(full_path):
                filename_to_find = os.path.basename(filepath)
                search_root = os.path.dirname(full_path)

                if not os.path.isdir(search_root):
                     return f"Error: File not found at '{full_path}' and parent directory '{search_root}' does not exist to start a search."

                found_files = []
                for root, dirs, files in os.walk(search_root):
                    # Prune common large/binary directories to speed up search
                    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.venv', 'build', 'dist']]
                    if filename_to_find in files:
                        found_files.append(os.path.join(root, filename_to_find))

                if len(found_files) == 1:
                    original_full_path = full_path
                    full_path = found_files[0]  # Update the path to the found file
                    search_message = f"File not found at '{original_full_path}'. Found at '{full_path}' instead. Reading from new location.\n\n"
                elif len(found_files) > 1:
                    paths_str = '\n - '.join(found_files)
                    return f"Error: File '{filename_to_find}' not found at specified path. Found multiple possible matches in '{search_root}':\n - {paths_str}\nPlease provide a more specific path."
                else:
                     return f"Error: File not found at '{full_path}' (and not found in a recursive search from '{search_root}')."

            # Check if this is an image file
            if readfile.is_image_file(full_path):
                image_result = readfile.process_image_file(full_path)
                return search_message + image_result

            # Attempt to read the file with UTF-8 encoding, replacing errors
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
            except Exception:
                # Fallback for non-UTF-8 files
                try:
                    with open(full_path, 'r', encoding='latin-1') as f:
                        file_content = f.read()
                except Exception as read_error:
                     return f"Error: Could not read file '{full_path}'. Tried UTF-8 and Latin-1. Reason: {read_error}"

            # Handle line-indexed reading if line_spec is provided
            if line_spec:
                try:
                    start_line, end_line = readfile._parse_line_range(line_spec)
                    line_content = readfile._extract_lines(file_content, start_line, end_line)
                    return search_message + line_content
                except ValueError:
                    return f"Error: Invalid line specification '{line_spec}'. Use formats like '15', '10-20', '10-', or '-20'."
                except Exception as e:
                    return f"Error extracting lines: {str(e)}"

            # Check if this file was previously truncated and user wants full output
            was_previously_truncated = readfile._was_file_truncated(full_path)

            # Clear any previous truncation state since general solution handles large files
            if was_previously_truncated:
                readfile._clear_truncation_state(full_path)

            # Return full content - main.py will automatically create resource if too large
            return search_message + file_content

        except Exception as e:
            return f"An unexpected error occurred in the readfile tool: {str(e)}"