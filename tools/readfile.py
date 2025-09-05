# tools/readfile.py
"""
This file implements the 'readfile' tool. It allows the agent to read the
content of a file from the local file system. To prevent overwhelming the
context window, it truncates large files, showing the beginning and end.
Now supports reading images and registering them as base64 data for vision models.
"""
import os
import base64
import mimetypes
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from tool_context import tool_context
    TOOL_CONTEXT_AVAILABLE = True
except ImportError:
    TOOL_CONTEXT_AVAILABLE = False

class readfile:
    """
    A tool to read the contents of a specified file.
    Paths are resolved relative to the current working directory.
    Large files are automatically truncated to fit within a reasonable context size.
    """
    # Define a character limit to prevent excessively large outputs.
    # This helps manage the LLM's context window and reduces token usage.
    CHUNK_LIMIT = 50000

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'readfile'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "readfile",
            "description": f"Reads the content of a file. Supports both text files and images (PNG, JPG, GIF, etc.). Text files larger than {readfile.CHUNK_LIMIT} characters will be truncated. Images are automatically encoded for vision analysis.",
            "example": "<readfile>./src/components/main.js</readfile> or <readfile>./images/screenshot.png</readfile>"
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

        Args:
            content: A string containing the path to the file to be read.

        Returns:
            The content of the file as a string, or an error message if the file
            cannot be read. Returns a truncated version for large files.
        """
        try:
            # Sanitize the input to get a clean file path
            filepath = content.strip()
            if not filepath:
                return "Error: No filepath provided."

            # Resolve the path relative to the current working directory for consistency and security
            full_path = os.path.abspath(filepath)

            # Check if the file exists and is actually a file
            if not os.path.isfile(full_path):
                return f"Error: File not found at '{full_path}'"

            # Check if this is an image file
            if readfile.is_image_file(full_path):
                return readfile.process_image_file(full_path)

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


            # Check if the content exceeds the chunk limit
            if len(file_content) > readfile.CHUNK_LIMIT:
                # --- FIX APPLIED HERE ---
                # The original error was caused by using float division (`/`), which
                # results in a float (e.g., 12500.0). Slice indices must be integers.
                # Using integer division (`//`) ensures the result is an integer.
                chunk_size = readfile.CHUNK_LIMIT // 4

                truncated_message = (
                    f"\n\n... (File is too large: {len(file_content)} characters. Truncating to show start and end.) ...\n\n"
                )

                # Construct the truncated view
                truncated_content = (
                    file_content[:chunk_size] +
                    truncated_message +
                    file_content[-chunk_size:]
                )
                return truncated_content
            else:
                # If the file is within the limit, return its full content
                return file_content

        except Exception as e:
            return f"An unexpected error occurred in the readfile tool: {str(e)}"