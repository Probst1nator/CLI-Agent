# tools/writefile.py
"""
This file implements the 'writefile' tool. It allows the agent to write or
overwrite a file with specified content within its current working directory
or an absolute path if provided.
"""
import os
import re
from core.globals import g

class writefile:
    @staticmethod
    def get_delim() -> str:
        return 'writefile'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "writefile",
            "description": "Writes or overwrites a file with specified content. Paths are resolved relative to the current working directory. Requires `<filepath>` and `<content>` tags.",
            "example": "<writefile><filepath>./my_file.txt</filepath><content>This is the file content.</content></writefile>"
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML-like input to extract a filepath and content, then writes
        the content to the specified file.
        """
        try:
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)

            if not filepath_match or not content_match:
                return "Error: Invalid format. Input must contain both <filepath> and <content> tags."

            filepath = filepath_match.group(1).strip()
            file_content = content_match.group(1)

            if not filepath:
                return "Error: The <filepath> tag cannot be empty."

            # This is a specific hook for the memory compaction process.
            # It intercepts writes to a special 'MEMORY' path.
            if hasattr(g, 'AGENT_IS_COMPACTING') and g.AGENT_IS_COMPACTING and "MEMORY" in filepath:
                from tools.disabled.memory import memory
                if hasattr(memory, 'store'):
                    memory.store(file_content)
                    return "Content stored in agent's long-term memory."

            # Resolve the path relative to the current working directory for safety.
            full_path = os.path.abspath(filepath)

            # Check if write is allowed via --allow-write-path
            allowed_write_paths = getattr(g, 'ALLOWED_WRITE_PATHS', [])
            if allowed_write_paths and full_path not in allowed_write_paths:
                return f"Error: Write access denied. File '{full_path}' is not in the allowed write paths list. Allowed paths: {allowed_write_paths}"

            # Ensure the parent directory exists before writing the file.
            parent_dir = os.path.dirname(full_path)
            os.makedirs(parent_dir, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

            # Track recently accessed file path for autocomplete
            if full_path not in g.RECENT_FILE_PATHS:
                g.RECENT_FILE_PATHS.insert(0, full_path)
                g.RECENT_FILE_PATHS = g.RECENT_FILE_PATHS[:g.MAX_RECENT_FILES]

            return f"Successfully wrote {len(file_content)} bytes to {full_path}"

        except Exception as e:
            return f"Error executing writefile tool: {str(e)}"