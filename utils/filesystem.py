# utils/filesystem.py

import os
from py_classes.cls_util_base import UtilBase
from typing import Literal
import json

class FileSystem(UtilBase):
    """
    A comprehensive utility for performing common filesystem operations.
    This tool provides a safe and reliable interface for reading, writing,
    patching, and listing files and directories. It does NOT use an LLM.
    """

    @staticmethod
    async def read(path: str) -> str:
        """
        Reads and returns the full content of a specified file.

        Args:
            path (str): The absolute or relative path to the file.

        Returns:
            str: The content of the file, or a JSON string with an 'error' key if it fails.
        """
        try:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return json.dumps({"error": f"File not found at '{abs_path}'."})
            if not os.path.isfile(abs_path):
                return json.dumps({"error": f"Path '{abs_path}' is a directory, not a file."})
                
            with open(abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return json.dumps({"error": f"Could not read file '{path}'. Reason: {e}"})

    @staticmethod
    async def write(path: str, content: str, mode: Literal['w', 'a'] = 'w') -> str:
        """
        Writes or appends content to a file, creating parent directories if needed.

        Args:
            path (str): The path for the file.
            content (str): The content to write.
            mode (Literal['w', 'a']): 'w' to overwrite, 'a' to append. Defaults to 'w'.

        Returns:
            str: A JSON string with the absolute path on success, or an 'error' key on failure.
        """
        try:
            abs_path = os.path.abspath(path)
            parent_dir = os.path.dirname(abs_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            with open(abs_path, mode, encoding='utf-8') as f:
                f.write(content)

            return json.dumps({"result": {"status": "Success", "path": abs_path}})
        except Exception as e:
            return json.dumps({"error": f"Could not write to file '{path}'. Reason: {e}"})

    @staticmethod
    async def list(path: str = '.', recursive: bool = False, depth: int = 1) -> str:
        """
        Lists files and directories at a given path.

        Args:
            path (str): The directory path to list. Defaults to '.'.
            recursive (bool): If True, lists recursively up to the specified depth.
            depth (int): The maximum recursion depth.

        Returns:
            str: A JSON string of the directory listing, or an 'error' key on failure.
        """
        try:
            abs_path = os.path.abspath(path)
            if not os.path.isdir(abs_path):
                return json.dumps({"error": f"Directory not found at '{abs_path}'."})
            
            items = []
            if recursive:
                for root, dirs, files in os.walk(abs_path, topdown=True):
                    current_depth = root.count(os.sep) - abs_path.count(os.sep)
                    if current_depth >= depth:
                        dirs[:] = []
                    
                    for d in sorted(dirs):
                        items.append(os.path.join(root, d).replace(abs_path, '', 1).strip(os.sep) + '/')
                    for f in sorted(files):
                        items.append(os.path.join(root, f).replace(abs_path, '', 1).strip(os.sep))
            else:
                for item in sorted(os.listdir(abs_path)):
                    item_path = os.path.join(abs_path, item)
                    items.append(f"{item}{'/' if os.path.isdir(item_path) else ''}")
            
            return json.dumps({"path": abs_path, "contents": items}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Could not list directory '{path}'. Reason: {e}"}) 