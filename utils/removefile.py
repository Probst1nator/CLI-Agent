import os
import json
import shutil
import datetime
from typing import Dict, Any
import markpickle

from py_classes.cls_util_base import UtilBase

class RemoveFile(UtilBase):
    """
    A utility to safely "remove" a file by moving it to a backup directory.
    It can also list recently deleted files.
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["delete file", "remove file", "trash file", "archive file", "safe delete"],
            "use_cases": [
                "Remove the temporary file 'temp_data.csv'.",
                "Delete 'old_script.py' but keep a backup.",
                "List the files I have deleted recently."
            ],
            "arguments": {
                "path": "The absolute or relative path to the file to be removed.",
                "list_recently_deleted": "If greater than 0, lists recently deleted files."
            },
            "code_examples": [
                {
                    "description": "Remove a file",
                    "code": "```python\nfrom utils.removefile import RemoveFile\nresult = RemoveFile.run(path='file_to_remove.txt')\n```"
                },
                {
                    "description": "List recently removed files",
                    "code": "```python\nfrom utils.removefile import RemoveFile\nresult = RemoveFile.run(list_recently_deleted=5)\n```"
                }
            ]
        }


    @staticmethod
    def _run_logic(path: str = None, list_recently_deleted: int = 0) -> str:
        """
        Moves a file to a backup directory or lists a number of recently deleted files.

        Args:
            path (str, optional): The absolute or relative path to the file to be removed.
                                  Required if list_recently_deleted is 0. Defaults to None.
            list_recently_deleted (int): If greater than 0, lists up to that many of the most
                                         recently deleted files with metadata. If 0, proceeds
                                         with file removal. Defaults to 0.

        Returns:
            str: A Markdown string with a success message or error.
        """
        backup_dir = os.path.join(".cliagent", "removed")
        os.makedirs(backup_dir, exist_ok=True)
        metadata_file = os.path.join(backup_dir, "metadata.json")

        if list_recently_deleted > 0:
            if not os.path.exists(metadata_file):
                return markpickle.dumps({"result": "No recently deleted files found."})
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                # Sort by date, newest first
                metadata_sorted = sorted(metadata, key=lambda x: x.get('deleted_at', ''), reverse=True)
                # Get the requested number of items
                items_to_show = metadata_sorted[:list_recently_deleted]
                return markpickle.dumps({"result": {"recently_deleted_files": items_to_show}})
            except (json.JSONDecodeError, FileNotFoundError):
                return markpickle.dumps({"result": "No recently deleted files found or metadata is corrupted."})

        if not path:
            return markpickle.dumps({"error": "The 'path' argument is required when not listing deleted files."})

        if not os.path.exists(path):
            return markpickle.dumps({"error": f"File not found at path: {path}"})

        if not os.path.isfile(path):
            return markpickle.dumps({"error": f"Path is not a file: {path}"})

        try:
            # Generate new filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = os.path.basename(path)
            new_filename = f"{timestamp}_{original_filename}"
            destination_path = os.path.join(backup_dir, new_filename)

            # Move the file
            shutil.move(path, destination_path)

            # Update metadata
            metadata = []
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    try:
                        metadata = json.load(f)
                    except json.JSONDecodeError:
                        metadata = []  # Reset if file is corrupted

            metadata.append({
                "original_path": os.path.abspath(path),
                "backup_path": os.path.abspath(destination_path),
                "deleted_at": datetime.datetime.now().isoformat()
            })

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            result = {
                "result": {
                    "status": "Success",
                    "message": "Successfully moved file to backup location.",
                    "original_path": os.path.abspath(path),
                    "backup_path": os.path.abspath(destination_path)
                }
            }
            return markpickle.dumps(result)

        except Exception as e:
            error_result = {"error": f"Could not remove file {path}. Reason: {e}"}
            return markpickle.dumps(error_result)


# Module-level run function for CLI-Agent compatibility
def run(path: str = None, list_recently_deleted: int = 0) -> str:
    """
    Module-level wrapper for RemoveFile._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        path (str): The file path to remove/backup
        list_recently_deleted (int): Number of recently deleted files to list
        
    Returns:
        str: Markdown string with result or error
    """
    return RemoveFile._run_logic(path=path, list_recently_deleted=list_recently_deleted)