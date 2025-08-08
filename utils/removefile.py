import os
import json
import shutil
import datetime
from py_classes.cls_util_base import UtilBase

class RemoveFile(UtilBase):
    """
    A utility to safely "remove" a file by moving it to a backup directory.
    It can also list recently deleted files.
    """

    @staticmethod
    def run(path: str = None, list_recently_deleted: int = 0) -> str:
        """
        Moves a file to a backup directory or lists a number of recently deleted files.

        Args:
            path (str, optional): The absolute or relative path to the file to be removed.
                                  Required if list_recently_deleted is 0. Defaults to None.
            list_recently_deleted (int): If greater than 0, lists up to that many of the most
                                         recently deleted files with metadata. If 0, proceeds
                                         with file removal. Defaults to 0.

        Returns:
            str: A JSON string with a 'result' key containing a success message,
                 or an 'error' key on failure. When listing files, returns a
                 'recently_deleted' key with the metadata.
        """
        backup_dir = os.path.join(".cliagent", "removed")
        os.makedirs(backup_dir, exist_ok=True)
        metadata_file = os.path.join(backup_dir, "metadata.json")

        if list_recently_deleted > 0:
            if not os.path.exists(metadata_file):
                return json.dumps({"result": "No recently deleted files found."}, indent=2)
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                # Sort by date, newest first
                metadata_sorted = sorted(metadata, key=lambda x: x.get('deleted_at', ''), reverse=True)
                # Get the requested number of items
                items_to_show = metadata_sorted[:list_recently_deleted]
                return json.dumps({"recently_deleted": items_to_show}, indent=2)
            except (json.JSONDecodeError, FileNotFoundError):
                return json.dumps({"result": "No recently deleted files found or metadata is corrupted."}, indent=2)

        if not path:
            return json.dumps({"error": "The 'path' argument is required when not listing deleted files."}, indent=2)

        if not os.path.exists(path):
            return json.dumps({"error": f"File not found at path: {path}"}, indent=2)

        if not os.path.isfile(path):
            return json.dumps({"error": f"Path is not a file: {path}"}, indent=2)

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
            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {"error": f"Could not remove file {path}. Reason: {e}"}
            return json.dumps(error_result, indent=2)
