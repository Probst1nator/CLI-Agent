# utils/viewfiles.py
import os
import datetime
import json
from typing import Dict, Any, List, Union

from py_classes.cls_util_base import UtilBase

class ViewFiles(UtilBase):
    """
    A utility to view the content of files or list directories.
    It prints a summary and the content of successfully accessed paths directly to stdout
    and returns a structured JSON summary.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["read file", "view content", "display file", "show file", "file content", "examine file", "inspect file", "list directory", "directory contents", "file metadata", "multiple files"],
            "use_cases": [
                "Get the contents of 'config.json' and 'main.py'.",
                "List all files and folders in the current directory, and see a summary of what was accessed.", 
                "Examine the source code of multiple python files.",
                "View the contents of 'app.js' with line numbers for easier code review."
            ],
            "arguments": {
                "paths": "A list of absolute or relative paths to files or directories to be viewed.",
                "show_line_numbers": "A boolean indicating whether to show line numbers for file content. Helpful for edits and precision."
            },
            "code_examples": [
                {
                    "description": "View a single file's content and get a summary",
                    "code": """```python
from utils.viewfiles import ViewFiles
# This will print a summary and the file's content, then return a JSON status.
result_json = ViewFiles.run(paths=["main.py"])
```"""
                },
                {
                    "description": "View multiple paths, including one that might fail",
                    "code": """```python
from utils.viewfiles import ViewFiles
result_json = ViewFiles.run(paths=["src/main.py", "docs/", "nonexistent_file.txt"])
```"""
                },
                {
                    "description": "View a file's content with line numbers",
                    "code": """```python
from utils.viewfiles import ViewFiles
# This will show main.py with line numbers, which is great for discussing specific lines.
result_json = ViewFiles.run(paths=["main.py"], show_line_numbers=True)
```"""
                }
            ]
        }

    @staticmethod
    def _run_logic(paths: List[str], show_line_numbers: bool = False) -> str:
        """
        Processes a list of paths, printing markdown content and a summary to stdout.
    
        Args:
            paths (List[str]): A list of paths to process.
            show_line_numbers (bool): If True, prepend line numbers to file content.
    
        Returns:
            str: A JSON string summarizing the operation's result.
        """
        content_parts = []
        success_summary_details = []
        failed_summary_details = []

        for path in paths:
            abs_path = os.path.abspath(path)
            if not os.path.exists(path):
                failed_summary_details.append({"path": path, "reason": "Path not found"})
                continue
    
            if os.path.isfile(abs_path):
                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if show_line_numbers:
                        lines = content.splitlines()
                        max_line_num_width = len(str(len(lines)))
                        numbered_lines = [
                            f"{str(i + 1).rjust(max_line_num_width)} | {line}" 
                            for i, line in enumerate(lines)
                        ]
                        content = "\n".join(numbered_lines)

                    file_extension = os.path.splitext(path)[1].lstrip('.') or 'md'
                    content_parts.append(f"# {abs_path}\n```{file_extension}\n{content}\n```")
                    success_summary_details.append({"type": "File", "path": abs_path, "size_chars": len(content)})
    
                except Exception as e:
                    failed_summary_details.append({"path": abs_path, "reason": str(e)})
    
            elif os.path.isdir(abs_path):
                try:
                    dir_info = f"# Directory: {abs_path}\n"
                    dir_content_details = []
                    item_names = sorted(os.listdir(abs_path))
                    
                    for item_name in item_names:
                        item_path = os.path.join(abs_path, item_name)
                        try:
                            stat = os.stat(item_path)
                            mod_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            
                            if os.path.isdir(item_path):
                                item_type, size_str = "dir", f"{len(os.listdir(item_path))} items"
                            else:
                                item_type, size_str = "file", f"{stat.st_size} bytes"

                            dir_content_details.append(f"{mod_time:20} {item_type:6} {size_str:>15} {item_name}")
                        except OSError:
                            dir_content_details.append(f"{' ':20} {'?':6} {'?':>15} {item_name} (metadata error)")
                    
                    dir_info += "```text\n" + "\n".join(dir_content_details) + "\n```"
                    content_parts.append(dir_info)
                    success_summary_details.append({"type": "Dir", "path": abs_path, "items": len(item_names)})
    
                except Exception as e:
                    failed_summary_details.append({"path": abs_path, "reason": str(e)})

        # --- Print the human-readable summary and content to stdout ---
        summary_lines = ["--- ViewFiles Summary ---"]
        if success_summary_details:
            summary_lines.append("Accessed Paths:")
            for item in success_summary_details:
                size_info = f"({item.get('size_chars', 0)} chars)" if item['type'] == 'File' else f"({item.get('items', 0)} items)"
                summary_lines.append(f"  - [{item['type']}] {item['path']} {size_info}")
        
        if failed_summary_details:
            summary_lines.append("Failed Paths:")
            for item in failed_summary_details:
                summary_lines.append(f"  - [Fail] {item['path']} (Reason: {item['reason']})")
        summary_lines.append("--- End Summary ---")
        
        print("\n".join(summary_lines))
        
        # Also print the actual file/dir content to stdout
        if content_parts:
            print("\n\n" + "\n\n".join(content_parts).strip())
        
        # --- Return a structured JSON string for programmatic use ---
        result = {
            "status": "success" if not failed_summary_details else "partial_success",
            "accessed": success_summary_details,
            "failed": failed_summary_details
        }
        return json.dumps(result, indent=2)


# Module-level run function for CLI-Agent compatibility
def run(paths: Union[str, List[str]], show_line_numbers: bool = False) -> str:
    """
    Module-level wrapper for ViewFiles._run_logic(). It intelligently handles
    either a single path string or a list of path strings.
    """
    if isinstance(paths, str):
        paths_to_process = [paths]
    elif isinstance(paths, list):
        paths_to_process = paths
    else:
        error_result = {
            "status": "error",
            "message": f"Invalid input to viewfiles.run(): Expected a string or a list of strings, but got {type(paths).__name__}."
        }
        print(f"\n--- ViewFiles Error ---\n{error_result['message']}\n--- End Error ---")
        return json.dumps(error_result, indent=2)

    return ViewFiles._run_logic(paths=paths_to_process, show_line_numbers=show_line_numbers)