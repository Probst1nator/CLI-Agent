# utils/viewfiles.py
import os
import datetime
from typing import Dict, Any, List, Union
import markpickle

from py_classes.cls_util_base import UtilBase

class ViewFiles(UtilBase):
    """
    A utility to view the content of files or list directories.
    Returns a single, comprehensive Markdown string containing all content and a summary.
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
# This will return a Markdown string containing the file's content and a summary.
result_md = ViewFiles.run(paths=["main.py"])
print(result_md)
```"""
                },
                {
                    "description": "View multiple paths, including one that might fail",
                    "code": """```python
from utils.viewfiles import ViewFiles
result_md = ViewFiles.run(paths=["src/main.py", "docs/", "nonexistent_file.txt"])
print(result_md)
```"""
                },
                {
                    "description": "View a file's content with line numbers",
                    "code": """```python
from utils.viewfiles import ViewFiles
# This will show main.py with line numbers, which is great for discussing specific lines.
result_md = ViewFiles.run(paths=["main.py"], show_line_numbers=True)
print(result_md)
```"""
                }
            ]
        }

    @staticmethod
    def _run_logic(paths: List[str], show_line_numbers: bool = False) -> str:
        """
        Processes a list of paths, returning a single formatted Markdown string.
    
        Args:
            paths (List[str]): A list of paths to process.
            show_line_numbers (bool): If True, prepend line numbers to file content.
    
        Returns:
            str: A Markdown string summarizing the operation and showing content.
        """
        content_parts = []
        success_summary = []
        failed_summary = []

        for path in paths:
            abs_path = os.path.abspath(path)
            if not os.path.exists(path):
                failed_summary.append({"path": path, "reason": "Path not found"})
                continue
    
            if os.path.isfile(abs_path):
                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()
                    
                    if show_line_numbers:
                        lines = file_content.splitlines()
                        max_width = len(str(len(lines)))
                        numbered_lines = [f"{str(i + 1).rjust(max_width)} | {line}" for i, line in enumerate(lines)]
                        file_content = "\n".join(numbered_lines)

                    ext = os.path.splitext(path)[1].lstrip('.') or 'text'
                    content_parts.append(f"### File: `{abs_path}`\n```{ext}\n{file_content}\n```")
                    success_summary.append(f"Viewed file: `{abs_path}` ({len(file_content)} chars)")
    
                except Exception as e:
                    failed_summary.append({"path": abs_path, "reason": str(e)})
    
            elif os.path.isdir(abs_path):
                try:
                    items = sorted(os.listdir(abs_path))
                    dir_listing = [f"### Directory: `{abs_path}` ({len(items)} items)\n```text"]
                    for item in items:
                        item_path = os.path.join(abs_path, item)
                        is_dir = os.path.isdir(item_path)
                        dir_listing.append(f"- {item}{'/' if is_dir else ''}")
                    dir_listing.append("```")
                    content_parts.append("\n".join(dir_listing))
                    success_summary.append(f"Listed directory: `{abs_path}`")
    
                except Exception as e:
                    failed_summary.append({"path": abs_path, "reason": str(e)})

        # Assemble the final markdown output
        final_output = []
        if content_parts:
            final_output.extend(content_parts)
        
        # Add a summary section
        final_output.append("\n---\n### Summary")
        if success_summary:
            final_output.extend([f"- {s}" for s in success_summary])
        if failed_summary:
            final_output.append("\n**Failed to access:**")
            for item in failed_summary:
                final_output.append(f"- `{item['path']}`: {item['reason']}")
        
        if not success_summary and not failed_summary:
            final_output.append("No paths were processed.")
            
        return markpickle.dumps({"result": "\n".join(final_output)})


def run(paths: Union[str, List[str]], show_line_numbers: bool = False) -> str:
    """Module-level wrapper for ViewFiles._run_logic()."""
    if isinstance(paths, str):
        paths_to_process = [paths]
    elif isinstance(paths, list):
        paths_to_process = paths
    else:
        return markpickle.dumps({
            "error": f"Invalid input: Expected a string or a list of strings, but got {type(paths).__name__}."
        })

    return ViewFiles._run_logic(paths=paths_to_process, show_line_numbers=show_line_numbers)


# --- Minimal & Reproducible Test Showcase ---
if __name__ == "__main__":
    import tempfile
    
    # Define all test cases in a simple, data-driven list of dictionaries.
    test_cases = [
        {
            "description": "Test 1: View a single file",
            "setup": lambda: tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, 
                                                        content="Hello World\nLine 2"),
            "args": lambda temp_file: {"paths": [temp_file.name]},
            "assertion": lambda res: "result" in res and "Hello World" in res["result"]
        },
        {
            "description": "Test 2: View non-existent file",
            "args": {"paths": ["nonexistent_file.txt"]},
            "assertion": lambda res: "result" in res and "Failed to access" in res["result"]
        },
        {
            "description": "Test 3: View file with line numbers",
            "setup": lambda: tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                                        content="def hello():\n    print('world')"),
            "args": lambda temp_file: {"paths": [temp_file.name], "show_line_numbers": True},
            "assertion": lambda res: "result" in res and "1 |" in res["result"]
        }
    ]

    print("="*50)
    print(f"   Running Self-Tests for {__name__}   ")
    print("="*50)
    
    passed_count = 0
    # Generic test runner that iterates through the defined cases.
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- {test['description']} ---")
        try:
            # Setup temporary file if needed
            temp_file = None
            if 'setup' in test:
                temp_file = test['setup']()
                temp_file.flush()
                args = test['args'](temp_file)
            else:
                args = test['args']
            
            # Execute the utility's run function with the test arguments.
            result_str = run(**args)
            result_dict = markpickle.loads(result_str)
            
            print(f"Input: {args}")
            print(f"Output: {str(result_dict)[:200]}...")
            
            # Check if the result meets the assertion criteria.
            if test['assertion'](result_dict):
                print("Status: PASSED ✔️")
                passed_count += 1
            else:
                print("Status: FAILED ❌ (Assertion logic failed)")
                
            # Cleanup
            if temp_file:
                os.unlink(temp_file.name)
                
        except Exception as e:
            print(f"Status: FAILED ❌ (An unexpected exception occurred: {e})")

    # Final summary of the test run.
    print("\n" + "="*50)
    if passed_count == len(test_cases):
        print(f"  Summary: All {len(test_cases)} tests passed! ✅")
    else:
        print(f"  Summary: {passed_count}/{len(test_cases)} tests passed. Please review failures. ❌")
    print("="*50)