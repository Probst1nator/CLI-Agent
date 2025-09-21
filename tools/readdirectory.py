# tools/readdirectory.py
"""
This file implements the 'readdirectory' tool. It allows the agent to
recursively list the contents of a directory and automatically load all
readable text files and PDF documents into the searchable resource manager.

NOTE: PDF processing requires the 'pymupdf' package to be installed (`pip install pymupdf`).
"""
import os
import re
from typing import Tuple

# Attempt to import pathspec for .gitignore handling
try:
    import pathspec
except ImportError:
    pathspec = None

# Attempt to import PyMuPDF (fitz) for PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Import the TextResourceManager, which is central to this tool's functionality
try:
    from shared.utils.text_resource_manager import TextResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False

# A set of common binary file extensions to avoid reading non-text files.
# .pdf is now handled separately and removed from this list.
BINARY_EXTENSIONS = {
    '.pyc', '.pyo', '.o', '.so', '.a', '.dll', '.exe', '.lib',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico',
    '.mp3', '.wav', '.flac', '.ogg', '.mp4', '.avi', '.mov', '.mkv',
    '.zip', '.tar', '.gz', '.bz2', '.rar', '.7z',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.db', '.sqlite', '.sqlite3',
    '.eot', '.otf', '.ttf', '.woff', '.woff2'
}

class readdirectory:
    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool."""
        return 'readdirectory'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        info = {
            "name": "readdirectory",
            "description": "Recursively lists directory contents and loads readable files into the resource manager. Supports text files and automatically converts PDFs to text. Respects .gitignore rules.",
            "example": "<readdirectory><path>./documents</path></readdirectory>"
        }
        if not PYMUPDF_AVAILABLE:
            info["description"] += " (NOTE: PDF processing is disabled because 'pymupdf' is not installed.)"
        return info

    @staticmethod
    def _find_project_root_and_spec(start_path: str) -> Tuple[pathspec.PathSpec | None, str | None]:
        """
        Finds the project root (.git) and parses .gitignore files.
        Logic is adapted from the grepsearch tool for consistency.
        """
        if not pathspec:
            return None, None

        gitignore_patterns = []
        project_root = None
        current_path = os.path.abspath(start_path)

        while True:
            if os.path.isdir(os.path.join(current_path, '.git')):
                project_root = current_path

            gitignore_path = os.path.join(current_path, '.gitignore')
            if os.path.isfile(gitignore_path):
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    gitignore_patterns = f.readlines() + gitignore_patterns

            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                break

            current_path = parent_path
            if project_root and not current_path.startswith(project_root):
                break

        if gitignore_patterns:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_patterns)
            return spec, project_root

        return None, project_root

    @staticmethod
    def _process_pdf_file(file_path: str) -> str | None:
        """Extracts all text content from a PDF file using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            return None

        try:
            doc = fitz.open(file_path)
            full_text = []
            for page in doc:
                full_text.append(page.get_text())
            doc.close()
            return "\n".join(full_text)
        except Exception as e:
            # Gracefully handle corrupted, encrypted, or image-based PDFs
            print(f"Skipping PDF '{os.path.basename(file_path)}' due to extraction error: {e}")
            return None

    @staticmethod
    def run(content: str) -> str:
        """
        Executes the directory scan and content ingestion for text and PDF files.
        """
        if not RESOURCE_MANAGER_AVAILABLE:
            return "Error: TextResourceManager is not available. This tool cannot function."

        try:
            path_match = re.search(r'<path>(.*?)</path>', content, re.DOTALL)
            search_path = path_match.group(1).strip() if path_match else '.'

            full_path = os.path.abspath(search_path)
            if not os.path.isdir(full_path):
                return f"Error: Path does not exist or is not a directory: '{full_path}'"

            resource_manager = TextResourceManager._get_manager()
            spec, project_root = readdirectory._find_project_root_and_spec(full_path)

            all_paths = []
            ingested_files = []

            for root, dirs, files in os.walk(full_path, topdown=True):
                # Prune ignored directories
                if spec and project_root:
                    rel_root = os.path.relpath(root, project_root) if project_root else root
                    paths_to_check_dirs = [os.path.join(rel_root, d).replace('\\', '/') for d in dirs]
                    ignored_dirs = {os.path.basename(p) for p in spec.match_files(paths_to_check_dirs)}
                    dirs[:] = [d for d in dirs if d not in ignored_dirs and d != '.git']
                else:
                    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.venv']]

                for d in dirs:
                    all_paths.append(os.path.join(root, d) + os.sep)

                for filename in files:
                    file_path = os.path.join(root, filename)

                    is_ignored = False
                    if spec and project_root:
                        rel_path = os.path.relpath(file_path, project_root).replace('\\', '/')
                        if spec.match_file(rel_path):
                            is_ignored = True

                    if is_ignored:
                        continue

                    all_paths.append(file_path)

                    _, ext = os.path.splitext(filename.lower())
                    file_content = None
                    content_source_type = "text"

                    # --- Content Extraction Logic ---
                    if ext == '.pdf':
                        content_source_type = "pdf"
                        file_content = readdirectory._process_pdf_file(file_path)
                    elif ext not in BINARY_EXTENSIONS:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                file_content = f.read()
                        except (IOError, OSError):
                            continue # Skip unreadable files

                    # --- Ingestion Logic ---
                    if file_content and file_content.strip():
                        try:
                            resource = resource_manager.create_resource(
                                content=file_content,
                                source=file_path,
                                source_type=content_source_type
                            )
                            ingested_files.append({
                                "path": file_path,
                                "resource_id": resource.resource_id,
                                "lines": len(file_content.splitlines()),
                                "type": content_source_type
                            })
                        except Exception:
                            continue

            # --- Format the final output ---
            output = [f"Directory scan of '{full_path}' complete."]

            if ingested_files:
                output.append("\n--- Ingested Files ---")
                output.append("The following files have been loaded into the searchable resource manager:")
                for item in ingested_files:
                    rel_path = os.path.relpath(item['path'], full_path)
                    file_type = item['type'].upper()
                    output.append(f"- Path: {rel_path} (Type: {file_type}, Lines: {item['lines']})")
                    output.append(f"  Resource ID: {item['resource_id']}")
                output.append("\nUse the <searchtext> tool to inspect these resources.")
            else:
                output.append("\nNo readable text or PDF files were found to ingest.")

            tree_output = ["\n--- Directory Tree ---"]
            tree_paths = sorted([os.path.relpath(p, full_path) for p in all_paths])

            for path in tree_paths:
                if path == '.': continue
                parts = path.split(os.sep)
                indent = ' ' * 4 * (len(parts) - 1) + '|-- '
                tree_output.append(f"{indent}{parts[-1]}")

            output.extend(tree_output)

            return "\n".join(output)

        except Exception as e:
            return f"Error executing readdirectory tool: {str(e)}"