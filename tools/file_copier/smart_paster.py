# smart_paster.py
import os
import re
import base64
from typing import List, Tuple, Optional, Dict, Set

# Local import for the AI class
from ai_path_finder import AIFixPath

IGNORE_DIRS: Set[str] = {"__pycache__", "node_modules", "venv", "dist", "build", ".git", ".idea", ".vscode"}
IGNORE_FILES: Set[str] = {".DS_Store", ".gitignore", ".env"}
CACHE_FILENAME = ".file_copier_cache.json"

# --- All clipboard/request processing logic is now centralized here ---

def generate_project_tree(directory: str) -> str:
    """Generates a string representation of the project's file tree."""
    file_list = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for name in files:
            if name not in IGNORE_FILES and name != CACHE_FILENAME:
                rel_path = os.path.relpath(os.path.join(root, name), directory).replace('\\', '/')
                file_list.append(rel_path)
    return "\n".join(sorted(file_list))

def parse_clipboard_for_paths_and_code(message: str, directory: str) -> Tuple[List[str], List[str]]:
    """
    Parses a message to distinguish between existing filepaths and orphan code blocks.
    Returns (list of absolute paths, list of orphan code blocks).
    """
    found_files_abs = []
    orphan_code_blocks = []
    filepath_regex = re.compile(r"^(?:[./\w-]+/)?[\w-]+\.[\w]+$|^(?:[./\w-]+/)?[\w-]+$", re.MULTILINE)
    potential_paths = [(m.group(0), m.start(), m.end()) for m in filepath_regex.finditer(message)]
    last_pos = 0
    for path_str, start, end in potential_paths:
        code_before = message[last_pos:start].strip()
        full_path = os.path.join(directory, path_str.replace('/', os.sep))
        if os.path.exists(full_path) and os.path.isfile(full_path):
            found_files_abs.append(os.path.abspath(full_path))
            if code_before: orphan_code_blocks.append(code_before)
        else:
            block = (code_before + '\n' + path_str).strip()
            if block: orphan_code_blocks.append(block)
        last_pos = end
    remaining_code = message[last_pos:].strip()
    if remaining_code: orphan_code_blocks.append(remaining_code)
    if not found_files_abs and len(orphan_code_blocks) > 0:
        return [], [message.strip()]
    return list(set(found_files_abs)), [block for block in orphan_code_blocks if block]

async def handle_missing_filepaths(message: str, missed_code_blocks: List[str], directory: str) -> List[Tuple[str, str]]:
    """Uses AI to find paths for orphan code blocks and returns them as a list of (path, code) pairs."""
    project_tree = generate_project_tree(directory)
    fixer = AIFixPath()
    resolved_pairs = []
    print(f"AI Assistant: Analyzing {len(missed_code_blocks)} orphan block(s)...")
    for block in missed_code_blocks:
        suggested_path = await fixer.find_path(code_block=block, full_project_context=message, project_tree=project_tree)
        if suggested_path:
            resolved_pairs.append((suggested_path, block))
    return resolved_pairs

async def process_smart_request(user_request: str, directory: str) -> List[str]:
    """
    The main orchestrator for smart file discovery.
    Finds files by path and by AI analysis, then returns a combined list of relative paths.
    """
    # 1. Parse for known files and unknown code blocks
    found_files_abs, orphan_code_blocks = parse_clipboard_for_paths_and_code(user_request, directory)
    
    # 2. Convert absolute paths to relative for consistency
    found_rel_paths = {os.path.relpath(p, directory).replace('\\', '/') for p in found_files_abs}

    # 3. Run AI on orphan blocks if any exist
    resolved_pairs = []
    if orphan_code_blocks:
        resolved_pairs = await handle_missing_filepaths(user_request, orphan_code_blocks, directory)

    # 4. De-duplicate and combine
    ai_rel_paths = {pair[0] for pair in resolved_pairs}
    combined_paths = sorted(list(found_rel_paths.union(ai_rel_paths)))
    
    return combined_paths

# --- Legacy functions for backward compatibility ---

def get_language_hint(filename: str) -> str:
    lang = os.path.splitext(filename)[1][1:].lower()
    return {'yml': 'yaml', 'sh': 'bash', 'py': 'python'}.get(lang, lang)

def get_current_project_state(directory: str) -> Dict[str, float]:
    state = {}
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for name in files:
            if name in IGNORE_FILES or name == CACHE_FILENAME: continue
            try:
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, directory).replace(os.path.sep, '/')
                state[rel_path] = os.path.getmtime(abs_path)
            except OSError: continue
    return state

def build_clipboard_content(file_paths: List[str], root_directory: str, max_size: Optional[int] = None) -> str:
    parts, current_size = [], 0
    for i, abs_path in enumerate(file_paths):
        rel_path = os.path.relpath(abs_path, root_directory).replace(os.path.sep, '/')
        if max_size and current_size > max_size:
            parts.append(f"\n... and {len(file_paths) - i} more file(s) omitted due to size limit ...")
            break
        try:
            if os.path.splitext(rel_path)[1].lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
                with open(abs_path, 'rb') as f: b64 = base64.b64encode(f.read()).decode('ascii')
                block = f"# {rel_path}\n```base64\n{b64}\n```"
            else:
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f: content = f.read()
                lang = os.path.splitext(rel_path)[1][1:].lower()
                lang_hint = {'yml': 'yaml', 'sh': 'bash', 'py': 'python'}.get(lang, lang)
                block = f"# {rel_path}\n```{lang_hint}\n{content.strip()}\n```"
            parts.append(block)
            if max_size: current_size += len(block)
        except Exception as e:
            parts.append(f"# ERROR: Could not read {rel_path}\n```{e}\n```")
    return "\n\n".join(parts)

def apply_changes_to_files(content_to_apply: str, root_directory: str) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {"success": [], "errors": []}
    pattern = re.compile(r"^#\s*([^\n]+?)\s*\n```(?:[a-zA-Z0-9]*)?\n(.*?)\n```", re.DOTALL | re.MULTILINE)
    if not pattern.search(content_to_apply):
        pattern = re.compile(r"^```(?:[a-zA-Z0-9]*)?\n\s*#\s*([^\n]+?)\n(.*?)\n```", re.DOTALL | re.MULTILINE)

    for file_path, content in pattern.findall(content_to_apply):
        file_path = file_path.strip()
        if ".." in file_path or os.path.isabs(file_path):
            results["errors"].append(f"Skipped unsafe path: {file_path}")
            continue
        full_path = os.path.join(root_directory, file_path.replace('/', os.path.sep))
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if "base64" in content_to_apply.split(file_path,1)[1].split("```",1)[0]:
                with open(full_path, 'wb') as f: f.write(base64.b64decode(content.strip()))
            else:
                with open(full_path, 'w', encoding='utf-8') as f: f.write(content.strip() + '\n')
            results["success"].append(file_path)
        except Exception as e:
            results["errors"].append(f"Failed to write {file_path}: {e}")
    
    if not results["success"] and not results["errors"]:
        results["errors"].append("No valid file blocks found to apply.")
        
    return results