#!/usr/bin/env python3
"""
Generate INDEX.md files for all directories in a project.

Each INDEX.md lists all files in the directory with functional descriptions
of their purpose in the codebase using LLM analysis.

Usage:
    python3 generate_index_files.py /path/to/project              # Generate with LLM analysis
    python3 generate_index_files.py /path/to/project --max-depth 2  # Limit recursion depth
"""

import argparse
import asyncio
import fnmatch
import re
import sys
from pathlib import Path
from typing import Optional

# Save original argv FIRST, before any modifications
_ORIGINAL_ARGV = sys.argv.copy()

# Add parent directory to import core modules (if available)
sys.path.insert(0, str(Path(__file__).parent.parent))

# CRITICAL: Set sys.argv to minimal form IMMEDIATELY to prevent main.py argparse conflicts
# The CLI-Agent's argparse runs during some module initializations and will complain
# about our script's arguments. We'll restore the real argv only for OUR argparse.
sys.argv = [sys.argv[0]]  # Keep only script name

# Default skip patterns (can be overridden by .gitignore)
DEFAULT_SKIP_PATTERNS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
}

# Binary/compiled file extensions to skip
BINARY_EXTENSIONS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".o", ".a"}

# Max file size to read (500KB)
MAX_FILE_SIZE = 500 * 1024


def parse_gitignore(gitignore_path: Path) -> set[str]:
    """
    Parse .gitignore file and return set of patterns.

    Simplified parser - handles basic patterns.
    """
    patterns = set()

    if not gitignore_path.exists():
        return patterns

    try:
        with open(gitignore_path) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                patterns.add(line)
    except Exception:
        pass

    return patterns


def should_skip(path: Path, skip_patterns: set[str], root: Path) -> bool:
    """
    Check if path should be skipped based on patterns.

    Args:
        path: Path to check
        skip_patterns: Set of gitignore-style patterns
        root: Project root (for relative pattern matching)
    """
    name = path.name

    # Skip hidden files/dirs (except .gitignore, .env.example)
    if name.startswith(".") and name not in {".gitignore", ".env.example"}:
        return True

    # Skip binary files
    if path.is_file() and path.suffix in BINARY_EXTENSIONS:
        return True

    # Check against patterns
    for pattern in skip_patterns:
        # Try matching against name
        if fnmatch.fnmatch(name, pattern):
            return True

        # Try matching against relative path
        try:
            rel_path = path.relative_to(root)
            if fnmatch.fnmatch(str(rel_path), pattern):
                return True
            # Also check if any parent directory matches
            for parent in rel_path.parents:
                if fnmatch.fnmatch(parent.name, pattern):
                    return True
        except ValueError:
            pass

    return False


def discover_directories(
    root: Path, skip_patterns: set[str], max_depth: Optional[int] = None
) -> list[Path]:
    """
    Discover all directories in project that should have INDEX.md.

    Args:
        root: Project root directory
        skip_patterns: Patterns to skip (from .gitignore + defaults)
        max_depth: Maximum recursion depth (None = unlimited)

    Returns:
        List of directories to process
    """
    directories = []

    def scan_directory(path: Path, depth: int = 0):
        """Recursively scan directory."""
        if max_depth is not None and depth > max_depth:
            return

        # Check if directory should be skipped
        if should_skip(path, skip_patterns, root):
            return

        # Add this directory
        directories.append(path)

        # Scan subdirectories
        try:
            for item in path.iterdir():
                if item.is_dir():
                    scan_directory(item, depth + 1)
        except PermissionError:
            pass

    scan_directory(root)
    return sorted(directories)


def read_file_contents(file_path: Path) -> Optional[str]:
    """Read file contents safely (returns None if too large or binary)."""
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return None
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def clean_description(text: str, max_length: int = 200) -> str:
    """
    Clean LLM-generated description to plain text.

    Removes markdown artifacts, code blocks, normalizes whitespace,
    and truncates at word boundaries.
    """
    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove code blocks (```language\ncode\n``` or ```code```)
    text = re.sub(r'```[\w]*\n.*?\n```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Remove markdown headers (# Header)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove Python comment markers (# comment)
    text = re.sub(r'^#\s+', '', text, flags=re.MULTILINE)

    # Remove backticks
    text = text.replace('`', '')

    # Normalize newlines and multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    # Strip again after all replacements
    text = text.strip()

    # Smart truncation at word boundary if needed
    if len(text) > max_length:
        # Try to truncate at last complete word before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')

        if last_space > max_length * 0.8:  # Only if we're not losing too much
            text = truncated[:last_space] + "..."
        else:
            text = truncated.rstrip() + "..."

    return text


def generate_file_description_with_llm(
    file_path: Path,
    file_contents: str,
    directory_context: str,
    root: Path,
    directory: Path,
    skip_patterns: set[str],
) -> Optional[str]:
    """
    Generate functional description using LLM by reading file contents.

    This function:
    1. Creates a Chat object
    2. Adds system message with instructions
    3. Adds user message with comprehensive context (root INDEX, dir INDEX, file list, contents)
    4. Calls generate_completion() to get description
    """
    # Import here to avoid argparse conflicts with main.py
    try:
        from core.chat import Chat, Role
    except ImportError as e:
        print(f"  Warning: Cannot import Chat module: {e}")
        return None

    # Create chat for this request
    chat = Chat()

    # System message: Instructions for LLM
    system_message = """You are a technical documentation expert analyzing a codebase.

Generate a concise functional description of a file (max 200 characters).

CRITICAL OUTPUT REQUIREMENTS:
- Output ONLY plain text - NO markdown, NO code blocks, NO formatting
- Single line only - NO newlines or line breaks
- NO backticks, NO # headers, NO comments, NO code syntax
- Just simple, clean descriptive text

Guidelines:
- Focus on WHAT the file does and WHY it exists in the codebase
- Use active voice (e.g., "Manages X", "Provides Y", "Implements Z")
- Be specific about functionality, not generic
- Avoid redundant information
- Include code quality observations when relevant

Examples of GOOD descriptions:
- "Permission settings management (load/save/check)"
- "Chat-related API endpoints (load, save, search, edit) - Includes unfinished mocks"
- "WebSocket communication manager with auto-reconnect - Misses typing and docstrings"

Examples of BAD descriptions:
- "Python module for settings" (too generic)
- "File that does chat stuff" (vague)
- "# Main script for processing" (has markdown header)
- "```python\nprocessing script\n```" (has code block)

Output ONLY the plain text description, nothing else."""

    # Gather comprehensive context
    context_parts = []

    # 1. Root INDEX.md (if exists)
    root_index = root / "INDEX.md"
    if root_index.exists():
        try:
            root_index_content = root_index.read_text(encoding="utf-8", errors="ignore")
            context_parts.append(f"""# Root Project Index (INDEX.md from root)
```
{root_index_content[:3000]}
```
""")
        except Exception:
            pass

    # 2. Current directory INDEX.md (if exists and different from root)
    dir_index = directory / "INDEX.md"
    if dir_index.exists() and directory != root:
        try:
            dir_index_content = dir_index.read_text(encoding="utf-8", errors="ignore")
            context_parts.append(f"""# Current Directory Index (INDEX.md from {directory.name}/)
```
{dir_index_content[:2000]}
```
""")
        except Exception:
            pass

    # 3. List of all files in current directory
    files_in_dir = []
    try:
        for item in sorted(directory.iterdir()):
            if item.is_file() and not should_skip(item, skip_patterns, root):
                files_in_dir.append(item.name)
    except Exception:
        pass

    if files_in_dir:
        context_parts.append(f"""# Files in Current Directory ({directory.name}/)
{', '.join(files_in_dir)}
""")

    # Build user message with all context
    context_section = "\n".join(context_parts) if context_parts else ""

    user_message = f"""You are analyzing a file to generate its functional description.

{context_section}

# Target File Information
File: {file_path.name}
Directory: {directory_context}
Extension: {file_path.suffix}

# File Contents (first 8000 chars)
```
{file_contents[:8000]}
```

Based on the project context, directory context, and file contents above, generate a concise functional description (max 100 characters):"""

    # Build context chat with add_message calls
    chat.add_message(Role.SYSTEM, content=system_message)
    chat.add_message(Role.USER, content=user_message)

    try:
        # Import LlmRouter (sys.argv already cleared at module level to prevent conflicts)
        from core.llm_router import LlmRouter

        # Generate completion using asyncio.run() since this is a sync script
        print(
            f"    Generating description for {file_path.name}...", end=" ", flush=True
        )
        response = asyncio.run(
            LlmRouter.generate_completion(
                chat=chat,
                preferred_models=["deepcoder:1.5b"],  # Fast, small Ollama model for descriptions
                force_preferred_model=True,  # Don't fall back to other models
                # Use default streaming callback to show generation in real-time
            )
        )

        # Clean and normalize the description
        description = clean_description(response, max_length=200)

        print("✓")
        return description

    except Exception as e:
        print(f"✗ (Error: {e})")
        return None


def get_file_description(
    file_path: Path,
    dir_name: str,
    root: Optional[Path] = None,
    directory: Optional[Path] = None,
    skip_patterns: Optional[set[str]] = None,
) -> str:
    """
    Generate a functional description for a file using LLM analysis.

    Args:
        file_path: Path to the file
        dir_name: Directory name containing the file
        root: Project root path (for LLM context)
        directory: Current directory path (for LLM context)
        skip_patterns: Skip patterns (for LLM context)

    Returns:
        Functional description string
    """
    name = file_path.name

    # Special files with hardcoded descriptions (don't need LLM)
    special_descriptions = {
        "INDEX.md": "Directory index with functional descriptions of all files",
        "README.md": "Overview and documentation for this directory",
        "__init__.py": "Python package initialization",
        ".gitignore": "Git ignore patterns for version control",
    }

    if name in special_descriptions:
        return special_descriptions[name]

    # Use LLM for all other files
    file_contents = read_file_contents(file_path)
    if file_contents and root and directory and skip_patterns is not None:
        llm_description = generate_file_description_with_llm(
            file_path, file_contents, dir_name, root, directory, skip_patterns
        )
        if llm_description:
            return llm_description

    # Fallback if LLM fails or file cannot be read
    return f"File: {name}"




def generate_index_content(
    directory: Path,
    root: Path,
    skip_patterns: set[str],
) -> str:
    """Generate INDEX.md content for a directory using LLM analysis."""
    dir_name = directory.name
    try:
        relative_path = directory.relative_to(root)
    except ValueError:
        relative_path = directory

    lines = []
    lines.append(f"# {dir_name}")
    lines.append("")
    lines.append(f"**Location:** `{relative_path}/`")
    lines.append("")

    # Get all files in directory (non-recursive)
    files = []
    subdirs = []

    for item in sorted(directory.iterdir()):
        # Use skip patterns to filter
        if should_skip(item, skip_patterns, root):
            continue

        if item.is_file():
            files.append(item)
        elif item.is_dir():
            subdirs.append(item)

    # List files
    if files:
        lines.append("## Files")
        lines.append("")
        lines.append("| File | Description |")
        lines.append("|------|-------------|")

        for file in files:
            desc = get_file_description(
                file,
                dir_name,
                root=root,
                directory=directory,
                skip_patterns=skip_patterns,
            )
            lines.append(f"| `{file.name}` | {desc} |")

        lines.append("")

    # List subdirectories
    if subdirs:
        lines.append("## Subdirectories")
        lines.append("")

        for subdir in subdirs:
            # Check if subdirectory has INDEX.md
            has_index = (subdir / "INDEX.md").exists()
            index_marker = " 📋" if has_index else ""
            lines.append(f"- `{subdir.name}/`{index_marker}")

        lines.append("")
        if any((subdir / "INDEX.md").exists() for subdir in subdirs):
            lines.append("📋 = Has INDEX.md")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*This INDEX.md file provides functional descriptions of all files in this directory.*"
    )
    lines.append("")
    lines.append("*Descriptions generated using LLM analysis of file contents.*")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate INDEX.md files for a project using LLM analysis (automatically traverses all subdirectories)"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the project directory to generate INDEX.md files for",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum directory depth to scan (default: unlimited)",
    )
    parser.add_argument(
        "--ignore-gitignore",
        action="store_true",
        help="Don't use .gitignore patterns (only use defaults)",
    )

    # Restore original argv for OUR argparse (was cleared earlier to prevent main.py conflicts)
    sys.argv = _ORIGINAL_ARGV

    args = parser.parse_args()

    # IMMEDIATELY clear sys.argv again to prevent main.py argparse conflicts during LLM imports
    sys.argv = [_ORIGINAL_ARGV[0]]

    # Get root directory from required argument
    root_dir = Path(args.directory).resolve()

    # Validate directory exists
    if not root_dir.exists():
        print(f"Error: Directory does not exist: {root_dir}")
        sys.exit(1)

    if not root_dir.is_dir():
        print(f"Error: Path is not a directory: {root_dir}")
        sys.exit(1)

    print(f"Processing directory: {root_dir}")

    # Build skip patterns
    skip_patterns = DEFAULT_SKIP_PATTERNS.copy()

    if not args.ignore_gitignore:
        gitignore_path = root_dir / ".gitignore"
        if gitignore_path.exists():
            gitignore_patterns = parse_gitignore(gitignore_path)
            skip_patterns.update(gitignore_patterns)
            print(f"Loaded {len(gitignore_patterns)} patterns from .gitignore")

    print()
    print("Generating INDEX.md files using LLM analysis...")
    print("(This may take a while depending on project size)")
    print()

    created = []
    updated = []
    skipped_unchanged = []

    # Discover all directories to process
    print(
        f"Discovering directories (max depth: {args.max_depth or 'unlimited'})..."
    )
    directories = discover_directories(root_dir, skip_patterns, args.max_depth)
    print(f"Found {len(directories)} directories to process")
    print()

    for directory in directories:
        try:
            rel_path = directory.relative_to(root_dir)
        except ValueError:
            rel_path = directory

        print(f"Processing {rel_path}/")

        index_path = directory / "INDEX.md"
        content = generate_index_content(
            directory, root_dir, skip_patterns
        )

        if index_path.exists():
            existing = index_path.read_text()
            if existing == content:
                print("  ⏭️  Unchanged\n")
                skipped_unchanged.append(rel_path)
            else:
                index_path.write_text(content)
                print("  ✏️  Updated\n")
                updated.append(rel_path)
        else:
            index_path.write_text(content)
            print("  ✅ Created\n")
            created.append(rel_path)

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Created: {len(created)}")
    print(f"  Updated: {len(updated)}")
    print(f"  Unchanged: {len(skipped_unchanged)}")
    print()

    if created:
        print("Created files:")
        for item in created[:10]:  # Show first 10
            print(f"  - {item}/INDEX.md")
        if len(created) > 10:
            print(f"  ... and {len(created) - 10} more")
        print()


if __name__ == "__main__":
    main()
