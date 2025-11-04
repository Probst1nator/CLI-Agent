#!/usr/bin/env python3
"""
Generate INDEX.md files for all directories in any project.

Each INDEX.md lists all files in the directory with functional descriptions
of their purpose in the codebase.

Usage:
    python3 generate_index_files.py                  # Heuristic mode (fast)
    python3 generate_index_files.py --use-llm        # LLM mode (accurate)
    python3 generate_index_files.py --use-llm --clear-cache
    python3 generate_index_files.py --max-depth 2    # Limit recursion depth
    python3 generate_index_files.py --root /path/to/project
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to import core modules (if available)
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def get_project_root(start_path: Path) -> Path:
    """
    Find project root by looking for common markers.

    Markers: .git, pyproject.toml, package.json, etc.
    """
    current = start_path.resolve()

    # Project root markers
    markers = {
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        ".gitignore",
    }

    while current != current.parent:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent

    # If no marker found, return original path
    return start_path.resolve()


def get_cache_file(project_root: Path) -> Path:
    """Get project-specific cache file path."""
    # Use project name in cache file
    project_name = project_root.name.lower().replace(" ", "_")
    cache_dir = Path.home() / ".cli-agent" / "index_cache"
    return cache_dir / f"{project_name}_descriptions.json"


def load_description_cache(cache_file: Path) -> dict[str, str]:
    """Load cached file descriptions."""
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_description_cache(cache: dict[str, str], cache_file: Path):
    """Save file descriptions to cache."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


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


def generate_file_description_with_llm(
    file_path: Path, file_contents: str, directory_context: str
) -> str:
    """
    Generate functional description using LLM by reading file contents.

    This function:
    1. Creates a Chat object
    2. Adds system message with instructions
    3. Adds user message with file contents
    4. Calls generate_completion() to get description
    """
    try:
        from core.chat import Chat, Role
        from core.llm_router import LlmRouter
    except ImportError as e:
        print(f"  Warning: Cannot import LLM modules: {e}")
        return None

    # Create chat for this request
    chat = Chat()

    # System message: Instructions for LLM
    system_message = """You are a technical documentation expert analyzing a codebase.

Generate a concise functional description of a file (max 100 characters).

Guidelines:
- Focus on WHAT the file does and WHY it exists in the codebase
- Use active voice (e.g., "Manages X", "Provides Y", "Implements Z")
- Be specific about functionality, not generic
- Avoid redundant information
- Include code quality observations

Examples of GOOD descriptions:
- "Permission settings management (load/save/check)"
- "Chat-related API endpoints (load, save, search, edit) - Includes unfinished mocks"
- "WebSocket communication manager with auto-reconnect - Misses typing and docstrings"

Examples of BAD descriptions:
- "Python module for settings" (too generic)
- "File that does chat stuff" (vague)

Output ONLY the description, nothing else."""

    # User message: File context
    user_message = f"""File: {file_path.name}
Directory: {directory_context}
Extension: {file_path.suffix}

File contents:
```
{file_contents[:8000]}
```

Generate a concise functional description (max 100 characters):"""

    # Build context chat with add_message calls
    chat.add_message(Role.SYSTEM, content=system_message)
    chat.add_message(Role.USER, content=user_message)

    try:
        # Generate completion
        print(
            f"    Generating description for {file_path.name}...", end=" ", flush=True
        )
        response = LlmRouter.generate_completion(
            chat=chat,
        )

        description = response.strip()
        if len(description) > 150:
            description = description[:147] + "..."

        print("✓")
        return description

    except Exception as e:
        print(f"✗ (Error: {e})")
        return None


def get_file_description(
    file_path: Path,
    dir_name: str,
    use_llm: bool = False,
    cache: Optional[dict[str, str]] = None,
) -> str:
    """
    Generate a functional description for a file.

    Args:
        file_path: Path to the file
        dir_name: Directory name containing the file
        use_llm: Whether to use LLM for description generation
        cache: Cache dictionary for storing/retrieving descriptions

    Returns:
        Functional description string
    """
    # Create cache key
    cache_key = str(file_path.relative_to(Path.cwd()))

    # Check cache first
    if cache and cache_key in cache:
        return cache[cache_key]

    name = file_path.name
    stem = file_path.stem
    ext = file_path.suffix

    # Special files that don't need LLM
    special_files = {"INDEX.md", "README.md", "__init__.py", ".gitignore"}

    # Try LLM if enabled and not a special file
    if use_llm and name not in special_files:
        file_contents = read_file_contents(file_path)
        if file_contents:
            llm_description = generate_file_description_with_llm(
                file_path, file_contents, dir_name
            )
            if llm_description:
                if cache is not None:
                    cache[cache_key] = llm_description
                return llm_description

    # Fall back to heuristic description
    description = _get_heuristic_description(file_path, dir_name)
    if cache is not None:
        cache[cache_key] = description
    return description


def _get_heuristic_description(file_path: Path, dir_name: str) -> str:
    """Heuristic-based description (fallback when LLM unavailable)."""
    name = file_path.name
    stem = file_path.stem
    ext = file_path.suffix

    # Special file types
    if name == "INDEX.md":
        return "Directory index with functional descriptions of all files"
    if name == "README.md":
        return "Overview and documentation for this directory"
    if name == "__init__.py":
        return "Python package initialization"
    if name == "CLAUDE.md":
        return "AI-assisted development guide and codebase rules"
    if name == "pyproject.toml":
        return "Python project configuration (dependencies, build system, tool configs)"
    if name == "package.json":
        return "Node.js project configuration and dependencies"
    if name == "tsconfig.json":
        return "TypeScript compiler configuration"
    if name == "vite.config.ts":
        return "Vite build tool configuration for frontend"
    if name == ".gitignore":
        return "Git ignore patterns for version control"
    if name == "requirements.txt":
        return "Python dependencies specification"
    if name == "Dockerfile":
        return "Docker container build instructions"
    if name.startswith("docker-compose"):
        return f"Docker Compose configuration for {stem.replace('docker-compose.', '')}"

    # Python files
    if ext == ".py":
        # Main entry points
        if stem == "main":
            return "Main application entry point"
        if stem.startswith("test_"):
            return f"Tests for {stem[5:]} module"
        if stem.endswith("_test"):
            return f"Tests for {stem[:-5]} module"

        # API files
        if dir_name == "api":
            if "server" in stem:
                return "Web server with WebSocket, session management, and API routing"
            if "chat" in stem:
                return "Chat-related API endpoints (load, save, search, edit)"
            if "audio" in stem:
                return "Audio/voice API endpoints (transcription, TTS)"
            if "tools" in stem:
                return "Tool discovery and configuration endpoints"
            if "profile" in stem:
                return "LLM profile management endpoints"
            if "agent_state" in stem:
                return "Agent state management endpoints (voice mode, auto-execute)"

        # Core files
        if dir_name == "core":
            if "chat" in stem:
                return "Core chat management (message handling, history, persistence)"
            if "llm" in stem or "router" in stem:
                return "LLM routing and provider management"
            if "cache" in stem:
                return "LLM response caching system"

        # Permission files
        if "permissions" in dir_name:
            if "settings" in stem:
                return "Permission settings management (load/save/check)"
            if "prompt" in stem:
                return "User permission prompts (CLI and GUI)"
            if "hints" in stem:
                return "Development tool detection and suggestions"

        # Provider files
        if "providers" in dir_name:
            if "anthropic" in stem:
                return "Claude API integration"
            if "openai" in stem:
                return "OpenAI GPT API integration"
            if "google" in stem or "gemini" in stem:
                return "Google Gemini API integration"
            if "groq" in stem:
                return "Groq API integration"
            if "ollama" in stem:
                return "Ollama local LLM integration"

        # Tool files
        if dir_name == "tools":
            if stem == "bash":
                return "Execute bash commands with permission checking"
            if "read" in stem:
                return "Read file contents"
            if "write" in stem:
                return "Write file contents"
            if "edit" in stem:
                return "Edit file contents (insert/delete lines)"
            if "search" in stem or "grep" in stem:
                return "Search codebase for patterns"
            if "web" in stem:
                return "Web search with LLM summarization"

        # Generic Python file
        return f"Python module: {stem.replace('_', ' ')}"

    # TypeScript/JavaScript
    if ext in {".ts", ".tsx"}:
        if stem == "main":
            return "Main application orchestrator"
        if "websocket" in stem:
            return "WebSocket communication manager"
        if "audio" in stem:
            return "Audio recording and TTS playback"
        if "chat" in stem:
            return "Chat state and message management"
        if "logger" in stem:
            return "Client-side logging system"
        return f"TypeScript module: {stem.replace('-', ' ')}"

    if ext == ".js":
        return f"JavaScript module: {stem.replace('-', ' ')}"

    # CSS
    if ext == ".css":
        return f"Stylesheet: {stem} styles"

    # HTML
    if ext == ".html":
        return f"HTML page: {stem}"

    # Markdown
    if ext == ".md":
        return f"Documentation: {stem.replace('_', ' ').replace('-', ' ')}"

    # Configuration
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".conf"}:
        return f"Configuration: {stem}"

    # Shell scripts
    if ext == ".sh":
        return f"Shell script: {stem.replace('_', ' ')}"

    # Default
    return f"{ext[1:].upper()} file: {stem}"


def generate_index_content(
    directory: Path,
    root: Path,
    skip_patterns: set[str],
    use_llm: bool = False,
    cache: Optional[dict[str, str]] = None,
) -> str:
    """Generate INDEX.md content for a directory."""
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
            desc = get_file_description(file, dir_name, use_llm=use_llm, cache=cache)
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

    if use_llm:
        lines.append("")
        lines.append("*Descriptions generated using LLM analysis of file contents.*")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate INDEX.md files for any project (dynamically discovers directories)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for accurate descriptions (slower but better)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached descriptions and regenerate all",
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Generate INDEX.md for specific directory only (no recursion)",
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Project root directory (default: auto-detect from CWD)",
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

    args = parser.parse_args()

    # Determine root directory
    if args.root:
        root_dir = Path(args.root).resolve()
    else:
        root_dir = get_project_root(Path.cwd())

    print(f"Project root: {root_dir}")

    # Get project-specific cache file
    cache_file = get_cache_file(root_dir)

    use_llm = args.use_llm

    # Load or clear cache
    if args.clear_cache:
        print("Clearing description cache...")
        cache = {}
    else:
        cache = load_description_cache(cache_file)
        if cache:
            print(f"Loaded {len(cache)} cached descriptions")

    # Build skip patterns
    skip_patterns = DEFAULT_SKIP_PATTERNS.copy()

    if not args.ignore_gitignore:
        gitignore_path = root_dir / ".gitignore"
        if gitignore_path.exists():
            gitignore_patterns = parse_gitignore(gitignore_path)
            skip_patterns.update(gitignore_patterns)
            print(f"Loaded {len(gitignore_patterns)} patterns from .gitignore")

    print()
    print("Generating INDEX.md files...")
    if use_llm:
        print("Using LLM for accurate descriptions (this may take a while)")
    else:
        print("Using heuristic descriptions (fast)")
    print()

    created = []
    updated = []
    skipped_unchanged = []

    # Determine which directories to process
    if args.directory:
        # Single directory mode
        target_dir = Path(args.directory)
        if not target_dir.is_absolute():
            target_dir = root_dir / target_dir

        if not target_dir.exists():
            print(f"⚠️  Directory not found: {target_dir}")
            return

        directories = [target_dir]
    else:
        # Auto-discover mode
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
            directory, root_dir, skip_patterns, use_llm=use_llm, cache=cache
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

    # Save cache if using LLM
    if use_llm:
        save_description_cache(cache, cache_file)
        print(f"\n💾 Saved {len(cache)} descriptions to cache")

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

    if use_llm:
        print(f"Cache file: {cache_file}")
        print("Run with --clear-cache to regenerate all descriptions")


if __name__ == "__main__":
    main()
