# tools/searchfile.py
"""
Implements the 'searchfile' tool for intelligent file and directory searching.

Directory mode: Uses BM25 ranking to find the most relevant files containing the query.
File mode: Searches through file contents and returns matching lines with line numbers.
"""

import os
import re
from pathlib import Path

# Set up tool-specific logging
from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)

# Try to import BM25, handle gracefully if not available
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed. Directory search will use fallback method.")


class searchfile:
    """
    Unified search tool supporting both directory-level and file-level searches.
    """

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool."""
        return 'searchfile'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "searchfile",
            "description": (
                "Search for content in files/directories using BM25 ranking. "
                "Directory mode finds top matching files. File mode returns matching lines with line numbers and context. "
                "Use <help>searchfile</help> for advanced options and pattern syntax."
            ),
            "example": (
                "<searchfile><query>authentication logic</query><path>./src</path></searchfile>\n"
                "<searchfile><query>error|exception</query><path>./app.py</path></searchfile>"
            )
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SEARCHFILE TOOL - EXTENDED HELP                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  <searchfile><query>search terms</query><path>./directory</path></searchfile>
  <searchfile><query>regex pattern</query><path>./file.py</path></searchfile>

TWO OPERATING MODES:

1. DIRECTORY MODE (path is a directory)
   - Searches all files in directory tree
   - Uses BM25 ranking algorithm for relevance
   - Returns top-k most relevant files
   - Respects .gitignore patterns

2. FILE MODE (path is a file)
   - Searches within a single file
   - Returns matching lines with line numbers
   - Supports regex patterns
   - Shows context around matches

PARAMETERS:
  <query>         - Search query or regex pattern [REQUIRED]
  <path>          - File or directory path [REQUIRED]
  <top_k>         - Number of results for directory mode (default: 10) [OPTIONAL]

DIRECTORY MODE EXAMPLES:

1. Basic directory search:
   <searchfile>
     <query>authentication logic</query>
     <path>./src</path>
   </searchfile>

2. Search with more results:
   <searchfile>
     <query>database connection</query>
     <path>./src</path>
     <top_k>20</top_k>
   </searchfile>

FILE MODE EXAMPLES:

1. Search within a file:
   <searchfile>
     <query>error|exception</query>
     <path>./app.py</path>
   </searchfile>

2. Regex pattern search:
   <searchfile>
     <query>class \w+Controller</query>
     <path>./controllers.py</path>
   </searchfile>

QUERY SYNTAX:

DIRECTORY MODE:
  - Simple keywords: "authentication database"
  - Multiple terms matched with BM25 ranking
  - Case-insensitive by default

FILE MODE:
  - Supports regex patterns
  - Use | for OR: "error|exception|warning"
  - Use \w, \d, \s for character classes
  - Case-insensitive by default

BM25 RANKING ALGORITHM:

Directory mode uses BM25 (Best Matching 25), a state-of-the-art ranking algorithm:
  - Considers term frequency in documents
  - Accounts for document length
  - Weights rare terms higher than common ones
  - Returns results sorted by relevance score

PERFORMANCE TIPS:

1. Use specific paths instead of searching from root
2. Use directory mode to identify relevant files first
3. Then use file mode on specific files for exact locations
4. Binary files are automatically skipped

WORKFLOW RECOMMENDATIONS:

1. DISCOVERY: Use directory mode to find relevant files
   <searchfile><query>login validation</query><path>./src</path></searchfile>

2. INSPECTION: Use file mode to find exact locations
   <searchfile><query>def validate_login</query><path>./src/auth.py</path></searchfile>

3. READING: Use readfile to get the actual content
   <readfile><filepath>./src/auth.py</filepath><lines>45-75</lines></readfile>

LIMITATIONS:
  - Directory mode requires rank_bm25 library (fallback available)
  - Very large files may take longer to search
  - Binary files are skipped automatically
  - Regex in directory mode not supported (file mode only)

COMPARISON WITH OTHER TOOLS:

searchfile vs grepsearch:
  - searchfile: BM25 ranking, better for finding relevant files
  - grepsearch: Traditional grep, better for exact pattern matching

searchfile vs readfile:
  - searchfile: Find and preview matches
  - readfile: Read full file content

Use searchfile first to locate, then readfile to inspect!

╚══════════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def _parse_xml(content: str) -> dict:
        """Parse XML parameters from tool content."""
        params = {}

        # Required: query
        query_match = re.search(r'<query>(.*?)</query>', content, re.DOTALL)
        if query_match:
            params['query'] = query_match.group(1).strip()

        # Required: path
        path_match = re.search(r'<path>(.*?)</path>', content, re.DOTALL)
        if path_match:
            params['path'] = path_match.group(1).strip()

        # Optional: top_k (for directory search)
        top_k_match = re.search(r'<top_k>(\d+)</top_k>', content, re.DOTALL)
        if top_k_match:
            params['top_k'] = int(top_k_match.group(1))
        else:
            params['top_k'] = 10  # Default

        return params

    @staticmethod
    def _should_ignore_file(filepath: str) -> bool:
        """Check if file should be ignored (binary, cache, etc.)."""
        ignore_patterns = {
            # Directories
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            'build', 'dist', '.pytest_cache', '.mypy_cache',
            # File extensions
            '.pyc', '.pyo', '.so', '.dll', '.dylib', '.exe',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
            '.pdf', '.zip', '.tar', '.gz', '.bz2',
            '.mp3', '.mp4', '.avi', '.mov', '.wav',
        }

        filepath_lower = filepath.lower()

        # Check if any part of path matches ignore patterns
        for pattern in ignore_patterns:
            if pattern in filepath_lower:
                return True

        return False

    @staticmethod
    def _search_directory_bm25(query: str, directory: str, top_k: int) -> str:
        """Search directory using BM25 ranking."""
        if not BM25_AVAILABLE:
            return searchfile._search_directory_fallback(query, directory, top_k)

        logger.info(f"BM25 directory search: query='{query}', path='{directory}', top_k={top_k}")

        try:
            # Collect all text files
            documents = []
            file_paths = []

            dir_path = Path(directory).resolve()
            if not dir_path.exists():
                return f"Error: Directory not found: {directory}"

            # Walk through directory
            for root, dirs, files in os.walk(dir_path):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if not searchfile._should_ignore_file(d)]

                for file in files:
                    filepath = os.path.join(root, file)

                    # Skip ignored files
                    if searchfile._should_ignore_file(filepath):
                        continue

                    try:
                        # Read file content
                        with open(filepath, encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Tokenize content for BM25
                        tokens = content.lower().split()
                        documents.append(tokens)
                        file_paths.append(filepath)

                    except Exception as e:
                        logger.debug(f"Could not read file {filepath}: {e}")
                        continue

            if not documents:
                return f"No readable text files found in {directory}"

            # Create BM25 index
            bm25 = BM25Okapi(documents)

            # Search
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)

            # Get top_k results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            # Filter out zero-score results
            results = [(file_paths[i], scores[i]) for i in top_indices if scores[i] > 0]

            if not results:
                return f"No files found matching query '{query}' in {directory}"

            # Format output
            output = [f"Top {len(results)} files matching '{query}' (BM25 search):\n"]

            for rank, (filepath, score) in enumerate(results, 1):
                rel_path = os.path.relpath(filepath, dir_path)
                output.append(f"{rank}. {rel_path} (score: {score:.2f})")

            logger.info(f"Found {len(results)} matching files")
            return '\n'.join(output)

        except Exception as e:
            logger.error(f"BM25 search failed: {e}", exc_info=True)
            return f"Error during directory search: {str(e)}"

    @staticmethod
    def _search_directory_fallback(query: str, directory: str, top_k: int) -> str:
        """Fallback directory search using simple grep-like matching."""
        logger.info(f"Fallback directory search: query='{query}', path='{directory}'")

        try:
            dir_path = Path(directory).resolve()
            if not dir_path.exists():
                return f"Error: Directory not found: {directory}"

            results = []
            pattern = re.compile(query, re.IGNORECASE)

            # Walk through directory
            for root, dirs, files in os.walk(dir_path):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if not searchfile._should_ignore_file(d)]

                for file in files:
                    filepath = os.path.join(root, file)

                    # Skip ignored files
                    if searchfile._should_ignore_file(filepath):
                        continue

                    try:
                        with open(filepath, encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Count matches
                        matches = len(pattern.findall(content))
                        if matches > 0:
                            results.append((filepath, matches))

                    except Exception:
                        continue

            if not results:
                return f"No files found matching query '{query}' in {directory}"

            # Sort by match count
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

            # Format output
            output = [f"Top {len(results)} files matching '{query}' (fallback search):\n"]
            output.append("⚠️  Note: Install rank-bm25 for better search quality: pip install rank-bm25\n")

            for rank, (filepath, count) in enumerate(results, 1):
                rel_path = os.path.relpath(filepath, dir_path)
                output.append(f"{rank}. {rel_path} ({count} matches)")

            return '\n'.join(output)

        except Exception as e:
            logger.error(f"Fallback search failed: {e}", exc_info=True)
            return f"Error during directory search: {str(e)}"

    @staticmethod
    def _search_file(query: str, filepath: str) -> str:
        """Search within a single file and return matching lines with line numbers."""
        logger.info(f"File search: query='{query}', file='{filepath}'")

        try:
            file_path = Path(filepath).resolve()

            if not file_path.exists():
                return f"Error: File not found: {filepath}"

            if not file_path.is_file():
                return f"Error: Path is not a file: {filepath}"

            # Compile regex pattern
            try:
                pattern = re.compile(query, re.IGNORECASE)
            except re.error as e:
                return f"Error: Invalid regex pattern '{query}': {str(e)}"

            # Read file and search
            matches = []
            try:
                with open(file_path, encoding='utf-8', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern.search(line):
                            matches.append((line_num, line.rstrip('\n')))
            except Exception as e:
                return f"Error reading file: {str(e)}"

            if not matches:
                return f"No matches found for '{query}' in {file_path.name}"

            # Format output
            filename = file_path.name
            output = [f"Found {len(matches)} match{'es' if len(matches) != 1 else ''} in {filename}:\n"]

            # Limit to first 50 matches to avoid overwhelming output
            display_matches = matches[:50]
            for line_num, line in display_matches:
                # Truncate very long lines
                if len(line) > 200:
                    line = line[:197] + "..."
                output.append(f"Line {line_num}: {line}")

            if len(matches) > 50:
                output.append(f"\n... and {len(matches) - 50} more matches (showing first 50)")

            logger.info(f"Found {len(matches)} matches in file")
            return '\n'.join(output)

        except Exception as e:
            logger.error(f"File search failed: {e}", exc_info=True)
            return f"Error during file search: {str(e)}"

    @staticmethod
    def run(content: str) -> str:
        """
        Main entry point for the searchfile tool.

        Automatically detects whether to search a directory or a file based on the path.
        """
        try:
            # Parse parameters
            params = searchfile._parse_xml(content)

            # Validate required parameters
            if 'query' not in params or not params['query']:
                return "Error: Missing required parameter <query>"

            if 'path' not in params or not params['path']:
                return "Error: Missing required parameter <path>"

            query = params['query']
            path = params['path']
            top_k = params['top_k']

            # Resolve path
            resolved_path = Path(path).resolve()

            # Determine mode based on path type
            if resolved_path.is_dir():
                return searchfile._search_directory_bm25(query, str(resolved_path), top_k)
            elif resolved_path.is_file():
                return searchfile._search_file(query, str(resolved_path))
            else:
                return f"Error: Path does not exist: {path}"

        except Exception as e:
            logger.error(f"searchfile tool error: {e}", exc_info=True)
            return f"Error in searchfile tool: {str(e)}"
