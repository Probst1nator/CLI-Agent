"""
Prompt Regenerator for CLI-Agent

Intelligently generates contextual prompt suggestions by analyzing
recently modified files in the user's Studium directory using LLM analysis.

This replaces the shell script approach with a Python implementation that
integrates properly with the CLI-Agent framework and logging system.
"""

import argparse
import asyncio
import json
import logging
import pickle
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

from rank_bm25 import BM25Plus

logger = logging.getLogger(__name__)


class PromptRegenerator:
    """
    Generates intelligent prompt suggestions based on recent study activities.
    """

    def __init__(
        self,
        studium_dir: str | None = None,
        prompts_file: str | None = None,
        num_files: int = 5,
        time_frame_days: int = 30,
    ):
        """
        Initialize the prompt regenerator.

        Args:
            studium_dir: Path to Studium directory (default: ~/OneDrive/Studium)
            prompts_file: Path to prompts JSON file (default: ~/.cli-agent/suggested_prompts.json)
            num_files: Number of random files to analyze (default: 5)
            time_frame_days: How many days back to look for modified files (default: 30)
        """
        self.studium_dir = Path(
            studium_dir or Path.home() / "OneDrive" / "Studium"
        ).expanduser()
        self.prompts_file = Path(
            prompts_file or Path.home() / ".cli-agent" / "suggested_prompts.json"
        ).expanduser()
        self.bm25_cache_dir = (
            Path.home() / ".cli-agent" / "bm25_cache" / "studium_index"
        )
        self.num_files = num_files
        self.time_frame_days = time_frame_days

        # Default prompts if generation fails (will have paths added by validate_and_fix_prompt_paths)
        self.default_prompts = [
            f"Help me understand a complex concept from my recent studies {self.studium_dir}",
            f"Create practice problems based on my coursework {self.studium_dir}",
            f"Explain connections between different topics I'm learning {self.studium_dir}",
            f"Generate study questions for my upcoming exam {self.studium_dir}",
            f"Summarize key points from my recent lecture notes {self.studium_dir}",
        ]

    def find_recent_files(self) -> list[tuple[Path, datetime]]:
        """
        Find recently modified files in Studium directory.

        Returns:
            List of (file_path, modification_time) tuples
        """
        if not self.studium_dir.exists():
            logger.warning(f"Studium directory not found: {self.studium_dir}")
            return []

        logger.info(
            f"Searching for files modified in last {self.time_frame_days} days in {self.studium_dir}"
        )

        # Allowed file extensions (whitelist)
        allowed_extensions = {
            # Text & Documents
            ".txt",
            ".md",
            ".pdf",
            # Code
            ".py",
            ".ipynb",
            ".cpp",
            ".c",
            ".h",
            ".java",
            ".js",
            ".ts",
            ".rs",
            ".go",
            # Images
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".gif",
            ".svg",
            # Office documents (proprietary)
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            # Office documents (open formats)
            ".odt",
            ".ods",
            ".odp",
            ".odg",
            # Data files
            ".csv",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            # Other
            ".tex",
            ".html",
            ".htm",
            ".css",
        }

        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(days=self.time_frame_days)
        cutoff_timestamp = cutoff_time.timestamp()

        files_with_mtime = []

        try:
            # Find directories containing pyvenv.cfg (Python virtual environments)
            venv_dirs = set()
            for cfg_path in self.studium_dir.rglob("pyvenv.cfg"):
                if cfg_path.is_file():
                    venv_dirs.add(cfg_path.parent)
                    logger.debug(f"Detected virtual environment: {cfg_path.parent}")

            if venv_dirs:
                logger.info(
                    f"Excluding {len(venv_dirs)} virtual environment directories"
                )

            # Recursively walk the directory tree
            for file_path in self.studium_dir.rglob("*"):
                # Skip directories
                if not file_path.is_file():
                    continue

                # Skip files in virtual environment directories
                if any(file_path.is_relative_to(venv_dir) for venv_dir in venv_dirs):
                    continue

                # Skip hidden files (any part of path starts with .)
                if any(part.startswith(".") for part in file_path.parts):
                    continue

                # Only include files with allowed extensions
                if file_path.suffix.lower() not in allowed_extensions:
                    continue

                try:
                    # Check modification time
                    stat = file_path.stat()
                    if stat.st_mtime >= cutoff_timestamp:
                        mtime = datetime.fromtimestamp(stat.st_mtime)
                        files_with_mtime.append((file_path, mtime))
                except (OSError, PermissionError) as e:
                    logger.debug(f"Failed to access {file_path}: {e}")
                    continue

            logger.info(f"Found {len(files_with_mtime)} recent files")
            return files_with_mtime

        except Exception as e:
            logger.error(f"Error finding files: {e}")
            return []

    def _tokenize_text(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def _read_file_content(self, file_path: Path) -> str | None:
        """
        Read text content from a file, handling various formats.

        Args:
            file_path: Path to file

        Returns:
            Text content or None if reading fails
        """
        try:
            suffix = file_path.suffix.lower()

            # Plain text files
            if suffix in {'.txt', '.md', '.py', '.cpp', '.c', '.h', '.java',
                         '.js', '.ts', '.rs', '.go', '.json', '.yaml', '.yml',
                         '.xml', '.tex', '.html', '.htm', '.css', '.csv'}:
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    return f.read()

            # PDF files
            elif suffix == '.pdf':
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    return text
                except Exception as e:
                    logger.debug(f"Failed to read PDF {file_path}: {e}")
                    return None

            # Office documents - use readfile tool's logic
            elif suffix in {'.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
                           '.odt', '.ods', '.odp'}:
                # For now, skip office docs in indexing (can be enhanced later)
                # TODO: Implement office doc text extraction for BM25
                logger.debug(f"Skipping office document for BM25: {file_path}")
                return None

            # Jupyter notebooks
            elif suffix == '.ipynb':
                try:
                    with open(file_path, encoding='utf-8') as f:
                        nb_data = json.load(f)
                        text_parts = []
                        for cell in nb_data.get('cells', []):
                            source = cell.get('source', [])
                            if isinstance(source, list):
                                text_parts.append(''.join(source))
                            else:
                                text_parts.append(source)
                        return '\n'.join(text_parts)
                except Exception as e:
                    logger.debug(f"Failed to read notebook {file_path}: {e}")
                    return None

            # Image files - skip for now
            elif suffix in {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.gif', '.svg'}:
                return None

            else:
                # Try reading as text
                try:
                    with open(file_path, encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except:
                    return None

        except Exception as e:
            logger.debug(f"Error reading {file_path}: {e}")
            return None

    def _get_all_files(self) -> list[tuple[Path, datetime]]:
        """
        Get all files from Studium directory (not just recent ones).

        Returns:
            List of (file_path, modification_time) tuples
        """
        if not self.studium_dir.exists():
            logger.warning(f"Studium directory not found: {self.studium_dir}")
            return []

        logger.info(f"Scanning all files in {self.studium_dir}")

        # Use same allowed extensions as find_recent_files
        allowed_extensions = {
            '.txt', '.md', '.pdf', '.py', '.ipynb', '.cpp', '.c', '.h',
            '.java', '.js', '.ts', '.rs', '.go', '.png', '.jpg', '.jpeg',
            '.tiff', '.tif', '.gif', '.svg', '.docx', '.doc', '.pptx',
            '.ppt', '.xlsx', '.xls', '.odt', '.ods', '.odp', '.odg',
            '.csv', '.json', '.yaml', '.yml', '.xml', '.tex', '.html',
            '.htm', '.css'
        }

        files_with_mtime = []

        try:
            # Find virtual environment directories
            venv_dirs = set()
            for cfg_path in self.studium_dir.rglob("pyvenv.cfg"):
                if cfg_path.is_file():
                    venv_dirs.add(cfg_path.parent)

            # Recursively walk directory
            for file_path in self.studium_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip venv directories
                if any(file_path.is_relative_to(venv_dir) for venv_dir in venv_dirs):
                    continue

                # Skip hidden files
                if any(part.startswith(".") for part in file_path.parts):
                    continue

                # Only include allowed extensions
                if file_path.suffix.lower() not in allowed_extensions:
                    continue

                try:
                    stat = file_path.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    files_with_mtime.append((file_path, mtime))
                except (OSError, PermissionError) as e:
                    logger.debug(f"Failed to access {file_path}: {e}")
                    continue

            logger.info(f"Found {len(files_with_mtime)} total files")
            return files_with_mtime

        except Exception as e:
            logger.error(f"Error scanning files: {e}")
            return []

    def _check_cache_validity(self, all_files: list[tuple[Path, datetime]]) -> bool:
        """
        Check if cached BM25 index is still valid.

        Args:
            all_files: List of current (file_path, mtime) tuples

        Returns:
            True if cache is valid, False otherwise
        """
        metadata_path = self.bm25_cache_dir / "metadata.json"

        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, encoding='utf-8') as f:
                metadata = json.load(f)

            cached_files = metadata.get('files', {})

            # Check if same number of files
            if len(cached_files) != len(all_files):
                logger.info("Cache invalid: file count changed")
                return False

            # Check if any file modified
            for file_path, mtime in all_files:
                file_str = str(file_path)
                cached_mtime = cached_files.get(file_str)

                if cached_mtime is None:
                    logger.info(f"Cache invalid: new file detected: {file_path}")
                    return False

                if mtime.timestamp() > cached_mtime:
                    logger.info(f"Cache invalid: file modified: {file_path}")
                    return False

            logger.info("BM25 cache is valid")
            return True

        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False

    def _save_bm25_cache(
        self,
        bm25_index: BM25Plus,
        file_paths: list[Path],
        all_files: list[tuple[Path, datetime]]
    ) -> None:
        """
        Save BM25 index and metadata to cache.

        Args:
            bm25_index: BM25Plus index object
            file_paths: List of file paths in index (same order as corpus)
            all_files: List of (file_path, mtime) tuples
        """
        try:
            # Create cache directory
            self.bm25_cache_dir.mkdir(parents=True, exist_ok=True)

            # Save index
            index_path = self.bm25_cache_dir / "index.pkl"
            with open(index_path, 'wb') as f:
                pickle.dump({
                    'bm25': bm25_index,
                    'file_paths': file_paths
                }, f)

            # Save metadata
            metadata = {
                'last_indexed': datetime.now().isoformat(),
                'files': {
                    str(file_path): mtime.timestamp()
                    for file_path, mtime in all_files
                }
            }

            metadata_path = self.bm25_cache_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved BM25 cache with {len(file_paths)} files")

        except Exception as e:
            logger.error(f"Failed to save BM25 cache: {e}")

    def _load_bm25_cache(self) -> tuple[BM25Plus, list[Path]] | None:
        """
        Load BM25 index from cache.

        Returns:
            Tuple of (BM25Plus index, file_paths) or None if loading fails
        """
        index_path = self.bm25_cache_dir / "index.pkl"

        if not index_path.exists():
            return None

        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)

            bm25_index = data['bm25']
            file_paths = data['file_paths']

            logger.info(f"Loaded BM25 cache with {len(file_paths)} files")
            return bm25_index, file_paths

        except Exception as e:
            logger.error(f"Failed to load BM25 cache: {e}")
            return None

    def build_or_load_bm25_index(self) -> tuple[BM25Plus, list[Path]] | None:
        """
        Build or load BM25 index for Studium directory.

        Returns:
            Tuple of (BM25Plus index, file_paths) or None if building fails
        """
        logger.info("Building or loading BM25 index...")

        # Get all files
        all_files = self._get_all_files()

        if not all_files:
            logger.warning("No files found for BM25 indexing")
            return None

        # Check cache validity
        if self._check_cache_validity(all_files):
            cached_data = self._load_bm25_cache()
            if cached_data:
                return cached_data

        # Build new index
        logger.info(f"Building BM25 index for {len(all_files)} files...")

        corpus = []
        file_paths = []

        for file_path, _ in all_files:
            content = self._read_file_content(file_path)

            if content and len(content.strip()) > 0:
                tokens = self._tokenize_text(content)
                if tokens:  # Only add if we got tokens
                    corpus.append(tokens)
                    file_paths.append(file_path)

        if not corpus:
            logger.warning("No readable content found for BM25 indexing")
            return None

        logger.info(f"Building BM25 index from {len(corpus)} documents...")
        bm25_index = BM25Plus(corpus)

        # Save to cache
        self._save_bm25_cache(bm25_index, file_paths, all_files)

        logger.info("BM25 index built successfully")
        return bm25_index, file_paths

    def search_files_with_bm25(
        self,
        query: str,
        top_n: int = 20,
        return_top_recent: int = 5
    ) -> list[tuple[Path, str]] | None:
        """
        Search files using BM25 and return top recent matches.

        Args:
            query: Search query (custom bias text)
            top_n: Number of top BM25 results to get (default: 20)
            return_top_recent: Number of most recent files to return (default: 5)

        Returns:
            List of (file_path, file_content) tuples for top recent matches,
            or None if search fails
        """
        logger.info(f"Searching files with BM25 for query: '{query}'")

        # Build or load index
        index_data = self.build_or_load_bm25_index()
        if not index_data:
            logger.error("Failed to build/load BM25 index")
            return None

        bm25_index, file_paths = index_data

        # Tokenize query
        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            logger.warning("Query produced no tokens")
            return None

        # Get BM25 scores
        scores = bm25_index.get_scores(query_tokens)

        # Get top N indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_n]

        logger.info(f"Got {len(top_indices)} BM25 results")

        # Get file paths with scores and modification times
        results_with_mtime = []
        for idx in top_indices:
            file_path = file_paths[idx]
            score = scores[idx]

            try:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                results_with_mtime.append((file_path, score, mtime))
            except (OSError, PermissionError) as e:
                logger.debug(f"Failed to get mtime for {file_path}: {e}")
                continue

        # Sort by modification time (most recent first)
        results_with_mtime.sort(key=lambda x: x[2], reverse=True)

        # Get top N most recent
        top_recent = results_with_mtime[:return_top_recent]

        logger.info(f"Selected {len(top_recent)} most recent files from BM25 results")

        # Read file contents
        results = []
        for file_path, score, mtime in top_recent:
            content = self._read_file_content(file_path)
            if content:
                results.append((file_path, content))
                logger.debug(f"  - {file_path} (score: {score:.2f}, modified: {mtime.strftime('%Y-%m-%d')})")

        return results if results else None

    def extract_relevant_content(
        self,
        content: str,
        query: str,
        max_chars: int = 5000,
        num_excerpts: int = 3
    ) -> str:
        """
        Extract relevant excerpts from large file content using BM25.

        Args:
            content: Full file content
            query: Search query to find relevant sections
            max_chars: Maximum characters before extracting excerpts (default: 5000)
            num_excerpts: Number of excerpts to extract (default: 3)

        Returns:
            Full content if small, or extracted excerpts if large
        """
        # If content is small enough, return as-is
        if len(content) <= max_chars:
            return content

        logger.debug(f"Extracting excerpts from large content ({len(content)} chars)")

        # Split content into paragraphs
        # Try multiple separators: double newline, single newline, periods
        paragraphs = []

        # First try double newlines (typical paragraph separator)
        if '\n\n' in content:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        # Fall back to single newlines
        elif '\n' in content:
            # Group consecutive lines into chunks of ~500 chars
            lines = content.split('\n')
            current_chunk = []
            current_length = 0

            for line in lines:
                current_chunk.append(line)
                current_length += len(line)

                if current_length >= 500:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        paragraphs.append(chunk_text)
                    current_chunk = []
                    current_length = 0

            # Add remaining chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    paragraphs.append(chunk_text)
        else:
            # No newlines, split by sentences (periods followed by space)
            paragraphs = [s.strip() for s in re.split(r'\.\s+', content) if s.strip()]

        if not paragraphs or len(paragraphs) == 1:
            # Can't split meaningfully, just truncate
            logger.debug("Could not split into paragraphs, truncating")
            return content[:max_chars] + "\n\n[... content truncated ...]"

        logger.debug(f"Split into {len(paragraphs)} paragraphs/chunks")

        # Tokenize paragraphs for BM25
        tokenized_paragraphs = [self._tokenize_text(p) for p in paragraphs]

        # Build BM25 index for paragraphs
        try:
            paragraph_bm25 = BM25Plus(tokenized_paragraphs)

            # Tokenize query
            query_tokens = self._tokenize_text(query)

            # Get scores
            scores = paragraph_bm25.get_scores(query_tokens)

            # Get top N paragraph indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:num_excerpts]

            # Sort indices to maintain document order
            top_indices.sort()

            # Extract excerpts
            excerpts = [paragraphs[i] for i in top_indices]

            # Join with clear separators
            result = '\n\n[...]\n\n'.join(excerpts)

            logger.debug(f"Extracted {len(excerpts)} relevant excerpts ({len(result)} chars)")
            return result

        except Exception as e:
            logger.warning(f"Failed to extract excerpts with BM25: {e}")
            # Fall back to truncation
            return content[:max_chars] + "\n\n[... content truncated ...]"

    def load_clicked_prompts(self) -> list[str]:
        """
        Load recently clicked prompts from the tracking file.

        Returns:
            List of recently clicked prompt strings (up to 3 most recent)
        """
        clicked_prompts_path = Path.home() / ".cli-agent" / "clicked_prompts.json"

        if not clicked_prompts_path.exists():
            logger.debug("No clicked prompts file found")
            return []

        try:
            with open(clicked_prompts_path, encoding="utf-8") as f:
                data = json.load(f)
                prompts = data.get("prompts", [])
                # Return up to 3 most recent
                recent_prompts = prompts[:3]
                logger.info(f"Loaded {len(recent_prompts)} recently clicked prompts")
                return recent_prompts
        except Exception as e:
            logger.warning(f"Failed to load clicked prompts: {e}")
            return []

    def parse_timetable_for_today(self) -> list[dict]:
        """
        Parse the Campo timetable and extract today's schedule.

        Returns:
            List of today's classes with time, course, and type.
            Format: [{"time": "17:00 - 18:30", "course": "...", "type": "..."}]
            Returns empty list if timetable not found or today not in schedule.
        """
        timetable_path = Path.home() / "OneDrive" / "Studium" / "campo_timetable.md"

        if not timetable_path.exists():
            logger.warning(f"Timetable file not found: {timetable_path}")
            return []

        try:
            # Get today's date and format
            today = datetime.now()
            # German weekday abbreviations: Mo., Di., Mi., Do., Fr., Sa., So.
            weekday_map = {
                0: "Mo.",  # Monday
                1: "Di.",  # Tuesday
                2: "Mi.",  # Wednesday
                3: "Do.",  # Thursday
                4: "Fr.",  # Friday
                5: "Sa.",  # Saturday
                6: "So.",  # Sunday
            }
            weekday_abbr = weekday_map[today.weekday()]
            date_str = today.strftime("%d.%m.%Y")  # e.g., "03.11.2025"
            today_header = f"## {weekday_abbr}, {date_str}"

            logger.info(f"Looking for timetable section: {today_header}")

            with open(timetable_path, encoding="utf-8") as f:
                content = f.read()

            # Find today's section
            if today_header not in content:
                logger.info(f"Today's date ({today_header}) not found in timetable")
                return []

            # Extract the section for today
            lines = content.split("\n")
            section_start = None
            section_lines = []

            for i, line in enumerate(lines):
                if line.strip() == today_header:
                    section_start = i
                elif section_start is not None:
                    # Stop at next ## heading or end of file
                    if line.startswith("## ") and i > section_start:
                        break
                    section_lines.append(line)

            if not section_lines:
                logger.info("No content found for today's section")
                return []

            # Parse the markdown table
            classes = []
            in_table = False

            for line in section_lines:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Detect table start (header line with Time | Course | ...)
                if "Time" in line and "|" in line:
                    in_table = True
                    continue

                # Skip separator line (|---|---|...)
                if in_table and line.startswith("|") and "---" in line:
                    continue

                # Parse table rows
                if in_table and line.startswith("|"):
                    cells = [cell.strip() for cell in line.split("|")]
                    # Filter out empty cells
                    cells = [c for c in cells if c]

                    # Table format: Time | Course | Room | Instructor | Type
                    if len(cells) >= 5:
                        time = cells[0]
                        course = cells[1]
                        class_type = cells[4]

                        classes.append(
                            {
                                "time": time,
                                "course": course,
                                "type": class_type,
                            }
                        )

            logger.info(f"Parsed {len(classes)} classes for today")
            return classes

        except Exception as e:
            logger.error(f"Error parsing timetable: {e}", exc_info=True)
            return []

    def build_llm_prompt(
        self,
        files: list[tuple[Path, datetime]],
        timetable: list[dict] | None = None,
        custom_bias_text: str | None = None,
        bm25_results: list[tuple[Path, str]] | None = None,
    ) -> str:
        """
        Build the LLM prompt for generating contextual suggestions.

        Args:
            files: List of (file_path, modification_time) tuples
            timetable: Optional list of today's classes from timetable
            custom_bias_text: Optional topic/subject to bias generation toward
            bm25_results: Optional list of (file_path, content) tuples from BM25 search

        Returns:
            Formatted prompt string
        """
        # Get current date info
        today = datetime.now()
        weekday_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        weekday = weekday_names[today.weekday()]
        date_str = today.strftime("%B %d, %Y")  # e.g., "November 3, 2025"
        current_time = today.strftime("%H:%M")

        file_list = "\n".join(
            [
                f"- {file_path} (modified: {mtime.strftime('%Y-%m-%d')})"
                for file_path, mtime in files
            ]
        )

        # Load recently clicked prompts for examples
        clicked_prompts = self.load_clicked_prompts()
        examples_section = ""
        if clicked_prompts:
            examples_list = "\n".join(
                [f'   - "{prompt}"' for prompt in clicked_prompts]
            )
            examples_section = f"""
## Examples of prompts the user previously clicked:
{examples_list}

Use these as inspiration for the style and type of prompts the user finds useful.
"""

        # Build timetable section if available
        timetable_section = ""
        if timetable:
            class_list = []
            for cls in timetable:
                class_list.append(
                    f"   - {cls['time']}: {cls['course']} ({cls['type']})"
                )

            timetable_section = f"""
## Today's Class Schedule ({weekday}):
{chr(10).join(class_list)}

**Instructions for timetable-aware prompts:**
- For classes within 2 hours of current time ({current_time}), include the specific time in the prompt
- For other classes today, focus on course topics and preparation
- Generate prompts for preparing, reviewing, or creating practice materials
- Mix timetable-based prompts (6-8 prompts) with general file-based prompts (2-4 prompts)
"""

        # Build custom bias section if provided
        custom_bias_section = ""
        if custom_bias_text:
            custom_bias_section = f"""
## 🎯 CUSTOM FOCUS AREA (HIGHEST PRIORITY)
The user has specifically requested prompts focused on: **"{custom_bias_text}"**

**CRITICAL INSTRUCTIONS:**
- Generate prompts that are DIRECTLY relevant to "{custom_bias_text}"
- Prioritize files and concepts that relate to this topic
- Frame ALL prompts around how this topic connects to the study materials
- If the topic is related to any of today's classes, emphasize that connection
- Still ensure prompts end with file/directory paths, but make the content highly relevant to "{custom_bias_text}"
- You MUST bias heavily toward this focus area - this is the user's explicit request
"""

        # Build BM25 file contents section if provided
        bm25_contents_section = ""
        if bm25_results:
            file_contents = []
            for file_path, content in bm25_results:
                # Extract excerpts from large files
                processed_content = self.extract_relevant_content(
                    content,
                    custom_bias_text or "",
                    max_chars=5000
                )

                # Get modification time
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    mtime_str = mtime.strftime("%Y-%m-%d")
                except:
                    mtime_str = "unknown"

                file_contents.append(f"""
### File: {file_path}
**Modified:** {mtime_str}
**Content:**
```
{processed_content}
```
""")

            bm25_contents_section = f"""
## 📄 RELEVANT FILE CONTENTS (BM25 Search Results)
The following files were identified as most relevant to the focus area "{custom_bias_text}".
Use these actual file contents to generate highly specific and contextual prompts.

{chr(10).join(file_contents)}

**Instructions for using these files:**
- Reference specific concepts, formulas, methods, or topics from the file contents
- Generate prompts that help the user engage with this specific material
- Mention file names when creating prompts about specific content
- These files are your PRIMARY source - use them extensively
"""

        n_prompts = 10
        prompt = f"""I need you to generate {n_prompts} contextual prompt suggestions based on my current study context.

## Current Date & Time
Today is {weekday}, {date_str} at {current_time}
{custom_bias_section}
{bm25_contents_section}
{timetable_section}
## Context: Recently modified files
Here are randomly selected files from the last {self.time_frame_days} days:
{file_list}
{examples_section}
## Your Task

1. **Understand my current focus**
   - Analyze the provided files to identify key topics and subjects
   - Explore the directory structure and read additional files within {self.studium_dir} to build richer context
   - Use your judgment on how much exploration is useful vs. excessive
   {"- Prioritize today's classes when generating prompts" if timetable and not custom_bias_text else ""}
   {"- MOST IMPORTANTLY: Focus all prompts on the custom topic above" if custom_bias_text else ""}

2. **Generate {n_prompts} prompts**
   - Each prompt should be exactly one sentence
   - **CRITICAL REQUIREMENT**: EVERY single prompt MUST end with a relevant file or directory path
   - For file-specific prompts: end with the exact file path from the context above
   - For general/timetable prompts: end with {self.studium_dir}
   - Format: "Your prompt sentence here /path/to/file.txt" (the path will be hidden from user display)
   - Example with file: "Explain the simulation methods covered in this lecture /home/user/Documents/lecture_notes.pdf"
   - Example with directory: "Review key concepts from today's Informatik lecture {self.studium_dir}"
   {"- Mix timetable-aware prompts with general study prompts as instructed above" if timetable and not custom_bias_text else ""}

3. **Output Format**
    Think about the context first then respond with a JSON object in this exact format:
   ```json
   {{
     "prompts": [
       "Prompt 1 here",
       "Prompt 2 here",
       "Prompt 3 here",
       ...
     ]
   }}
   ```

## Guidelines
- Be intelligent about inferring my real-world context and upcoming needs
- Think across subject boundaries - connect concepts from different domains
- Aim for prompts that are engaging and intellectually stimulating
{"- Use time-awareness: mention specific times for imminent classes, topics for later classes" if timetable and not custom_bias_text else ""}
- DO NOT include any text outside the JSON object"""

        return prompt

    def validate_and_fix_prompt_paths(self, prompts: list[str]) -> list[str]:
        """
        Validate that every prompt has a file/directory path. Add Studium path if missing.

        Args:
            prompts: List of generated prompts

        Returns:
            List of prompts with validated paths
        """
        import re

        fixed_prompts = []
        path_pattern = r"(/[^\s]+|~[^\s]*)"  # Matches absolute paths or ~ paths

        for i, prompt in enumerate(prompts):
            # Check if prompt contains a path (starts with / or ~)
            has_path = bool(re.search(path_pattern, prompt))

            if has_path:
                fixed_prompts.append(prompt)
                logger.debug(f"Prompt {i + 1}: Path detected ✓")
            else:
                # Add Studium directory path at the end
                fixed_prompt = f"{prompt.rstrip()} {self.studium_dir}"
                fixed_prompts.append(fixed_prompt)
                logger.warning(
                    f"Prompt {i + 1}: Missing path - added {self.studium_dir}"
                )

        return fixed_prompts

    async def generate_prompts_with_llm(
        self,
        files: list[tuple[Path, datetime]],
        timetable: list[dict] | None = None,
        custom_bias_text: str | None = None,
    ) -> list[str] | None:
        """
        Use LLM to generate contextual prompts based on file analysis.

        Args:
            files: List of (file_path, modification_time) tuples
            timetable: Optional list of today's classes from timetable
            custom_bias_text: Optional topic/subject to bias generation toward

        Returns:
            List of generated prompts, or None if generation fails
        """
        try:
            from core.llm_router import LlmRouter

            logger.info("Generating prompts using LLM...")

            # Perform BM25 search if custom bias text is provided
            bm25_results = None
            if custom_bias_text:
                logger.info(f"Performing BM25 search for custom bias: '{custom_bias_text}'")
                bm25_results = self.search_files_with_bm25(
                    query=custom_bias_text,
                    top_n=20,
                    return_top_recent=5
                )
                if bm25_results:
                    logger.info(f"BM25 search returned {len(bm25_results)} relevant files")
                else:
                    logger.warning("BM25 search returned no results, continuing without file contents")

            llm_router = LlmRouter()
            prompt = self.build_llm_prompt(files, timetable, custom_bias_text, bm25_results)

            # Print header for streamed output
            from termcolor import colored

            print(colored("\nGenerating prompts:\n", "cyan"))

            # Generate completion with streaming to console
            response = await llm_router.generate_completion(
                chat=prompt,  # Pass prompt directly as string
                generation_stream_callback=lambda char: print(
                    colored(char, "green"), end="", flush=True
                ),
            )

            print(colored("\n\n✅ Generation complete!\n", "cyan"))

            if not response:
                logger.error("LLM returned empty response")
                return None

            # Response is a string, not a dict
            content = response
            logger.debug(f"LLM response: {content[:200]}...")

            # Extract JSON from response
            # Look for JSON block between ```json and ```
            json_match = content
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    json_match = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    json_match = content[start:end].strip()

            # Try to find JSON object if no code blocks
            if "{" in json_match and "}" in json_match:
                start = json_match.find("{")
                end = json_match.rfind("}") + 1
                json_match = json_match[start:end]

            # Parse JSON
            try:
                data = json.loads(json_match)
                prompts = data.get("prompts", [])

                if not prompts or not isinstance(prompts, list):
                    logger.error("LLM response missing 'prompts' array")
                    return None

                if len(prompts) != 10:
                    logger.warning(f"Expected 10 prompts, got {len(prompts)}")

                # Validate and fix paths
                logger.info("Validating prompt paths...")
                prompts = self.validate_and_fix_prompt_paths(prompts)

                logger.info(f"Successfully generated {len(prompts)} prompts")
                return prompts

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.debug(f"JSON content: {json_match}")
                return None

        except ImportError as e:
            logger.error(f"Failed to import LLM dependencies: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating prompts with LLM: {e}", exc_info=True)
            return None

    def write_prompts_file(self, prompts: list[str]) -> bool:
        """
        Write prompts to JSON file.

        Args:
            prompts: List of prompt strings

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.prompts_file.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON file
            with open(self.prompts_file, "w", encoding="utf-8") as f:
                json.dump({"prompts": prompts}, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Successfully wrote {len(prompts)} prompts to {self.prompts_file}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to write prompts file: {e}")
            return False

    async def regenerate(self, custom_bias_text: str | None = None) -> dict:
        """
        Main entry point: regenerate prompts based on recent study activity.

        Args:
            custom_bias_text: Optional topic/subject to bias prompt generation toward

        Returns:
            Dict with status and prompts:
            {
                "status": "success" | "partial" | "error",
                "prompts": [...],
                "message": "...",
                "files_analyzed": N,
                "todays_classes": N
            }
        """
        logger.info("Starting prompt regeneration...")
        if custom_bias_text:
            logger.info(f"Custom bias text provided: '{custom_bias_text}'")
        start_time = datetime.now()

        # 1. Parse today's timetable
        timetable = self.parse_timetable_for_today()
        if timetable:
            logger.info(f"Found {len(timetable)} classes in today's schedule")
        else:
            logger.info(
                "No timetable data for today (continuing with file-based prompts)"
            )

        # 2. Find recent files
        files = self.find_recent_files()
        if not files:
            logger.warning("No recent files found, using default prompts")
            self.write_prompts_file(self.default_prompts)
            return {
                "status": "partial",
                "prompts": self.default_prompts,
                "message": f"No files found in {self.studium_dir} (last {self.time_frame_days} days). Using default prompts.",
                "files_analyzed": 0,
                "todays_classes": len(timetable) if timetable else 0,
            }

        # 3. Select random subset
        if len(files) > self.num_files:
            selected_files = random.sample(files, self.num_files)
            logger.info(f"Randomly selected {len(selected_files)} files for analysis")
        else:
            selected_files = files

        # 4. Generate prompts with LLM (including timetable context and custom bias)
        prompts = await self.generate_prompts_with_llm(
            selected_files, timetable, custom_bias_text
        )
        if not prompts:
            logger.warning("LLM generation failed, using default prompts")
            self.write_prompts_file(self.default_prompts)
            return {
                "status": "partial",
                "prompts": self.default_prompts,
                "message": "LLM generation failed. Using default prompts.",
                "files_analyzed": len(selected_files),
                "todays_classes": len(timetable) if timetable else 0,
            }

        # 5. Write to file
        if not self.write_prompts_file(prompts):
            return {
                "status": "error",
                "prompts": [],
                "message": "Failed to write prompts file",
                "files_analyzed": len(selected_files),
                "todays_classes": len(timetable) if timetable else 0,
            }

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Prompt regeneration completed in {elapsed:.1f}s")

        # Build success message
        message_parts = [
            f"Successfully generated {len(prompts)} prompts from {len(selected_files)} files"
        ]
        if timetable:
            message_parts.append(f"with {len(timetable)} classes from today's schedule")

        return {
            "status": "success",
            "prompts": prompts,
            "message": " ".join(message_parts),
            "files_analyzed": len(selected_files),
            "todays_classes": len(timetable) if timetable else 0,
            "elapsed_seconds": elapsed,
        }


def main():
    """CLI entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Regenerate CLI-Agent prompt suggestions from recent study files"
    )
    parser.add_argument(
        "--studium-dir",
        type=str,
        help="Path to Studium directory (default: ~/OneDrive/Studium)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=5,
        help="Number of random files to analyze (default: 5)",
    )
    parser.add_argument(
        "--time-frame-days",
        type=int,
        default=30,
        help="Days to look back for modified files (default: 30)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run regeneration
    regenerator = PromptRegenerator(
        studium_dir=args.studium_dir,
        num_files=args.num_files,
        time_frame_days=args.time_frame_days,
    )
    result = asyncio.run(regenerator.regenerate())

    # Print results
    if result["status"] == "success":
        print(f"\n✅ {result['message']}")
        print("\nGenerated prompts:")
        for i, prompt in enumerate(result["prompts"], 1):
            print(f"  {i}. {prompt}")
    elif result["status"] == "partial":
        print(f"\n⚠️  {result['message']}")
        print("\nDefault prompts:")
        for i, prompt in enumerate(result["prompts"], 1):
            print(f"  {i}. {prompt}")
    else:
        print(f"\n❌ {result['message']}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
