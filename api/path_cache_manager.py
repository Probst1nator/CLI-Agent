"""
Path Cache Manager for Autocomplete

Background caching of filesystem paths for intelligent autocomplete suggestions.
Scans recursively without limits, runs async, refreshes on user messages.
"""

import os
import asyncio
import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class PathCacheManager:
    """Manages background caching of filesystem paths for autocomplete"""

    # Directories to skip during scanning
    SKIP_DIRS = {
        '.git', 'node_modules', '__pycache__', '.venv', 'venv',
        'build', 'dist', '.next', '.nuxt', 'target', '.cache',
        '.idea', '.vscode', '.pytest_cache', '__MACOSX', '.tox'
    }

    # File patterns to skip
    SKIP_PATTERNS = {
        '.pyc', '.pyo', '.so', '.o', '.a', '.lib', '.dll',
        '.exe', '.bin', '.class', '.jar', '.war'
    }

    # Warn threshold for large caches
    CACHE_SIZE_WARN_THRESHOLD = 10000

    def __init__(self):
        """Initialize the path cache manager"""
        self.is_scanning = False
        self.scan_task = None
        self.cwd = os.getcwd()

        # Get globals reference
        try:
            from core.globals import g
            self.g = g
        except ImportError:
            logger.error("Failed to import globals, path caching disabled")
            self.g = None

        logger.info("PathCacheManager initialized")

    def start_initial_scan(self):
        """Start initial filesystem scan on startup"""
        if not self.g:
            return

        logger.info("Starting initial filesystem scan...")
        asyncio.create_task(self._scan_filesystem())

    def refresh_cache(self):
        """Trigger cache refresh (cancel existing scan, start new one)"""
        if not self.g:
            return

        # Cancel existing scan if running
        if self.scan_task and not self.scan_task.done():
            logger.debug("Cancelling existing filesystem scan")
            self.scan_task.cancel()

        # Start new scan
        logger.info("Triggering filesystem cache refresh")
        self.scan_task = asyncio.create_task(self._scan_filesystem())

    async def _scan_filesystem(self):
        """Background task to scan filesystem recursively"""
        if self.is_scanning:
            logger.debug("Scan already in progress, skipping")
            return

        self.is_scanning = True
        scan_start = asyncio.get_event_loop().time()

        try:
            paths = await self._walk_directory(self.cwd)

            # Update global cache
            self.g.CACHED_FILE_PATHS = list(paths)
            self.g.PATH_CACHE_TASK = self.scan_task

            scan_duration = asyncio.get_event_loop().time() - scan_start
            cache_size = len(paths)

            logger.info(f"Filesystem scan complete: {cache_size} paths cached in {scan_duration:.2f}s")

            # Warn if cache is very large
            if cache_size > self.CACHE_SIZE_WARN_THRESHOLD:
                logger.warning(
                    f"Path cache is large ({cache_size} paths). "
                    f"This may impact autocomplete performance. "
                    f"Consider using .gitignore patterns or reducing scan scope."
                )

        except asyncio.CancelledError:
            logger.info("Filesystem scan cancelled")
            raise
        except Exception as e:
            logger.error(f"Error during filesystem scan: {e}", exc_info=True)
        finally:
            self.is_scanning = False

    async def _walk_directory(self, root_path: str) -> Set[str]:
        """
        Recursively walk directory and collect all file/folder paths.
        Runs in executor to avoid blocking event loop.
        """
        paths = set()

        def _sync_walk():
            """Synchronous directory walk (runs in executor)"""
            nonlocal paths
            try:
                for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
                    # Skip hidden and excluded directories
                    dirnames[:] = [
                        d for d in dirnames
                        if not d.startswith('.') and d not in self.SKIP_DIRS
                    ]

                    # Add directories
                    for dirname in dirnames:
                        full_path = os.path.join(dirpath, dirname)
                        paths.add(full_path)

                    # Add files (skip binary/compiled files)
                    for filename in filenames:
                        if filename.startswith('.'):
                            continue
                        if any(filename.endswith(pattern) for pattern in self.SKIP_PATTERNS):
                            continue

                        full_path = os.path.join(dirpath, filename)
                        paths.add(full_path)

            except PermissionError as e:
                logger.debug(f"Permission denied scanning {dirpath}: {e}")
            except Exception as e:
                logger.warning(f"Error scanning directory {dirpath}: {e}")

        # Run blocking I/O in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_walk)

        return paths

    def get_cached_paths(self) -> List[str]:
        """Get current cached paths (thread-safe read)"""
        if not self.g:
            return []
        return self.g.CACHED_FILE_PATHS.copy()

    def is_cache_ready(self) -> bool:
        """Check if cache has been populated"""
        if not self.g:
            return False
        return len(self.g.CACHED_FILE_PATHS) > 0
