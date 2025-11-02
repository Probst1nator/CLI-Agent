"""
Centralized Temporary File Lifecycle Management System

Provides coordinated temporary file creation, tracking, and cleanup
across all CLI-Agent components to prevent file accumulation.
"""

import glob
import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

try:
    from core.globals import g
except ImportError:
    # Fallback for standalone usage with XDG paths
    class MockGlobals:
        CLIAGENT_CONFIG_DIR = os.path.expanduser("~/.config/cli-agent")
        CLIAGENT_DATA_DIR = os.path.expanduser("~/.local/share/cli-agent")
        CLIAGENT_CACHE_DIR = os.path.expanduser("~/.cache/cli-agent")
        CLIAGENT_TEMP_STORAGE_PATH = os.path.expanduser("~/.cache/cli-agent/temp")
        # Legacy paths for migration
        CLIAGENT_PERSISTENT_STORAGE_PATH = os.path.expanduser("~/.cli-agent")
    g = MockGlobals()


@dataclass
class TempFileInfo:
    """Information about a tracked temporary file."""
    file_path: str
    created_at: float
    category: str
    auto_cleanup: bool
    ttl_seconds: int


class FileManager:
    """
    Centralized temporary file lifecycle management system.

    Tracks all temporary files created by the application and provides
    automatic cleanup based on configurable policies.
    """

    # File categories with default TTL (time to live) in seconds
    CATEGORIES = {
        'python_temp': 24 * 3600,      # 24 hours - Python execution files
        'audio_debug': 2 * 3600,       # 2 hours - Audio debugging files
        'audio_temp': 1 * 3600,        # 1 hour - Temporary audio processing
        'screenshot': 7 * 24 * 3600,   # 7 days - Screenshot files
        'pdf_cache': 30 * 24 * 3600,   # 30 days - PDF processing cache
        'model_cache': 7 * 24 * 3600,  # 7 days - Model download cache
        'vector_db': 30 * 24 * 3600,   # 30 days - Vector database files
        'general': 24 * 3600           # 24 hours - General temporary files
    }

    def __init__(self):
        """Initialize the file manager."""
        self.tracked_files: dict[str, TempFileInfo] = {}
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_stop_event = threading.Event()
        self.lock = threading.Lock()

        # Ensure temp directories exist
        self._ensure_temp_directories()

    def _ensure_temp_directories(self):
        """Ensure all necessary XDG-compliant directories exist."""
        directories = [
            # XDG Cache directories
            g.CLIAGENT_CACHE_DIR,
            g.CLIAGENT_TEMP_STORAGE_PATH,
            os.path.join(g.CLIAGENT_CACHE_DIR, "audio_temp"),
            os.path.join(g.CLIAGENT_CACHE_DIR, "python_temp"),
            os.path.join(g.CLIAGENT_CACHE_DIR, "screenshots"),
            # XDG Data directories
            g.CLIAGENT_DATA_DIR,
            os.path.join(g.CLIAGENT_DATA_DIR, "vector_db"),
            os.path.join(g.CLIAGENT_DATA_DIR, "model_cache"),
            os.path.join(g.CLIAGENT_DATA_DIR, "pdf_cache"),
            # XDG Config directories
            g.CLIAGENT_CONFIG_DIR
        ]

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                logging.debug(f"Could not create directory {directory}: {e}")

    def create_temp_file(
        self,
        category: str = 'general',
        suffix: str = '',
        prefix: str = 'cliagent_',
        directory: Optional[str] = None,
        auto_cleanup: bool = True,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Create a tracked temporary file.

        Args:
            category: File category for cleanup policy
            suffix: File suffix (e.g., '.py', '.wav')
            prefix: File prefix
            directory: Specific directory, or None for category default
            auto_cleanup: Whether file should be automatically cleaned up
            ttl_seconds: Custom TTL, or None for category default

        Returns:
            Path to the created temporary file
        """
        if directory is None:
            if category == 'python_temp':
                directory = g.CLIAGENT_TEMP_STORAGE_PATH
            elif category in ['audio_debug', 'audio_temp']:
                directory = os.path.join(g.CLIAGENT_CACHE_DIR, "audio_temp")
            elif category == 'screenshot':
                directory = os.path.join(g.CLIAGENT_CACHE_DIR, "screenshots")
            elif category in ['pdf_cache', 'model_cache']:
                directory = os.path.join(g.CLIAGENT_DATA_DIR, category)
            elif category == 'vector_db':
                directory = os.path.join(g.CLIAGENT_DATA_DIR, "vector_db")
            else:
                directory = g.CLIAGENT_TEMP_STORAGE_PATH

        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=directory,
            delete=False
        ) as temp_file:
            temp_path = temp_file.name

        # Track the file
        ttl = ttl_seconds or self.CATEGORIES.get(category, self.CATEGORIES['general'])

        with self.lock:
            self.tracked_files[temp_path] = TempFileInfo(
                file_path=temp_path,
                created_at=time.time(),
                category=category,
                auto_cleanup=auto_cleanup,
                ttl_seconds=ttl
            )

        return temp_path

    def create_temp_directory(
        self,
        category: str = 'general',
        prefix: str = 'cliagent_',
        parent_directory: Optional[str] = None,
        auto_cleanup: bool = True,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Create a tracked temporary directory.

        Args:
            category: Directory category for cleanup policy
            prefix: Directory prefix
            parent_directory: Parent directory, or None for category default
            auto_cleanup: Whether directory should be automatically cleaned up
            ttl_seconds: Custom TTL, or None for category default

        Returns:
            Path to the created temporary directory
        """
        if parent_directory is None:
            parent_directory = g.CLIAGENT_TEMP_STORAGE_PATH

        # Ensure parent directory exists
        os.makedirs(parent_directory, exist_ok=True)

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=parent_directory)

        # Track the directory
        ttl = ttl_seconds or self.CATEGORIES.get(category, self.CATEGORIES['general'])

        with self.lock:
            self.tracked_files[temp_dir] = TempFileInfo(
                file_path=temp_dir,
                created_at=time.time(),
                category=category,
                auto_cleanup=auto_cleanup,
                ttl_seconds=ttl
            )

        return temp_dir

    @contextmanager
    def temp_file(
        self,
        category: str = 'general',
        suffix: str = '',
        auto_cleanup: bool = True,
        ttl_seconds: Optional[int] = None
    ):
        """
        Context manager for temporary files that ensures cleanup.

        Args:
            category: File category
            suffix: File suffix
            auto_cleanup: Whether to clean up automatically
            ttl_seconds: Custom TTL

        Yields:
            Path to temporary file
        """
        temp_path = self.create_temp_file(
            category=category,
            suffix=suffix,
            auto_cleanup=auto_cleanup,
            ttl_seconds=ttl_seconds
        )

        try:
            yield temp_path
        finally:
            if auto_cleanup:
                self.cleanup_file(temp_path, force=True)

    def track_existing_file(
        self,
        file_path: str,
        category: str = 'general',
        auto_cleanup: bool = True,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Start tracking an existing file.

        Args:
            file_path: Path to existing file
            category: File category
            auto_cleanup: Whether to clean up automatically
            ttl_seconds: Custom TTL
        """
        if not os.path.exists(file_path):
            return

        ttl = ttl_seconds or self.CATEGORIES.get(category, self.CATEGORIES['general'])

        with self.lock:
            self.tracked_files[file_path] = TempFileInfo(
                file_path=file_path,
                created_at=time.time(),
                category=category,
                auto_cleanup=auto_cleanup,
                ttl_seconds=ttl
            )

    def untrack_file(self, file_path: str) -> None:
        """Stop tracking a file without deleting it."""
        with self.lock:
            self.tracked_files.pop(file_path, None)

    def cleanup_file(self, file_path: str, force: bool = False) -> bool:
        """
        Clean up a specific file.

        Args:
            file_path: Path to file to clean up
            force: Force cleanup regardless of TTL

        Returns:
            True if file was cleaned up, False otherwise
        """
        with self.lock:
            file_info = self.tracked_files.get(file_path)

        if not file_info and not force:
            return False

        if file_info and not file_info.auto_cleanup and not force:
            return False

        # Check TTL
        if file_info and not force:
            age_seconds = time.time() - file_info.created_at
            if age_seconds < file_info.ttl_seconds:
                return False

        # Remove file or directory
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.exists(file_path):
                os.remove(file_path)

            # Remove from tracking
            with self.lock:
                self.tracked_files.pop(file_path, None)

            return True
        except OSError as e:
            logging.debug(f"Could not clean up {file_path}: {e}")
            return False

    def cleanup_by_category(
        self,
        category: str,
        force_all: bool = False
    ) -> dict[str, Any]:
        """
        Clean up all files in a specific category.

        Args:
            category: Category to clean up
            force_all: Force cleanup regardless of TTL

        Returns:
            Cleanup statistics
        """
        stats = {'files_removed': 0, 'bytes_freed': 0, 'errors': []}
        current_time = time.time()

        files_to_cleanup = []
        with self.lock:
            for file_path, file_info in self.tracked_files.items():
                if file_info.category == category:
                    age_seconds = current_time - file_info.created_at
                    if force_all or (file_info.auto_cleanup and age_seconds >= file_info.ttl_seconds):
                        files_to_cleanup.append(file_path)

        for file_path in files_to_cleanup:
            try:
                if os.path.exists(file_path):
                    if os.path.isdir(file_path):
                        # Calculate directory size
                        dir_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(file_path)
                            for filename in filenames
                        )
                        shutil.rmtree(file_path)
                        stats['bytes_freed'] += dir_size
                    else:
                        stats['bytes_freed'] += os.path.getsize(file_path)
                        os.remove(file_path)
                    stats['files_removed'] += 1

                # Remove from tracking
                with self.lock:
                    self.tracked_files.pop(file_path, None)

            except OSError as e:
                stats['errors'].append(f"{file_path}: {str(e)}")

        return stats

    def cleanup_expired_files(self) -> dict[str, Any]:
        """
        Clean up all expired files across all categories.

        Returns:
            Cleanup statistics
        """
        stats = {'files_removed': 0, 'bytes_freed': 0, 'errors': [], 'by_category': {}}
        current_time = time.time()

        # Group files by category for cleanup
        files_by_category = {}
        with self.lock:
            for file_path, file_info in self.tracked_files.items():
                if file_info.auto_cleanup:
                    age_seconds = current_time - file_info.created_at
                    if age_seconds >= file_info.ttl_seconds:
                        category = file_info.category
                        if category not in files_by_category:
                            files_by_category[category] = []
                        files_by_category[category].append(file_path)

        # Clean up by category
        for category, file_paths in files_by_category.items():
            category_stats = {'files_removed': 0, 'bytes_freed': 0, 'errors': []}

            for file_path in file_paths:
                try:
                    if os.path.exists(file_path):
                        if os.path.isdir(file_path):
                            # Calculate directory size
                            dir_size = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(file_path)
                                for filename in filenames
                            )
                            shutil.rmtree(file_path)
                            category_stats['bytes_freed'] += dir_size
                        else:
                            category_stats['bytes_freed'] += os.path.getsize(file_path)
                            os.remove(file_path)
                        category_stats['files_removed'] += 1

                    # Remove from tracking
                    with self.lock:
                        self.tracked_files.pop(file_path, None)

                except OSError as e:
                    category_stats['errors'].append(f"{file_path}: {str(e)}")

            # Aggregate stats
            stats['files_removed'] += category_stats['files_removed']
            stats['bytes_freed'] += category_stats['bytes_freed']
            stats['errors'].extend(category_stats['errors'])
            stats['by_category'][category] = category_stats

        return stats

    def cleanup_legacy_files(self) -> dict[str, Any]:
        """
        Clean up legacy files that weren't tracked by the file manager.
        This extends the existing cleanup_temp_py_files functionality.

        Returns:
            Cleanup statistics
        """
        stats = {'files_removed': 0, 'bytes_freed': 0, 'errors': []}

        # Legacy cleanup patterns and locations (both old and new structures)
        cleanup_patterns = [
            # Python temp files (both old and new locations)
            (g.CLIAGENT_TEMP_STORAGE_PATH, r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.py$'),
            (os.path.join(g.CLIAGENT_CACHE_DIR, "python_temp"), r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.py$'),
            # Audio debug files (system tmp and cache directories)
            ('/tmp', 'faster_whisper_debug_*.wav'),
            ('/tmp', 'faster_whisper_input_*.wav'),
            ('/tmp', 'boosted_*.wav'),
            ('/tmp', 'sr_test.wav'),
            (os.path.join(g.CLIAGENT_CACHE_DIR, "audio_temp"), 'debug_audio_*.wav'),
            # Legacy audio files in old location
            (getattr(g, 'CLIAGENT_PERSISTENT_STORAGE_PATH', ''), 'debug_audio_*.wav'),
            # Screenshot files
            (os.path.join(g.CLIAGENT_CACHE_DIR, "screenshots"), 'screenshot_*.png'),
            # Legacy screenshot location
            (os.path.join(getattr(g, 'CLIAGENT_PERSISTENT_STORAGE_PATH', ''), 'sandbox', 'screenshots'), 'screenshot_*.png'),
            # Old cache files in legacy location
            (os.path.join(getattr(g, 'CLIAGENT_PERSISTENT_STORAGE_PATH', ''), 'cache'), '*'),
        ]

        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days

        for directory, pattern in cleanup_patterns:
            if not os.path.exists(directory):
                continue

            try:
                if '*' in pattern:
                    # Glob pattern
                    files = glob.glob(os.path.join(directory, pattern))
                else:
                    # Regex pattern
                    import re
                    regex = re.compile(pattern)
                    files = [
                        os.path.join(directory, f)
                        for f in os.listdir(directory)
                        if regex.search(f)
                    ]

                for file_path in files:
                    try:
                        if os.path.isfile(file_path):
                            file_age = os.path.getmtime(file_path)
                            if file_age < cutoff_time:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                stats['files_removed'] += 1
                                stats['bytes_freed'] += file_size
                    except OSError as e:
                        stats['errors'].append(f"{file_path}: {str(e)}")

            except OSError as e:
                stats['errors'].append(f"Directory {directory}: {str(e)}")

        return stats

    def get_file_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about tracked files."""
        stats = {
            'total_tracked': 0,
            'by_category': {},
            'total_size_estimate': 0,
            'expired_files': 0,
            'auto_cleanup_files': 0
        }

        current_time = time.time()

        with self.lock:
            for file_path, file_info in self.tracked_files.items():
                stats['total_tracked'] += 1

                # Category stats
                category = file_info.category
                if category not in stats['by_category']:
                    stats['by_category'][category] = {
                        'count': 0,
                        'expired': 0,
                        'auto_cleanup': 0,
                        'size_estimate': 0
                    }

                stats['by_category'][category]['count'] += 1

                # Check if expired
                age_seconds = current_time - file_info.created_at
                if age_seconds >= file_info.ttl_seconds:
                    stats['expired_files'] += 1
                    stats['by_category'][category]['expired'] += 1

                if file_info.auto_cleanup:
                    stats['auto_cleanup_files'] += 1
                    stats['by_category'][category]['auto_cleanup'] += 1

                # Size estimate
                try:
                    if os.path.exists(file_path):
                        if os.path.isdir(file_path):
                            dir_size = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(file_path)
                                for filename in filenames
                            )
                            stats['total_size_estimate'] += dir_size
                            stats['by_category'][category]['size_estimate'] += dir_size
                        else:
                            file_size = os.path.getsize(file_path)
                            stats['total_size_estimate'] += file_size
                            stats['by_category'][category]['size_estimate'] += file_size
                except OSError:
                    pass

        return stats

    def start_periodic_cleanup(self, interval_minutes: int = 60) -> None:
        """Start background thread for periodic cleanup."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        self.cleanup_stop_event.clear()
        self.cleanup_thread = threading.Thread(
            target=self._periodic_cleanup_worker,
            args=(interval_minutes,),
            daemon=True
        )
        self.cleanup_thread.start()

    def stop_periodic_cleanup(self) -> None:
        """Stop background cleanup thread."""
        if self.cleanup_thread:
            self.cleanup_stop_event.set()
            self.cleanup_thread.join(timeout=5.0)

    def _periodic_cleanup_worker(self, interval_minutes: int) -> None:
        """Background worker for periodic cleanup."""
        interval_seconds = interval_minutes * 60

        while not self.cleanup_stop_event.is_set():
            try:
                # Wait for stop event or timeout
                if self.cleanup_stop_event.wait(interval_seconds):
                    break

                # Perform cleanup
                self.cleanup_expired_files()
                self.cleanup_legacy_files()

            except Exception as e:
                logging.debug(f"Periodic file cleanup error: {e}")
                # Wait shorter interval on error
                if self.cleanup_stop_event.wait(600):  # 10 minutes
                    break

    def shutdown(self) -> dict[str, Any]:
        """Gracefully shutdown the file manager and perform final cleanup."""
        # Stop periodic cleanup
        self.stop_periodic_cleanup()

        # Perform final cleanup
        expired_stats = self.cleanup_expired_files()
        legacy_stats = self.cleanup_legacy_files()

        return {
            'expired_cleanup': expired_stats,
            'legacy_cleanup': legacy_stats,
            'total_files_removed': expired_stats['files_removed'] + legacy_stats['files_removed'],
            'total_bytes_freed': expired_stats['bytes_freed'] + legacy_stats['bytes_freed']
        }


# Global instance
_file_manager = None


def get_file_manager() -> FileManager:
    """Get the global file manager instance."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager


# Convenience functions for common operations
def create_temp_file(category: str = 'general', suffix: str = '', **kwargs) -> str:
    """Create a temporary file using the global file manager."""
    return get_file_manager().create_temp_file(category=category, suffix=suffix, **kwargs)


def create_temp_directory(category: str = 'general', **kwargs) -> str:
    """Create a temporary directory using the global file manager."""
    return get_file_manager().create_temp_directory(category=category, **kwargs)


def temp_file(category: str = 'general', suffix: str = '', **kwargs):
    """Context manager for temporary files."""
    return get_file_manager().temp_file(category=category, suffix=suffix, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test the file manager
    manager = FileManager()

    print("🧪 Testing File Manager...")
    print("=" * 50)

    # Create some test files
    test_py_file = manager.create_temp_file('python_temp', '.py')
    test_audio_file = manager.create_temp_file('audio_debug', '.wav')
    test_dir = manager.create_temp_directory('general')

    print("📝 Created test files:")
    print(f"  Python: {test_py_file}")
    print(f"  Audio: {test_audio_file}")
    print(f"  Directory: {test_dir}")

    # Get statistics
    stats = manager.get_file_statistics()
    print("\n📊 Statistics:")
    print(f"  Total tracked: {stats['total_tracked']}")
    print(f"  Auto cleanup: {stats['auto_cleanup_files']}")
    print(f"  Size estimate: {stats['total_size_estimate']} bytes")

    # Test context manager
    with manager.temp_file('general', '.txt') as temp_path:
        with open(temp_path, 'w') as f:
            f.write("Test content")
        print(f"📝 Context manager file: {temp_path}")
        print(f"  Exists during context: {os.path.exists(temp_path)}")
    print(f"  Exists after context: {os.path.exists(temp_path)}")

    # Test cleanup
    cleanup_stats = manager.cleanup_by_category('python_temp', force_all=True)
    print("\n🧹 Cleanup results:")
    print(f"  Files removed: {cleanup_stats['files_removed']}")
    print(f"  Bytes freed: {cleanup_stats['bytes_freed']}")

    print("\n✅ File manager test completed!")