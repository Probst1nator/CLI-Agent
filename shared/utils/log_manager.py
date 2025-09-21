# -*- coding: utf-8 -*-
"""
Centralized Log Retention and Cleanup Management System

Provides automatic log rotation, retention policies, and cleanup scheduling
for all CLI-Agent logging operations.
"""

import logging
import os
import threading
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from core.globals import g
except ImportError:
    # Fallback for standalone usage with XDG paths
    class MockGlobals:
        CLIAGENT_CACHE_DIR = os.path.expanduser("~/.cache/cli-agent")
        # Legacy path for compatibility
        CLIAGENT_PERSISTENT_STORAGE_PATH = os.path.expanduser("~/.cli-agent")
    g = MockGlobals()


@dataclass
class LogRetentionConfig:
    """Configuration for log retention policies."""
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    max_age_days: int = 7
    cleanup_interval_hours: int = 24


class LogRetentionManager:
    """
    Centralized log retention and cleanup management system.

    Handles automatic log rotation, old log cleanup, and provides
    statistics about log file usage across the application.
    """

    # Default retention policies for different log types
    DEFAULT_POLICIES = {
        'application': LogRetentionConfig(
            max_bytes=10 * 1024 * 1024,  # 10MB
            backup_count=5,
            max_age_days=7,
            cleanup_interval_hours=24
        ),
        'debug': LogRetentionConfig(
            max_bytes=5 * 1024 * 1024,   # 5MB
            backup_count=3,
            max_age_days=3,
            cleanup_interval_hours=12
        ),
        'tui': LogRetentionConfig(
            max_bytes=2 * 1024 * 1024,   # 2MB
            backup_count=2,
            max_age_days=1,
            cleanup_interval_hours=6
        )
    }

    def __init__(self):
        """Initialize the log retention manager."""
        self.policies: Dict[str, LogRetentionConfig] = self.DEFAULT_POLICIES.copy()
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_stop_event = threading.Event()
        self.managed_handlers: List[logging.Handler] = []

        # Ensure log directory exists in cache directory
        self.log_directory = Path(g.CLIAGENT_CACHE_DIR) / "logs"
        self.log_directory.mkdir(parents=True, exist_ok=True)

    def create_rotating_handler(
        self,
        log_file_path: str,
        policy_name: str = 'application',
        custom_policy: Optional[LogRetentionConfig] = None
    ) -> RotatingFileHandler:
        """
        Create a rotating file handler with specified retention policy.

        Args:
            log_file_path: Path to the log file
            policy_name: Name of the retention policy to use
            custom_policy: Custom retention policy (overrides policy_name)

        Returns:
            Configured RotatingFileHandler
        """
        policy = custom_policy or self.policies.get(policy_name, self.DEFAULT_POLICIES['application'])

        # Ensure parent directory exists
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            log_file_path,
            maxBytes=policy.max_bytes,
            backupCount=policy.backup_count,
            encoding='utf-8'
        )

        self.managed_handlers.append(handler)
        return handler

    def create_timed_rotating_handler(
        self,
        log_file_path: str,
        when: str = 'midnight',
        interval: int = 1,
        backup_count: int = 7
    ) -> TimedRotatingFileHandler:
        """
        Create a time-based rotating file handler.

        Args:
            log_file_path: Path to the log file
            when: When to rotate ('midnight', 'H', 'D', etc.)
            interval: Interval between rotations
            backup_count: Number of backup files to keep

        Returns:
            Configured TimedRotatingFileHandler
        """
        # Ensure parent directory exists
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

        handler = TimedRotatingFileHandler(
            log_file_path,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding='utf-8'
        )

        self.managed_handlers.append(handler)
        return handler

    def cleanup_old_logs(self, policy_name: str = None) -> Dict[str, Any]:
        """
        Clean up old log files based on retention policies.

        Args:
            policy_name: Specific policy to apply, or None for all policies

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'files_removed': 0,
            'bytes_freed': 0,
            'errors': [],
            'policies_applied': []
        }

        policies_to_apply = {}
        if policy_name and policy_name in self.policies:
            policies_to_apply[policy_name] = self.policies[policy_name]
        else:
            policies_to_apply = self.policies

        for name, policy in policies_to_apply.items():
            try:
                policy_stats = self._cleanup_by_policy(name, policy)
                stats['files_removed'] += policy_stats['files_removed']
                stats['bytes_freed'] += policy_stats['bytes_freed']
                stats['policies_applied'].append(name)
                stats['errors'].extend(policy_stats['errors'])
            except Exception as e:
                stats['errors'].append(f"Policy {name}: {str(e)}")

        return stats

    def _cleanup_by_policy(self, policy_name: str, policy: LogRetentionConfig) -> Dict[str, Any]:
        """Apply cleanup policy to matching log files."""
        stats = {'files_removed': 0, 'bytes_freed': 0, 'errors': []}

        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(days=policy.max_age_days)
        cutoff_timestamp = cutoff_time.timestamp()

        # Find log files based on naming patterns
        log_patterns = [
            f"*{policy_name}*.log*",
            "*.log*",  # Generic log files
            "*debug*.log*" if policy_name == 'debug' else None,
            "*tui*.log*" if policy_name == 'tui' else None
        ]
        log_patterns = [p for p in log_patterns if p]

        # Search in multiple locations
        search_locations = [
            self.log_directory,
            Path(g.CLIAGENT_CACHE_DIR),
            Path.cwd(),  # Project root
            Path("/tmp") if os.path.exists("/tmp") else None
        ]
        search_locations = [loc for loc in search_locations if loc and loc.exists()]

        for location in search_locations:
            for pattern in log_patterns:
                try:
                    for log_file in location.glob(pattern):
                        if self._should_remove_file(log_file, cutoff_timestamp):
                            file_size = log_file.stat().st_size
                            log_file.unlink()
                            stats['files_removed'] += 1
                            stats['bytes_freed'] += file_size
                except Exception as e:
                    stats['errors'].append(f"Pattern {pattern} in {location}: {str(e)}")

        return stats

    def _should_remove_file(self, file_path: Path, cutoff_timestamp: float) -> bool:
        """Determine if a file should be removed based on age and naming."""
        try:
            # Check file age
            if file_path.stat().st_mtime > cutoff_timestamp:
                return False

            # Don't remove currently active log files (without rotation suffix)
            if file_path.suffix == '.log' and not any(c.isdigit() for c in file_path.stem):
                return False

            # Don't remove files that are too large (might be actively written to)
            if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                return False

            return True
        except (OSError, IOError):
            return False

    def cleanup_temp_audio_files(self) -> Dict[str, Any]:
        """Clean up temporary audio files created during debugging."""
        stats = {'files_removed': 0, 'bytes_freed': 0, 'errors': []}

        # Audio debug file patterns
        audio_patterns = [
            "faster_whisper_debug_*.wav",
            "faster_whisper_input_*.wav",
            "boosted_*.wav",
            "sr_test.wav",
            "recorded_speech.wav",
            "debug_audio_*.wav"
        ]

        # Search locations
        search_locations = [
            Path("/tmp"),
            Path(g.CLIAGENT_CACHE_DIR) / "temp",
            Path.cwd()
        ]

        # Files older than 1 day
        cutoff_time = datetime.now() - timedelta(days=1)
        cutoff_timestamp = cutoff_time.timestamp()

        for location in search_locations:
            if not location.exists():
                continue

            for pattern in audio_patterns:
                try:
                    for audio_file in location.glob(pattern):
                        if audio_file.stat().st_mtime < cutoff_timestamp:
                            file_size = audio_file.stat().st_size
                            audio_file.unlink()
                            stats['files_removed'] += 1
                            stats['bytes_freed'] += file_size
                except Exception as e:
                    stats['errors'].append(f"Audio cleanup {pattern} in {location}: {str(e)}")

        return stats

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about log files."""
        stats = {
            'total_files': 0,
            'total_bytes': 0,
            'by_policy': {},
            'locations': {},
            'oldest_file': None,
            'newest_file': None,
            'largest_file': None
        }

        oldest_time = float('inf')
        newest_time = 0
        largest_size = 0

        # Search locations
        search_locations = [
            self.log_directory,
            Path(g.CLIAGENT_CACHE_DIR),
            Path.cwd()
        ]

        for location in search_locations:
            if not location.exists():
                continue

            location_stats = {'files': 0, 'bytes': 0}

            for log_file in location.glob("*.log*"):
                try:
                    file_stat = log_file.stat()
                    file_size = file_stat.st_size
                    file_mtime = file_stat.st_mtime

                    stats['total_files'] += 1
                    stats['total_bytes'] += file_size
                    location_stats['files'] += 1
                    location_stats['bytes'] += file_size

                    # Track extremes
                    if file_mtime < oldest_time:
                        oldest_time = file_mtime
                        stats['oldest_file'] = str(log_file)

                    if file_mtime > newest_time:
                        newest_time = file_mtime
                        stats['newest_file'] = str(log_file)

                    if file_size > largest_size:
                        largest_size = file_size
                        stats['largest_file'] = str(log_file)

                except (OSError, IOError):
                    continue

            if location_stats['files'] > 0:
                stats['locations'][str(location)] = location_stats

        # Convert timestamps to readable dates
        if stats['oldest_file']:
            stats['oldest_date'] = datetime.fromtimestamp(oldest_time).isoformat()
        if stats['newest_file']:
            stats['newest_date'] = datetime.fromtimestamp(newest_time).isoformat()

        return stats

    def start_periodic_cleanup(self) -> None:
        """Start background thread for periodic log cleanup."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        self.cleanup_stop_event.clear()
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def stop_periodic_cleanup(self) -> None:
        """Stop background cleanup thread."""
        if self.cleanup_thread:
            self.cleanup_stop_event.set()
            self.cleanup_thread.join(timeout=5.0)

    def _periodic_cleanup_worker(self) -> None:
        """Background worker for periodic cleanup."""
        while not self.cleanup_stop_event.is_set():
            try:
                # Determine shortest cleanup interval
                min_interval_hours = min(policy.cleanup_interval_hours for policy in self.policies.values())
                wait_seconds = min_interval_hours * 3600

                # Wait for stop event or timeout
                if self.cleanup_stop_event.wait(wait_seconds):
                    break

                # Perform cleanup
                self.cleanup_old_logs()
                self.cleanup_temp_audio_files()

            except Exception as e:
                # Log error but continue running
                logging.debug(f"Periodic cleanup error: {e}")
                # Wait 1 hour before retrying on error
                if self.cleanup_stop_event.wait(3600):
                    break

    def register_handler(self, handler: logging.Handler) -> None:
        """Register a handler for management by this system."""
        if handler not in self.managed_handlers:
            self.managed_handlers.append(handler)

    def unregister_handler(self, handler: logging.Handler) -> None:
        """Unregister a handler from management."""
        if handler in self.managed_handlers:
            self.managed_handlers.remove(handler)

    def shutdown(self) -> Dict[str, Any]:
        """Gracefully shutdown the log manager and perform final cleanup."""
        # Stop periodic cleanup
        self.stop_periodic_cleanup()

        # Perform final cleanup
        cleanup_stats = self.cleanup_old_logs()
        audio_cleanup_stats = self.cleanup_temp_audio_files()

        # Close managed handlers
        for handler in self.managed_handlers:
            try:
                handler.close()
            except:
                pass
        self.managed_handlers.clear()

        return {
            'log_cleanup': cleanup_stats,
            'audio_cleanup': audio_cleanup_stats,
            'handlers_closed': len(self.managed_handlers)
        }


# Global instance
_log_manager = None


def get_log_manager() -> LogRetentionManager:
    """Get the global log retention manager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogRetentionManager()
    return _log_manager


# setup_managed_logging function removed - was unused


# Example usage and testing
if __name__ == "__main__":
    # Test the log retention system
    manager = LogRetentionManager()

    print("🧪 Testing Log Retention Manager...")
    print("=" * 50)

    # Get statistics
    stats = manager.get_log_statistics()
    print("📊 Current log statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total size: {stats['total_bytes'] / 1024:.1f} KB")
    print(f"  Locations: {len(stats['locations'])}")

    # Test cleanup
    print("\n🧹 Testing cleanup...")
    cleanup_stats = manager.cleanup_old_logs()
    print(f"  Files removed: {cleanup_stats['files_removed']}")
    print(f"  Space freed: {cleanup_stats['bytes_freed'] / 1024:.1f} KB")
    print(f"  Policies applied: {cleanup_stats['policies_applied']}")

    # Test audio cleanup
    audio_stats = manager.cleanup_temp_audio_files()
    print(f"  Audio files removed: {audio_stats['files_removed']}")
    print(f"  Audio space freed: {audio_stats['bytes_freed'] / 1024:.1f} KB")

    if cleanup_stats['errors'] or audio_stats['errors']:
        print("\n⚠️ Errors encountered:")
        for error in cleanup_stats['errors'] + audio_stats['errors']:
            print(f"  - {error}")

    print("\n✅ Log retention manager test completed!")