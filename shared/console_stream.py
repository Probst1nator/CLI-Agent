"""
Unified Console Streaming Infrastructure

Captures all terminal output (stdout/stderr) and streams it to:
1. The actual terminal (preserving colors and formatting)
2. A log file for persistence
3. WebSocket clients for real-time browser display
"""

import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class TeeStream(io.TextIOBase):
    """
    Stream that writes to multiple destinations simultaneously.

    Captures output to both the original stream (terminal) and a log file,
    while optionally calling a broadcast callback for real-time streaming.
    """

    def __init__(
        self,
        original_stream,
        log_file_path: Path,
        broadcast_callback: Optional[Callable[[str], None]] = None,
        strip_ansi: bool = False
    ):
        self.original_stream = original_stream
        self.log_file_path = log_file_path
        self.broadcast_callback = broadcast_callback
        self.strip_ansi = strip_ansi

        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_file_path, 'a', encoding='utf-8', buffering=1)

        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if not text:
            return 0

        with self._lock:
            self.original_stream.write(text)
            self.original_stream.flush()

            clean_text = self._clean_text(text) if self.strip_ansi else text
            self.log_file.write(clean_text)
            self.log_file.flush()

            if self.broadcast_callback and clean_text.strip():
                try:
                    self.broadcast_callback(clean_text)
                except Exception:
                    pass

        return len(text)

    def _clean_text(self, text: str) -> str:
        """Remove ANSI escape codes from text"""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def flush(self):
        with self._lock:
            if hasattr(self.original_stream, 'flush'):
                self.original_stream.flush()
            if self.log_file:
                self.log_file.flush()

    def close(self):
        with self._lock:
            if self.log_file and not self.log_file.closed:
                self.log_file.close()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


class LogTailer:
    """
    Monitors a log file for new lines and broadcasts them via callback.

    Uses efficient file position tracking to avoid re-reading the entire file.
    Runs in a background thread with configurable polling interval.
    """

    def __init__(
        self,
        log_file_path: Path,
        broadcast_callback: Callable[[str], None],
        poll_interval: float = 0.1,
        max_backlog_lines: int = 100
    ):
        self.log_file_path = log_file_path
        self.broadcast_callback = broadcast_callback
        self.poll_interval = poll_interval
        self.max_backlog_lines = max_backlog_lines

        self._file_position = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start the log tailer thread"""
        if self._running:
            return

        self._running = True

        if self.log_file_path.exists():
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                f.seek(0, os.SEEK_END)
                self._file_position = f.tell()

                f.seek(max(0, self._file_position - 100000), os.SEEK_SET)
                lines = f.readlines()
                backlog = lines[-self.max_backlog_lines:] if lines else []

                for line in backlog:
                    try:
                        self.broadcast_callback(line.rstrip('\n\r'))
                    except Exception as e:
                        logger.debug(f"Error broadcasting backlog line: {e}")

        self._thread = threading.Thread(target=self._tail_loop, daemon=True, name="LogTailer")
        self._thread.start()
        logger.info(f"LogTailer started for {self.log_file_path}")

    def stop(self):
        """Stop the log tailer thread"""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("LogTailer stopped")

    def _tail_loop(self):
        """Main loop that checks for new lines"""
        while self._running:
            try:
                if not self.log_file_path.exists():
                    time.sleep(self.poll_interval)
                    continue

                with self._lock:
                    with open(self.log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                        f.seek(self._file_position)
                        new_lines = f.readlines()
                        self._file_position = f.tell()

                for line in new_lines:
                    clean_line = line.rstrip('\n\r')
                    if clean_line:
                        try:
                            self.broadcast_callback(clean_line)
                        except Exception as e:
                            logger.debug(f"Error broadcasting line: {e}")

                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in log tailer loop: {e}")
                time.sleep(1.0)

    def reset_position(self):
        """Reset file position to end of file"""
        with self._lock:
            if self.log_file_path.exists():
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    f.seek(0, os.SEEK_END)
                    self._file_position = f.tell()


class ConsoleStreamManager:
    """
    Manages console output capture and streaming.

    Coordinates TeeStream and LogTailer to provide unified console streaming.
    """

    def __init__(
        self,
        log_file_path: Path,
        broadcast_callback: Optional[Callable[[str], None]] = None,
        strip_ansi: bool = False
    ):
        self.log_file_path = log_file_path
        self.broadcast_callback = broadcast_callback
        self.strip_ansi = strip_ansi

        self._original_stdout = None
        self._original_stderr = None
        self._tee_stdout: Optional[TeeStream] = None
        self._tee_stderr: Optional[TeeStream] = None
        self._tailer: Optional[LogTailer] = None
        self._active = False

    def start(self):
        """Start console capture and streaming"""
        if self._active:
            return

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        self._tee_stdout = TeeStream(
            self._original_stdout,
            self.log_file_path,
            broadcast_callback=self.broadcast_callback,
            strip_ansi=self.strip_ansi
        )

        self._tee_stderr = TeeStream(
            self._original_stderr,
            self.log_file_path,
            broadcast_callback=self.broadcast_callback,
            strip_ansi=self.strip_ansi
        )

        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr

        if self.broadcast_callback:
            self._tailer = LogTailer(
                self.log_file_path,
                self.broadcast_callback,
                poll_interval=0.1,
                max_backlog_lines=50
            )
            self._tailer.start()

        self._active = True
        logger.info(f"Console streaming started to {self.log_file_path}")

    def stop(self):
        """Stop console capture and streaming"""
        if not self._active:
            return

        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        if self._tee_stdout:
            self._tee_stdout.close()
        if self._tee_stderr:
            self._tee_stderr.close()

        if self._tailer:
            self._tailer.stop()

        self._active = False
        logger.info("Console streaming stopped")

    def is_active(self) -> bool:
        """Check if console streaming is active"""
        return self._active


_global_stream_manager: Optional[ConsoleStreamManager] = None


def get_console_stream_manager() -> Optional[ConsoleStreamManager]:
    """Get the global console stream manager instance"""
    global _global_stream_manager
    return _global_stream_manager


def init_console_streaming(
    log_file_path: Path,
    broadcast_callback: Optional[Callable[[str], None]] = None,
    strip_ansi: bool = True
):
    """Initialize global console streaming"""
    global _global_stream_manager

    if _global_stream_manager and _global_stream_manager.is_active():
        logger.warning("Console streaming already initialized")
        return _global_stream_manager

    _global_stream_manager = ConsoleStreamManager(
        log_file_path,
        broadcast_callback,
        strip_ansi
    )
    _global_stream_manager.start()

    return _global_stream_manager


def stop_console_streaming():
    """Stop global console streaming"""
    global _global_stream_manager

    if _global_stream_manager:
        _global_stream_manager.stop()
        _global_stream_manager = None
