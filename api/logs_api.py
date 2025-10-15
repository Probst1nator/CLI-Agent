"""
API endpoints for accessing log files.

Provides endpoints to read and tail log files for debugging and monitoring.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
    AIOHTTP_AVAILABLE = False

try:
    from core.globals import g
except ImportError:
    class MockGlobals:
        CLIAGENT_CACHE_DIR = os.path.expanduser("~/.cache/cli-agent")
        CLIAGENT_PERSISTENT_STORAGE_PATH = os.path.expanduser("~/.cli-agent")
    g = MockGlobals()

logger = logging.getLogger(__name__)


class LogReader:
    """Utility for reading and parsing log files"""

    LOG_PATTERN = re.compile(
        r'^\[\s*(\d+\.\d+)s\]\s+\[([A-Z]+)\s*\]\s+(.*)$'
    )

    @staticmethod
    def get_log_directory() -> Path:
        """Get the log directory path"""
        legacy_dir = Path(g.CLIAGENT_PERSISTENT_STORAGE_PATH if hasattr(g, 'CLIAGENT_PERSISTENT_STORAGE_PATH') else Path.home() / ".cli-agent") / "logs"

        if legacy_dir.exists():
            log_files = list(legacy_dir.glob("*.log*"))
            if log_files:
                return legacy_dir

        cache_dir = Path(g.CLIAGENT_CACHE_DIR) if hasattr(g, 'CLIAGENT_CACHE_DIR') else Path.home() / ".cache" / "cli-agent"
        log_dir = cache_dir / "logs"

        return log_dir

    @staticmethod
    def list_available_logs() -> List[Dict[str, Any]]:
        """List all available log files with metadata"""
        log_dir = LogReader.get_log_directory()

        if not log_dir.exists():
            return []

        logs = []
        for log_file in sorted(log_dir.glob("*.log*"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                stat = log_file.stat()
                logs.append({
                    "name": log_file.name,
                    "path": str(log_file),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "type": LogReader._categorize_log(log_file.name)
                })
            except Exception as e:
                logger.warning(f"Error reading log file {log_file}: {e}")

        return logs

    @staticmethod
    def _categorize_log(filename: str) -> str:
        """Categorize log file by type"""
        if filename.startswith("session"):
            return "session"
        elif filename.startswith("tools_"):
            tool_name = filename.replace("tools_", "").replace(".log", "").split(".")[0]
            return f"tool:{tool_name}"
        elif "audit" in filename:
            return "audit"
        else:
            return "other"

    @staticmethod
    def tail_log(log_name: str, lines: int = 500, offset: int = 0) -> Dict[str, Any]:
        """
        Read the last N lines from a log file

        Args:
            log_name: Name of the log file (e.g., "session.log", "tools_bash.log")
            lines: Number of lines to read from the end
            offset: Number of lines to skip from the end (for pagination)

        Returns:
            Dictionary with log metadata and parsed entries
        """
        log_dir = LogReader.get_log_directory()
        log_path = log_dir / log_name

        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_name}")

        if not log_path.is_file():
            raise ValueError(f"Not a file: {log_name}")

        entries = []
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()

            start_idx = max(0, len(all_lines) - lines - offset)
            end_idx = len(all_lines) - offset if offset > 0 else len(all_lines)

            selected_lines = all_lines[start_idx:end_idx]

            for line_num, line in enumerate(selected_lines, start=start_idx + 1):
                line = line.rstrip('\n\r')
                if not line:
                    continue

                parsed = LogReader._parse_log_line(line)
                parsed['line_number'] = line_num
                entries.append(parsed)

            stat = log_path.stat()
            return {
                "success": True,
                "log_name": log_name,
                "total_lines": len(all_lines),
                "returned_lines": len(entries),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "entries": entries
            }

        except Exception as e:
            logger.error(f"Error reading log file {log_name}: {e}", exc_info=True)
            raise

    @staticmethod
    def _parse_log_line(line: str) -> Dict[str, str]:
        """Parse a log line into structured format"""
        match = LogReader.LOG_PATTERN.match(line)

        if match:
            timestamp = match.group(1)
            level = match.group(2).strip()
            message = match.group(3)

            return {
                "timestamp": timestamp,
                "level": level,
                "message": message,
                "raw": line
            }
        else:
            level = "INFO"
            if "ERROR" in line.upper():
                level = "ERROR"
            elif "WARN" in line.upper():
                level = "WARNING"
            elif "DEBUG" in line.upper():
                level = "DEBUG"

            return {
                "timestamp": "",
                "level": level,
                "message": line,
                "raw": line
            }


async def handle_list_logs(request: web.Request) -> web.Response:
    """GET /api/logs/list - List all available log files"""
    try:
        logs = LogReader.list_available_logs()
        return web.json_response({
            "status": "success",
            "data": {
                "logs": logs,
                "log_directory": str(LogReader.get_log_directory())
            }
        })
    except Exception as e:
        logger.error(f"Error listing logs: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "LIST_LOGS_ERROR"
        }, status=500)


async def handle_tail_log(request: web.Request) -> web.Response:
    """GET /api/logs/tail?log=session.log&lines=500&offset=0"""
    try:
        log_name = request.query.get('log', 'session.log')
        lines = int(request.query.get('lines', '500'))
        offset = int(request.query.get('offset', '0'))

        lines = min(max(1, lines), 10000)
        offset = max(0, offset)

        result = LogReader.tail_log(log_name, lines, offset)
        return web.json_response({
            "status": "success",
            "data": result
        })

    except FileNotFoundError as e:
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "LOG_NOT_FOUND"
        }, status=404)
    except ValueError as e:
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "INVALID_REQUEST"
        }, status=400)
    except Exception as e:
        logger.error(f"Error tailing log: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "TAIL_LOG_ERROR"
        }, status=500)


async def handle_tail_stream(request: web.Request) -> web.StreamResponse:
    """GET /api/logs/tail-stream?log=session.log - Server-Sent Events stream of log updates"""
    try:
        log_name = request.query.get('log', 'session.log')
        log_dir = LogReader.get_log_directory()
        log_path = log_dir / log_name

        if not log_path.exists():
            return web.json_response({
                "status": "error",
                "error": f"Log file not found: {log_name}",
                "code": "LOG_NOT_FOUND"
            }, status=404)

        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        await response.prepare(request)

        file_position = 0
        with open(log_path, 'r', encoding='utf-8') as f:
            f.seek(0, os.SEEK_END)
            file_position = f.tell()

            f.seek(max(0, file_position - 50000), os.SEEK_SET)
            backlog_lines = f.readlines()[-50:]

            for line in backlog_lines:
                entry = LogReader._parse_log_line(line.rstrip('\n\r'))
                data = json.dumps(entry)
                await response.write(f"data: {data}\n\n".encode('utf-8'))

        poll_interval = 0.5
        last_size = log_path.stat().st_size

        while not request.transport.is_closing():
            await asyncio.sleep(poll_interval)

            try:
                current_size = log_path.stat().st_size

                if current_size > last_size:
                    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                        f.seek(file_position)
                        new_lines = f.readlines()
                        file_position = f.tell()

                    for line in new_lines:
                        clean_line = line.rstrip('\n\r')
                        if clean_line:
                            entry = LogReader._parse_log_line(clean_line)
                            data = json.dumps(entry)
                            await response.write(f"data: {data}\n\n".encode('utf-8'))

                    last_size = current_size

                elif current_size < last_size:
                    file_position = 0
                    last_size = current_size

            except FileNotFoundError:
                break
            except Exception as e:
                logger.error(f"Error in SSE stream: {e}")
                break

        return response

    except Exception as e:
        logger.error(f"Error setting up SSE stream: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "STREAM_ERROR"
        }, status=500)


def setup_routes(app):
    """Register log API routes"""
    if not AIOHTTP_AVAILABLE:
        logger.warning("aiohttp not available, cannot setup log routes")
        return

    app.router.add_get('/api/logs/list', handle_list_logs)
    app.router.add_get('/api/logs/tail', handle_tail_log)
    app.router.add_get('/api/logs/tail-stream', handle_tail_stream)

    logger.info("Log API routes registered")
