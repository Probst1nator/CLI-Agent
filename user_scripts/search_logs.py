#!/usr/bin/env python3
"""
Enhanced Log Search Tool for CLI-Agent

Unified search across all log sources with powerful filtering:
- Backend session logs
- Frontend client errors
- Tool-specific logs
- Container logs
- Deployment logs

Features:
- Time range filtering
- Log level filtering (ERROR, WARNING, INFO, DEBUG)
- Request ID / Session ID correlation
- Regex pattern matching
- Context lines (before/after matches)
- JSON export
- Colored terminal output
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path


from typing import Optional
# Color codes for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    GRAY = '\033[0;90m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


@dataclass
class LogEntry:
    """Represents a single log entry"""
    timestamp: str
    level: str
    source: str  # backend, frontend, tool_bash, container_traefik, etc.
    message: str
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    context_before: list[str] = None
    context_after: list[str] = None
    raw_line: str = ""

    def __post_init__(self):
        if self.context_before is None:
            self.context_before = []
        if self.context_after is None:
            self.context_after = []


class LogSearcher:
    """Search and filter logs across all CLI-Agent sources"""

    def __init__(self):
        self.home = Path.home()
        self.log_dir = self.home / ".cli-agent" / "logs"
        self.audit_dir = self.home / ".cli-agent" / "audit_logs" / "sessions"
        self.deployment_dir = self.home / ".cli-agent" / "deployment_logs"

    def get_log_files(self, sources: list[str]) -> dict[str, list[Path]]:
        """Get log files organized by source"""
        files = defaultdict(list)

        if "backend" in sources:
            session_log = self.log_dir / "session.log"
            if session_log.exists():
                files["backend"].append(session_log)

        if "frontend" in sources:
            client_log = self.log_dir / "client_errors.log"
            if client_log.exists():
                files["frontend"].append(client_log)

        if "tools" in sources:
            for tool_log in self.log_dir.glob("tools_*.log"):
                tool_name = tool_log.stem.replace("tools_", "")
                files[f"tool_{tool_name}"].append(tool_log)

        if "deployment" in sources:
            if self.deployment_dir.exists():
                for deploy_log in self.deployment_dir.glob("*.log"):
                    files[f"deploy_{deploy_log.stem}"].append(deploy_log)

        if "containers" in sources:
            # Get container logs via docker logs command
            try:
                result = subprocess.run(
                    ["docker", "ps", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                container_names = [name for name in result.stdout.strip().split('\n')
                                  if 'cli-agent' in name or 'traefik' in name]
                for container in container_names:
                    files[f"container_{container}"].append(Path(f"docker:{container}"))
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass  # Docker not available or no containers

        return files

    def parse_log_line(self, line: str, source: str) -> Optional[LogEntry]:
        """Parse a log line into a LogEntry object"""
        if not line.strip():
            return None

        # Try different log formats

        # Backend format: [  0.12s] [req_123] [INFO    ] Message
        backend_pattern = r'\[\s*[\d.]+s\]\s*(?:\[([^\]]+)\])?\s*\[([A-Z]+)\s*\]\s*(.+)'
        match = re.match(backend_pattern, line)
        if match:
            request_id = match.group(1) if match.group(1) else None
            level = match.group(2).strip()
            message = match.group(3).strip()
            return LogEntry(
                timestamp=datetime.now().isoformat(),  # Backend doesn't include absolute timestamp
                level=level,
                source=source,
                message=message,
                request_id=request_id,
                raw_line=line
            )

        # Frontend format: [2025-10-28T12:34:56] [session_id] [req_id] [ERROR] Message
        frontend_pattern = r'\[([^\]]+)\]\s*\[([^\]]+)\]\s*\[([^\]]+)\]\s*\[([A-Z]+)\]\s*(.+)'
        match = re.match(frontend_pattern, line)
        if match:
            timestamp = match.group(1)
            session_id = match.group(2)
            request_id = match.group(3)
            level = match.group(4)
            message = match.group(5).strip()
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=source,
                message=message,
                request_id=request_id,
                session_id=session_id,
                raw_line=line
            )

        # Deployment logger format: [12:34:56] [INFO] Message
        deploy_pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s*\[([^\]]+)\]:\s*(.+)'
        match = re.match(deploy_pattern, line)
        if match:
            time_str = match.group(1)
            level = match.group(2).strip()
            message = match.group(3).strip()
            # Convert time to full timestamp (use today's date)
            today = datetime.now().date()
            timestamp = f"{today}T{time_str}"
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=source,
                message=message,
                raw_line=line
            )

        # Container Manager format: 2025-10-28 12:34:56 [INFO] name: Message
        container_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[([A-Z]+)\]\s+([^:]+):\s*(.+)'
        match = re.match(container_pattern, line)
        if match:
            timestamp = match.group(1).replace(' ', 'T')
            level = match.group(2)
            logger_name = match.group(3)
            message = match.group(4).strip()
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=source,
                message=f"[{logger_name}] {message}",
                raw_line=line
            )

        # Fallback: treat as unstructured log
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level="INFO",
            source=source,
            message=line.strip(),
            raw_line=line
        )

    def read_file_logs(self, file_path: Path, source: str) -> list[LogEntry]:
        """Read and parse logs from a file"""
        entries = []
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    entry = self.parse_log_line(line, source)
                    if entry:
                        entries.append(entry)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
        return entries

    def read_container_logs(self, container_name: str, source: str, since: str = "24h") -> list[LogEntry]:
        """Read logs from a Docker container"""
        entries = []
        try:
            result = subprocess.run(
                ["docker", "logs", "--since", since, "--timestamps", container_name],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.split('\n'):
                entry = self.parse_log_line(line, source)
                if entry:
                    entries.append(entry)
        except Exception as e:
            print(f"Warning: Could not read container logs for {container_name}: {e}")
        return entries

    def search(
        self,
        sources: list[str] = None,
        pattern: str = None,
        level: str = None,
        request_id: str = None,
        session_id: str = None,
        since: str = None,
        until: str = None,
        context_lines: int = 0
    ) -> list[LogEntry]:
        """
        Search logs with filters

        Args:
            sources: List of sources to search (backend, frontend, tools, deployment, containers)
            pattern: Regex pattern to match in message
            level: Log level filter (ERROR, WARNING, INFO, DEBUG)
            request_id: Filter by request ID
            session_id: Filter by session ID
            since: Start time (ISO format or relative like "1h", "30m")
            until: End time (ISO format)
            context_lines: Number of context lines before/after matches

        Returns:
            List of matching LogEntry objects
        """
        if sources is None:
            sources = ["backend", "frontend", "tools", "deployment", "containers"]

        # Parse time filters
        since_dt = self._parse_time(since) if since else None
        until_dt = self._parse_time(until) if until else None

        # Compile pattern if provided
        regex = re.compile(pattern, re.IGNORECASE) if pattern else None

        # Get all log files
        log_files = self.get_log_files(sources)

        all_entries = []

        # Read all logs
        for source, files in log_files.items():
            for file_path in files:
                if str(file_path).startswith("docker:"):
                    # Container logs
                    container_name = str(file_path).replace("docker:", "")
                    since_str = since if since else "24h"
                    entries = self.read_container_logs(container_name, source, since_str)
                else:
                    # File logs
                    entries = self.read_file_logs(file_path, source)

                all_entries.extend(entries)

        # Apply filters
        filtered = []
        for entry in all_entries:
            # Level filter
            if level and entry.level != level:
                continue

            # Request ID filter
            if request_id and entry.request_id != request_id:
                continue

            # Session ID filter
            if session_id and entry.session_id != session_id:
                continue

            # Time range filter
            try:
                entry_time = datetime.fromisoformat(entry.timestamp)
                if since_dt and entry_time < since_dt:
                    continue
                if until_dt and entry_time > until_dt:
                    continue
            except ValueError:
                pass  # Skip time filtering if timestamp invalid

            # Pattern filter
            if regex and not regex.search(entry.message):
                continue

            filtered.append(entry)

        # Sort by timestamp
        filtered.sort(key=lambda e: e.timestamp)

        return filtered

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string (ISO format or relative like '1h', '30m')"""
        # Try relative time
        if time_str.endswith('h'):
            hours = int(time_str[:-1])
            return datetime.now() - timedelta(hours=hours)
        elif time_str.endswith('m'):
            minutes = int(time_str[:-1])
            return datetime.now() - timedelta(minutes=minutes)
        elif time_str.endswith('d'):
            days = int(time_str[:-1])
            return datetime.now() - timedelta(days=days)
        else:
            # Try ISO format
            return datetime.fromisoformat(time_str)

    def format_output(self, entries: list[LogEntry], color: bool = True, format_type: str = "text") -> str:
        """Format search results for display"""
        if format_type == "json":
            return json.dumps([asdict(e) for e in entries], indent=2)

        lines = []

        for entry in entries:
            # Color based on level
            if color:
                if entry.level == "ERROR":
                    level_color = Colors.RED
                elif entry.level == "WARNING" or entry.level == "WARN":
                    level_color = Colors.YELLOW
                elif entry.level == "DEBUG":
                    level_color = Colors.GRAY
                else:
                    level_color = Colors.BLUE

                source_color = Colors.CYAN
                timestamp_color = Colors.GRAY
                nc = Colors.NC
            else:
                level_color = source_color = timestamp_color = nc = ""

            # Format line
            line_parts = [
                f"{timestamp_color}{entry.timestamp}{nc}",
                f"{source_color}[{entry.source}]{nc}",
                f"{level_color}[{entry.level}]{nc}"
            ]

            if entry.request_id:
                line_parts.append(f"{Colors.MAGENTA if color else ''}[{entry.request_id}]{nc}")
            if entry.session_id:
                line_parts.append(f"{Colors.MAGENTA if color else ''}[{entry.session_id}]{nc}")

            line_parts.append(entry.message)

            lines.append(" ".join(line_parts))

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced log search tool for CLI-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find all errors in the last hour
  search_logs.py --level ERROR --since 1h

  # Trace a specific request across all logs
  search_logs.py --request-id req_1730123456_abc123

  # Search for pattern in backend logs
  search_logs.py --sources backend --pattern "Failed to.*"

  # Find container errors
  search_logs.py --sources containers --level ERROR

  # Export results to JSON
  search_logs.py --level ERROR --format json > errors.json
        """
    )

    parser.add_argument("--sources", nargs="+",
                       choices=["backend", "frontend", "tools", "deployment", "containers"],
                       help="Log sources to search (default: all)")
    parser.add_argument("--pattern", help="Regex pattern to search for")
    parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "WARN", "ERROR"],
                       help="Filter by log level")
    parser.add_argument("--request-id", help="Filter by request ID")
    parser.add_argument("--session-id", help="Filter by session ID")
    parser.add_argument("--since", help="Start time (ISO format or relative: 1h, 30m, 2d)")
    parser.add_argument("--until", help="End time (ISO format)")
    parser.add_argument("--context", type=int, default=0,
                       help="Number of context lines before/after matches")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="Output format")
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output")

    args = parser.parse_args()

    searcher = LogSearcher()
    results = searcher.search(
        sources=args.sources,
        pattern=args.pattern,
        level=args.level,
        request_id=args.request_id,
        session_id=args.session_id,
        since=args.since,
        until=args.until,
        context_lines=args.context
    )

    output = searcher.format_output(
        results,
        color=not args.no_color,
        format_type=args.format
    )

    print(output)

    # Print summary
    if args.format == "text":
        print(f"\n{Colors.BOLD if not args.no_color else ''}Found {len(results)} matching log entries{Colors.NC if not args.no_color else ''}")


if __name__ == "__main__":
    main()
