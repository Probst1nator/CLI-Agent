# CLI-Agent User Scripts

This directory contains user-facing utility scripts for setting up, managing, and debugging CLI-Agent.

## Setup Scripts

### `setup_cli_agent.sh`
Sets up the `cli-agent` command for global system access.

**Usage:**
```bash
./setup_cli_agent.sh              # Install for current user
./setup_cli_agent.sh --user       # Install to ~/.local/bin
./setup_cli_agent.sh --system     # Install system-wide (requires sudo)
./setup_cli_agent.sh --check      # Check installation status
./setup_cli_agent.sh --uninstall  # Remove installation
```

**What it does:**
- Creates symlinks to make `cli-agent` available globally
- Adds CLI-Agent directory to PATH
- Sets up convenient `ca` alias
- Configures shell profile (.bashrc/.zshrc)

### `setup_bash_alias.sh`
Alternative script for setting up bash aliases for CLI-Agent.

**Usage:**
```bash
./setup_bash_alias.sh
```

### `setup_desktop_entry.sh`
Creates a desktop entry for launching CLI-Agent from your application menu.

**Usage:**
```bash
./setup_desktop_entry.sh
```

## Debugging Scripts

### `view_logs.sh`
Unified log viewer that shows logs from all sources in one place.

**Usage:**
```bash
./view_logs.sh           # Last 100 lines
./view_logs.sh 500       # Last 500 lines
./view_logs.sh | less    # Scrollable view
```

**Shows logs from:**
- Backend session logs (`~/.cli-agent/logs/session.log`)
- Frontend client errors (`~/.cli-agent/logs/client_errors.log`)
- Tool-specific logs (`~/.cli-agent/logs/tools_*.log`)
- Recent audit sessions
- Docker container logs (if running)

**Log files:**
- **Session logs**: 30 day retention, 50MB max
- **Tool logs**: 14 day retention, 20MB max
- **Audit logs**: 90 day retention

### `search_logs.py`
Advanced log search tool with powerful filtering capabilities.

**Usage:**
```bash
# Search by log level
./search_logs.py --level ERROR
./search_logs.py --level WARNING --level ERROR

# Search by time range
./search_logs.py --since 1h        # Last hour
./search_logs.py --since 30m       # Last 30 minutes
./search_logs.py --since 2d        # Last 2 days

# Search by request/session ID
./search_logs.py --request-id req_1730123456_abc123
./search_logs.py --session-id sess_xyz789

# Search by pattern
./search_logs.py --pattern "timeout"
./search_logs.py --pattern "error.*connection"

# Search specific sources
./search_logs.py --sources backend frontend
./search_logs.py --sources tool_bash

# Context lines
./search_logs.py --level ERROR -B 3 -A 3  # 3 lines before/after

# Export to JSON
./search_logs.py --level ERROR --format json > errors.json

# Combine filters
./search_logs.py --level ERROR --since 1h --sources backend --pattern "API"
```

**Features:**
- Time range filtering
- Log level filtering (ERROR, WARNING, INFO, DEBUG)
- Request ID / Session ID correlation (traces requests across frontend → backend → tools)
- Regex pattern matching
- Source filtering (backend, frontend, tools, containers)
- Context lines (show lines before/after matches)
- JSON export
- Colored terminal output

**Tip:** Request IDs enable full request tracing:
1. Frontend request → generates `req_TIMESTAMP_ID`
2. Backend receives request with same ID
3. Tools execute with same request ID in logs
4. Search by request ID to see entire flow

## Maintenance Scripts

### `manage_search_cache.py`
Manages the web search cache used by the `searchweb` tool.

**Usage:**
```bash
# View cache statistics
./manage_search_cache.py --stats

# Clear old entries
./manage_search_cache.py --clear-old

# Clear all cache
./manage_search_cache.py --clear-all

# Search cache contents
./manage_search_cache.py --search "python tutorial"
```

**Cache location:** `~/.cli-agent/web_search_cache.json`

## AI-Powered Scripts

### `regenerate_prompt_suggestions.sh`
Uses CLI-Agent to generate creative, contextual prompt suggestions based on your recent file activity.

**Usage:**
```bash
./regenerate_prompt_suggestions.sh           # Default: 10 recent files
./regenerate_prompt_suggestions.sh 20        # Use 20 recent files
```

**What it does:**
1. Finds recently modified files in `~/OneDrive/Studium` (configurable)
2. Uses CLI-Agent to read and analyze these files
3. Generates 5 creative, actionable prompt suggestions
4. Writes suggestions to `~/.cli-agent/suggested_prompts.json`
5. Web UI automatically picks up new suggestions

**Configuration:**
Edit the script to change:
- `STUDIUM_DIR`: Directory to scan for files
- `NUM_FILES`: Number of recent files to analyze

**Scheduled execution:**
Add to crontab for daily regeneration:
```bash
# Regenerate prompts at 8am daily
0 8 * * * /path/to/user_scripts/regenerate_prompt_suggestions.sh
```

**Log file:** `~/.cli-agent/logs/user_scripts/regenerate_prompt_suggestions.log`

### `run_screen_agent.sh`
Runs CLI-Agent with screenshot capabilities for visual tasks.

**Usage:**
```bash
./run_screen_agent.sh
```

## Script Standards

All scripts in this directory follow these conventions:

1. **Location-aware**: Can be run from any directory
2. **Self-documenting**: Header comments explain purpose and usage
3. **Error handling**: Use `set -e` for immediate exit on errors
4. **Colored output**: Consistent color scheme (GREEN, BLUE, YELLOW, RED)
5. **Logging**: Important scripts log to `~/.cli-agent/logs/user_scripts/`
6. **Executable**: All scripts are executable (`chmod +x`)

## Log Rotation

Scripts that generate logs use automatic rotation:
- Rotation threshold: 10MB
- Old logs saved as `.old` suffix
- Cleanup happens before each execution

## Tips

1. **Daily workflow**: Use `regenerate_prompt_suggestions.sh` to start your day with contextual prompts
2. **Debugging TTS**: Use `search_logs.py --pattern "TTS:" --level DEBUG` to trace voice flow
3. **Performance analysis**: Use `search_logs.py --pattern "elapsed" --since 1h` to find slow operations
4. **Error investigation**: Use `search_logs.py --level ERROR --request-id <id>` to trace full error context
5. **Setup verification**: Run `setup_cli_agent.sh --check` after installation

## Related Documentation

- [Logging System](../docs/LOGGING.md): Comprehensive logging architecture documentation
- [Debugging Enhancements](../docs/DEBUGGING_ENHANCEMENTS.md): Request tracing and error handling
- [Main README](../README.md): Installation and usage guide
- [CLAUDE.md](../CLAUDE.md): Development guide and architecture overview
