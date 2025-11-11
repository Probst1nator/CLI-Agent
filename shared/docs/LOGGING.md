# Logging System Documentation

## Overview

CLI-Agent features a comprehensive logging system with **always-on file logging**, automatic retention management, debug tracing, and tool-specific logging.

**Important: File logging is ALWAYS enabled at DEBUG level**, even without the `--debug` flag. This ensures that production issues can always be diagnosed by examining the log files, even if they're hard to reproduce.

## Quick Start

### Default Logging (Always On)

```bash
# Run normally - logs automatically go to ~/.cli-agent/logs/session.log at DEBUG level
python3 main.py

# Enable DEBUG output in console (file still logs everything)
python3 main.py --debug

# TUI mode with specialized formatting (file still logs everything)
python3 main.py --tui
```

**All runs create detailed DEBUG logs in `~/.cli-agent/logs/session.log`** regardless of console output level.

### Debug TTS Issues

When debugging Text-to-Speech (TTS) issues:

```bash
# Run with or without --debug, logs are always captured
python3 main.py -s

# View logs in real-time
tail -f ~/.cli-agent/logs/session.log | grep "TTS:"
```

The log file will contain detailed TTS flow information:
- Voice mode status
- Response content extraction
- Visualresponse detection and parsing
- LaTeX/markdown filtering steps
- Final TTS decision (speak/skip)

## Log Locations

### Default Locations

- **Session logs (ALWAYS ON)**: `~/.cli-agent/logs/session.log` (DEBUG level, 30 day retention)
- **Tool logs**: `~/.cli-agent/logs/tools_<toolname>.log`
- **Audit logs**: `~/.cli-agent/audit_logs/sessions/`
- **All logs**: Managed with automatic retention and rotation

### Temporary Logs

- **Fallback tool logs**: `/tmp/cli-agent-tool-<toolname>.log`
- **Debug audio**: `/tmp/*_whisper_input_*.wav`, `/tmp/*_screenshot_*.png`

## Log Retention Policies

The log manager automatically manages log files with the following retention policies:

| Policy Name | Retention Period | Max File Size | Use Case |
|-------------|------------------|---------------|----------|
| **debug** | 7 days | 10 MB | Debug sessions |
| **application** | 30 days | 10 MB | General application logs |
| **tui** | 14 days | 10 MB | TUI mode logs |
| **session** | 90 days | 50 MB | Session audit logs |
| **tool** | 14 days | 20 MB | Tool-specific logs |

### Automatic Cleanup

- Runs on startup and periodically every 24 hours
- Removes files older than retention period
- Rotates files exceeding max size
- Preserves currently active log files

## Log Levels

### Console Output

| Mode | Level | What You See |
|------|-------|--------------|
| **Normal** | INFO | Important events, warnings, errors |
| **Debug (`--debug`)** | DEBUG | All events including detailed traces |
| **TUI** | INFO | Simplified output without timestamps |

### File Output (ALWAYS ON)

All file handlers **always log at DEBUG level**, regardless of console settings. This means:

- ✅ **Production runs capture everything** - No need for `--debug` flag
- ✅ **Hard-to-reproduce issues are logged** - Just check `~/.cli-agent/logs/session.log`
- ✅ **TTS flow is always traced** - Every voice interaction is logged
- ✅ **Tool execution is recorded** - All tool calls and responses saved

## Log Formats

### Console Format

```
[MESSAGE]
```

- Colored output with ANSI codes
- Clean, user-friendly formatting
- No timestamps (uses system time)

### File Format

```
0:12.345 [DEBUG   ] Message content here
```

- `0:12.345`: Elapsed time since startup (minutes:seconds.milliseconds)
- `[DEBUG]`: Log level (padded to 8 chars)
- Message content with ANSI codes stripped

## TTS Debugging Guide

When TTS fails to speak content, search logs for these markers:

### Entry Point

```
TTS: Starting TTS processing (voice_mode=True)
TTS: Response length: 1234 chars
```

### Visualresponse Detection

```
TTS: Found 1 visualresponse blocks
TTS: Visualresponse type: markdown
TTS: Extracted markdown content (567 chars)
```

Or if not found:
```
TTS: No visualresponse blocks found
```

### Content Extraction

```
TTS: Extracted from <response> tag (234 chars)
```

Or:
```
TTS: Extracted by stripping XML tags (345 chars)
```

### Filtering

```
TTS: After LaTeX filtering: 300 chars (was 350)
```

### Final Decision

Success:
```
TTS: Speaking 300 chars: 'Ich werde Sie nun zu den Themen...'
```

Skipped:
```
TTS: Not speaking (skip_tts=True, response_text=0 chars)
```

Error:
```
Failed to speak response via TTS: [error message]
TTS error traceback: [full traceback]
```

## Common TTS Issues & Solutions

### Issue: TTS Not Speaking

**Check logs for:**
1. `voice_mode=False` → Enable with `-s` or `-v` flag
2. `No visualresponse blocks found` → Check XML structure
3. `No content blocks found in visualresponse` → Verify `<content>` tag exists
4. `skip_tts=True` → Visual-only content (html/code/js/css)
5. `response_text=0 chars` → Content extraction failed

### Issue: TTS Speaking Wrong Content

**Check logs for:**
1. Which extraction method was used (`<response>` tag vs XML stripping)
2. Content length before/after LaTeX filtering
3. First 100 chars of what was actually spoken

### Issue: TTS Crashes

**Check logs for:**
```
Failed to speak response via TTS: [exception]
TTS error traceback: [full stack trace]
```

## Tool-Specific Logging

Individual tools can have dedicated log files:

### Enable Tool Logging

```python
from shared.utils.logging_utils import setup_tool_logger

logger = setup_tool_logger(
    tool_name="mytool",
    enable_file_logging=True,
    log_level=logging.DEBUG
)
```

### Tool Log Location

- **Managed**: `~/.cli-agent/logs/tools_mytool.log` (with retention)
- **Fallback**: `/tmp/cli-agent-tool-mytool.log`

## Log Manager API

### Programmatic Access

```python
from shared.utils.log_manager import get_log_manager

log_manager = get_log_manager()

# Create rotating file handler
handler = log_manager.create_rotating_handler(
    log_file_path="/path/to/log.log",
    policy_name="application"
)

# Cleanup old logs
stats = log_manager.cleanup_old_logs()
print(f"Removed {stats['files_removed']} files")

# Get statistics
stats = log_manager.get_log_statistics()
print(f"Total log space: {stats['total_size'] / 1024 / 1024:.1f} MB")
```

### Define Custom Retention Policy

```python
log_manager.define_policy(
    name="custom",
    retention_days=45,
    max_file_size_mb=15,
    backup_count=5
)
```

## Best Practices

### For Users

1. **Logs are always available** at `~/.cli-agent/logs/session.log`
2. **When reporting issues**, include relevant log sections (search for error keywords)
3. **Note the approximate time** when the issue occurred
4. **Logs auto-clean after 30 days** - no manual maintenance needed
5. **Use `--debug` flag** only if you want verbose console output

### For Developers

1. **Use `logging.debug()` liberally** for trace information
2. **Use `logging.info()` for important state changes**
3. **Use `logging.warning()` for recoverable issues**
4. **Use `logging.error()` for failures requiring attention**
5. **Include context** in log messages (IDs, counts, states)

### Log Message Guidelines

**Good:**
```python
logging.debug(f"TTS: Extracted markdown content ({len(content)} chars)")
```

**Bad:**
```python
logging.debug("Got content")
```

**Good:**
```python
logging.error(f"Failed to parse XML block #{i}: {e}")
```

**Bad:**
```python
logging.error("Error")
```

## Troubleshooting

### Logs Not Being Created

1. Check file permissions on log directory
2. Verify `--log-file` path is writable
3. Check if log manager initialized (look for "Logging to file" message)

### Log Files Too Large

1. Check current policy: `grep "policy:" /tmp/cliagent.log`
2. Manually trigger cleanup: `log_manager.cleanup_old_logs()`
3. Reduce retention period or max file size

### Missing Debug Information

1. Ensure `--debug` flag is used
2. Check file handler is at DEBUG level
3. Verify logger hasn't been reconfigured

### Old Logs Not Being Cleaned

1. Check if periodic cleanup is running
2. Manually trigger: `log_manager.cleanup_old_logs(force=True)`
3. Check retention policy matches log file pattern

## Related Documentation

- **Audio System**: See `shared/utils_audio.py` for TTS implementation
- **Tool System**: See `docs/TOOLS.md` for tool development
- **Error Handling**: See `docs/ERROR_HANDLING.md` for exception patterns
- **File Management**: See `core/file_manager.py` for temp file cleanup

## Examples

### Debug a TTS Issue

```bash
# Run normally (logs are automatic at DEBUG level)
python3 main.py -s

# In another terminal, tail the default log
tail -f ~/.cli-agent/logs/session.log | grep "TTS:"

# After reproducing the issue, search for errors
grep -A 10 "Failed to speak" ~/.cli-agent/logs/session.log

# Or view the last 100 TTS-related log lines
grep "TTS:" ~/.cli-agent/logs/session.log | tail -100
```

### Analyze Tool Performance

```bash
# Run normally (logs automatically include all tool calls)
python3 main.py

# Check main session log for all tool activity
tail -f ~/.cli-agent/logs/session.log

# Or check tool-specific logs
tail -f ~/.cli-agent/logs/tools_readfile.log
tail -f ~/.cli-agent/logs/tools_bash.log
```

### Monitor Log Space Usage

```python
from shared.utils.log_manager import get_log_manager

log_manager = get_log_manager()
stats = log_manager.get_log_statistics()

print(f"Total files: {stats['total_files']}")
print(f"Total size: {stats['total_size'] / 1024 / 1024:.1f} MB")
print(f"Oldest: {stats['oldest_file']}")
print(f"Largest: {stats['largest_file']} ({stats['largest_size'] / 1024:.1f} KB)")
```

## Version History

- **v1.3** (2025-01-04): Fixed TTS not speaking visualresponse content (moved original_assistant_response assignment)
- **v1.2** (2025-01-04): Added comprehensive TTS debug logging and always-on file logging
- **v1.1** (2024-12): Added log manager with retention policies
- **v1.0** (2024-11): Initial logging system implementation
