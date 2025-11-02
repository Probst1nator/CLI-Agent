# Public Repository Sync

## Overview

This repository has a public version at https://github.com/Probst1nator/CLI-Agent that contains **safe tools only** (read-only operations).

## Quick Usage

```bash
# Preview changes
python user_scripts/sync_public_repo.py --dry-run

# Prepare filtered branch
python user_scripts/sync_public_repo.py --prepare

# Review
git diff main..public-release

# Push to public
python user_scripts/sync_public_repo.py --push
```

## What Gets Filtered

### Privileged Tools (Deleted from Public)
These tools are **completely removed** from the public repository:
- `bash.py` - Shell command execution
- `writefile.py`, `editfile.py`, `insertline.py`, `deletelines.py` - File modifications
- `computercontrol.py` - System control

### Safe Tools (Kept in Public)
- `readfile.py`, `readdirectory.py` - Read files and directories
- `grepsearch.py`, `searchconversations.py` - Search operations
- `viewpermissions.py` - View permission settings
- All other read-only tools

### Also Excluded
- `.env`, `*.log` - Secrets and logs
- Various internal development directories

## Result

**Public version**: Clean, safe agent with read-only capabilities
**Private version**: Full agent with all tools including privileged operations

## Configuration

Edit `.public-repo-config.json` to customize what gets filtered.

## Branch Structure

```
main (private dev - all tools)
  ↓
public-release (safe tools only)
  ↓
git_public:main (public repository)
```
