# Public Repository Sync

## Overview

This repository has a public version at https://github.com/Probst1nator/CLI-Agent that filters privileged tools for wider distribution.

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

### Privileged Tools → `tools/privileged/`
System-level tools moved to privileged directory with warning headers:
- `bash.py` - Shell command execution
- `writefile.py`, `editfile.py`, `insertline.py`, `deletelines.py` - File modifications
- `computercontrol.py` - System control

### Excluded
- `.env`, `*.log` - Secrets and logs
- Various internal development directories

## Configuration

Edit `.public-repo-config.json` to customize what gets filtered.

## Branch Structure

```
main (private dev)
  ↓
public-release (filtered)
  ↓
git_public:main
```
