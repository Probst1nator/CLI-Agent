# Public Repository Sync - Quick Reference

## Overview

This repository has a public version at https://github.com/Probst1nator/CLI-Agent that contains **safe tools only** (read-only operations).

## Three-Stage Workflow

```
main → PR → public-sync → sync script → public-release → git_public/main
```

1. **main** (private dev) - All features and tools
2. **public-sync** (review stage) - Approved for public
3. **public-release** (filtered) - Safe tools only

**Full guide**: See `docs/RELEASE_WORKFLOW.md`

## Quick Commands

```bash
# 1. Create PR for public review
gh pr create --base public-sync --head main

# 2. After PR merged, run sync from public-sync branch
git checkout public-sync
git pull git_private public-sync

# 3. Preview changes
python user_scripts/sync_public_repo.py --dry-run

# 4. Create filtered branch
python user_scripts/sync_public_repo.py --prepare

# 5. Review filtering
git checkout public-release
git diff public-sync..public-release --stat
python main.py  # Verify it works

# 6. Push to public
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
┌─────────────────────────────────────┐
│ git_private (Private Development)   │
├─────────────────────────────────────┤
│ main          - Active development  │
│ public-sync   - Public review stage │
│ public-release - Filtered (temporary)│
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│ git_public (Public Repository)      │
├─────────────────────────────────────┤
│ main          - Public releases     │
└─────────────────────────────────────┘
```

## Important Notes

- **Always work on `main`** for development
- **PR to `public-sync`** when ready for public release
- **Sync script runs from `public-sync`** (not main)
- **`public-release`** is temporary, deleted after push

## Full Documentation

- **Complete Workflow**: `docs/RELEASE_WORKFLOW.md`
- **PR Template**: `.github/pull_request_template.md`
- **Sync Script**: `user_scripts/sync_public_repo.py`
- **Config**: `.public-repo-config.json`
