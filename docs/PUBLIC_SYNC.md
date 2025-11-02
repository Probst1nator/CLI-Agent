# Public Repository Sync

## Overview

This repository has a public version at https://github.com/Probst1nator/CLI-Agent that filters experimental tools and infrastructure for wider distribution.

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

### Experimental Tools → `tools/experimental/`
System-modifying tools moved to experimental directory with warning headers:
- `bash.py` - Shell command execution
- `writefile.py`, `editfile.py`, `insertline.py`, `deletelines.py` - File modifications
- `computercontrol.py` - System control

### Excluded Completely
- `saas_scripts/` - Multi-user infrastructure (experimental, untested)
- `docker_files/traefik/` - Load balancer config (multi-user only)
- `.env`, `*.log` - Secrets and logs

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
