# Public Repository Sync

Public version at https://github.com/Probst1nator/CLI-Agent (safe tools only).

## Workflow

```
main → PR → public-sync → filter → public-release → git_public/main
```

### Commands

```bash
# 1. Create PR for review
gh pr create --base public-sync --head main

# 2. After merge, sync from public-sync
git checkout public-sync && git pull
python user_scripts/sync_public_repo.py --prepare

# 3. Verify filtered version
git checkout public-release
python main.py  # Test it works
git diff public-sync..public-release --stat

# 4. Push to public
python user_scripts/sync_public_repo.py --push
```

### What Gets Filtered

**Removed**: `bash.py`, `writefile.py`, `editfile.py`, `computercontrol.py`, internal dirs, secrets
**Kept**: All read-only tools (`readfile.py`, `grepsearch.py`, etc.)

### Review Checklist (for Public Sync PRs)

Before merging main → public-sync:
```bash
# Check for secrets
grep -r "API_KEY\|SECRET\|PASSWORD" . --exclude-dir=.git

# Check for private refs
grep -r "saas_scripts\|traefik" . --exclude-dir=.git
```

Verify:
- [ ] No secrets or credentials
- [ ] No private infrastructure references
- [ ] Documentation appropriate for public
- [ ] Changes add value to public repo

### Config

Edit `.public-repo-config.json` to customize filtering.
