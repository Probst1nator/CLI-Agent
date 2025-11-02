# Release Workflow: Public Repository Sync

This document describes the three-stage workflow for syncing changes from the private development repository to the public repository.

## Workflow Overview

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Private Development (git_private/main)              │
│ • Full development environment                               │
│ • All features, experimental tools, internal work            │
│ • No restrictions on commits                                 │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    │ GitHub PR: main → public-sync
                    │ Review Question: "Is this appropriate for public?"
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Public Sync Preparation (git_private/public-sync)   │
│ • Staging area for public releases                           │
│ • Same content as private, but reviewed for public readiness │
│ • No filtering yet - just approval gate                      │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    │ Sync Script: python user_scripts/sync_public_repo.py
                    │ Automated filtering: removes privileged tools, secrets
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Public Release (public-release → git_public/main)   │
│ • Filtered version: safe tools only                          │
│ • README updated, experimental tools removed                 │
│ • Ready for public consumption                               │
└──────────────────────────────────────────────────────────────┘
```

## Complete Workflow Steps

### Step 1: Develop on Main

Work normally on the `main` branch in `git_private`:

```bash
git checkout main
# ... make changes ...
git add .
git commit -m "Add new feature"
git push git_private main
```

### Step 2: Create PR for Public Review

When you have changes ready for public release, create a Pull Request:

**Option A: Using GitHub CLI**
```bash
gh pr create \
  --base public-sync \
  --head main \
  --title "Sync: [Brief description] for public release" \
  --body "$(cat <<EOF
## Changes
- Feature 1
- Feature 2

## Public Release Checklist
- [ ] No secrets or API keys
- [ ] No references to private infrastructure
- [ ] Documentation appropriate for public
- [ ] All experimental tools intentionally included
- [ ] Changes add value to public repository
EOF
)"
```

**Option B: On GitHub Website**
1. Go to https://github.com/Probst1nator/CLI-Agent_DEV
2. Click "Pull requests" → "New pull request"
3. Base: `public-sync` ← Compare: `main`
4. Fill in title and description using template
5. Create pull request

### Step 3: Review PR

Review the PR carefully:

**Review Checklist:**
- [ ] **No Secrets**: Search for API keys, passwords, tokens
- [ ] **No Private Infrastructure**: No mentions of saas_scripts, specific deployment details
- [ ] **Appropriate Documentation**: READMEs, comments suitable for public
- [ ] **Intentional Tool Inclusion**: Any experimental tools are deliberate
- [ ] **Value Addition**: Changes benefit public users

**How to Review:**
```bash
# Check diff for sensitive content
gh pr diff <PR-number> | grep -i "api_key\|password\|secret\|token"

# Review full changes
gh pr view <PR-number> --web
```

### Step 4: Merge PR

Once approved, merge the PR:

```bash
# Merge via CLI
gh pr merge <PR-number> --squash

# Or merge on GitHub website
```

### Step 5: Checkout public-sync and Run Sync

```bash
# Switch to public-sync branch
git checkout public-sync
git pull git_private public-sync

# Preview what will be filtered
python user_scripts/sync_public_repo.py --dry-run

# Create filtered public-release branch
python user_scripts/sync_public_repo.py --prepare
```

**What the sync script does:**
1. Creates `public-release` branch from `public-sync`
2. Deletes privileged tools (bash.py, writefile.py, etc.)
3. Removes internal directories (saas_scripts/, docker_files/traefik/)
4. Filters out .env, logs
5. Updates .gitignore for public

### Step 6: Verify Filtered Version

```bash
# Switch to public-release to verify
git checkout public-release

# Verify privileged tools are removed
ls tools/ | grep -E "(bash|writefile|editfile)"
# Should return nothing

# Verify internal dirs are removed
ls saas_scripts/ 2>&1
# Should show "No such file or directory"

# Test that agent still works (without privileged tools)
python main.py
# Should start without errors

# Review what was filtered
git diff public-sync..public-release --stat
```

### Step 7: Push to Public Repository

```bash
# From public-release branch
python user_scripts/sync_public_repo.py --push
```

This pushes `public-release` → `git_public/main`.

**Confirmation Prompt:**
```
⚠️  This will push to the public repository!
Remote: https://github.com/Probst1nator/CLI-Agent.git
Continue? (y/N):
```

Type `y` to proceed.

### Step 8: Verify on GitHub

Visit https://github.com/Probst1nator/CLI-Agent and verify:
- [ ] Latest commit is visible
- [ ] README is appropriate
- [ ] No `tools/bash.py` or other privileged tools
- [ ] No `saas_scripts/` directory
- [ ] All safe tools present (readfile.py, grepsearch.py, etc.)

## Quick Reference Commands

```bash
# Create PR for public sync
gh pr create --base public-sync --head main

# Merge approved PR
gh pr merge <number> --squash

# Run sync
git checkout public-sync
python user_scripts/sync_public_repo.py --dry-run
python user_scripts/sync_public_repo.py --prepare

# Verify
git checkout public-release
python main.py  # Should work
git diff public-sync..public-release --stat

# Push to public
python user_scripts/sync_public_repo.py --push
```

## Troubleshooting

### PR Shows Too Many Changes

**Problem**: PR includes commits you don't want public

**Solution**: Rebase or cherry-pick specific commits
```bash
git checkout public-sync
git cherry-pick <commit-hash>
git push git_private public-sync
```

### Sync Script Fails

**Problem**: `public-sync branch doesn't exist`

**Solution**: Create it
```bash
git checkout main
git checkout -b public-sync
git push git_private public-sync
```

### Public Version Has Import Errors

**Problem**: After filtering, `ModuleNotFoundError: No module named 'tools.bash'`

**Solution**: Bootstrap was already fixed to use readdirectory. If you see this, check:
```bash
grep "from tools.bash import" main.py
# Should find only safe imports
```

### Accidentally Pushed Secrets

**Problem**: API key was pushed to public

**Solution**:
1. Immediately rotate the API key
2. Remove from git history:
```bash
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/file' \
  --prune-empty --tag-name-filter cat -- --all
git push git_public main --force
```

## Best Practices

### Before Creating PR

1. **Search for secrets**:
   ```bash
   grep -r "API_KEY\|SECRET\|PASSWORD" . --exclude-dir=.git
   ```

2. **Check for private references**:
   ```bash
   grep -r "saas_scripts\|traefik" . --exclude-dir=.git
   ```

3. **Review commit messages**:
   ```bash
   git log main..public-sync --oneline
   ```

### Regular Sync Schedule

- **After major features**: Sync significant improvements
- **Monthly**: Keep public repo updated
- **Before conferences/presentations**: Ensure latest version available

### Communication

If the public repo is mentioned in papers/presentations:
- Note the specific commit or tag used
- Document any differences from private version

## Files Reference

- **Config**: `.public-repo-config.json` - Defines what gets filtered
- **Sync Script**: `user_scripts/sync_public_repo.py` - Handles filtering
- **This Guide**: `docs/RELEASE_WORKFLOW.md`
- **Quick Reference**: `docs/PUBLIC_SYNC.md`
- **PR Template**: `.github/pull_request_template.md`

## Questions?

- Check `docs/PUBLIC_SYNC.md` for quick reference
- Review `.public-repo-config.json` for filtering rules
- Open an issue on git_private for workflow improvements
