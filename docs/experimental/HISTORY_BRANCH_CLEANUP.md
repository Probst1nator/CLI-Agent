# Git Branch Structure

## ✅ Branch Cleanup Complete (2025-10-16)

All repositories now use standardized branch naming:

## Current Branch Structure

### Local
- `main` - Active development (default) ✓
- `production` - Release branch ✓

### git_private (CLI-Agent_DEV)
- `main` - Private development ✓
- `production` - Private releases ✓

### git_public (CLI-Agent)
- `main` - Public repository (default) ✓

## Branch Usage

### `main`
- Active development branch
- All feature work and fixes
- Continuous integration runs on push
- Push to both `git_private/main` (backup) and `git_public/main` (public)

### `production`
- Release branch for stable releases
- Updated via `saas_scripts/release_to_production.sh`
- Squashes all commits from `main` into single release commits
- Deploy to production infrastructure from this branch

## Workflow

```bash
# Normal development
git checkout main
# ... make changes ...
git commit -m "feature: add something"
git push  # pushes to git_private/main

# When ready to release
./saas_scripts/release_to_production.sh
# Enter release version (e.g., "v1.2.0 - Feature description")
# Script will:
# 1. Switch to production
# 2. Squash all new commits from main
# 3. Create single release commit
# 4. Optionally push to git_public/production

# Push main to public when ready
git push git_public main
```

## Deleted Branches
- ~~`master`~~ - Replaced with `main`
- ~~`auto-commandeer`~~ - Old feature branch
- ~~`cleanup-branch`~~ - Old cleanup branch
- ~~`CLI-Agent_RELEASE`~~ - Old release branch
- ~~`temp_branch_*`~~ - Temporary branches
