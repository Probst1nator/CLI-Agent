#!/bin/bash
# Release script: Squash all commits from main into production as a single release

set -e  # Exit on error

# Source deployment logger for persistent audit logging
source "$(dirname "$0")/deployment_logger.sh"

# Ensure we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    die "You must be on the 'main' branch to release. Current branch: $CURRENT_BRANCH"
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    log_error "You have uncommitted changes. Please commit or stash them first."
    git status --short
    exit 1
fi

# Prompt for version/release title
echo -e "${YELLOW}Enter the release version/title (e.g., 'v1.2.0 - Multi-user support'):${NC}"
read -r RELEASE_TITLE

if [ -z "$RELEASE_TITLE" ]; then
    die "Release title cannot be empty"
fi

# Show current state
log_info "Current branches:"
git branch -v

# Switch to production branch
log_info "Switching to production branch..."
git checkout production

# Count commits between production and main
COMMIT_COUNT=$(git rev-list --count production..main)
log_info "Found $COMMIT_COUNT commits to squash"

if [ "$COMMIT_COUNT" -eq 0 ]; then
    log_warning "Production is already up to date with main. Nothing to release."
    git checkout main
    exit 0
fi

# Squash all commits from production..main into one
log_info "Squashing commits into single release..."
git reset --soft main
git reset --soft HEAD~$COMMIT_COUNT

# Create a single squashed commit with the release title
log_info "Creating release commit..."
git commit -m "$RELEASE_TITLE"

# Show the result
log_success "Release commit created:"
git log -1 --oneline

# Ask if user wants to push
echo -e "\n${YELLOW}Push to origin/production? (y/n):${NC}"
read -r PUSH_CONFIRM

if [ "$PUSH_CONFIRM" = "y" ] || [ "$PUSH_CONFIRM" = "Y" ]; then
    log_info "Pushing to origin/production..."
    git push origin production --force-with-lease || die "Failed to push to origin/production"
    log_success "Successfully pushed to origin/production"
else
    log_warning "Skipped push. Run 'git push origin production --force-with-lease' manually when ready."
fi

# Switch back to main
log_info "Switching back to main branch..."
git checkout main

success
