#!/bin/bash
# Release script: Squash all commits from main into production as a single release

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Release to Production ===${NC}\n"

# Ensure we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${RED}Error: You must be on the 'main' branch to release.${NC}"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: You have uncommitted changes. Please commit or stash them first.${NC}"
    git status --short
    exit 1
fi

# Prompt for version/release title
echo -e "${YELLOW}Enter the release version/title (e.g., 'v1.2.0 - Multi-user support'):${NC}"
read -r RELEASE_TITLE

if [ -z "$RELEASE_TITLE" ]; then
    echo -e "${RED}Error: Release title cannot be empty.${NC}"
    exit 1
fi

# Show current state
echo -e "\n${GREEN}Current branches:${NC}"
git branch -v

# Switch to production branch
echo -e "\n${GREEN}Switching to production branch...${NC}"
git checkout production

# Reset production to main's commit, then reset tree (proper squash)
echo -e "\n${GREEN}Resetting production to match main...${NC}"
git reset --hard main
git reset --soft HEAD~1

# Create a single squashed commit with the release title
echo -e "\n${GREEN}Creating release commit...${NC}"
git commit -m "$RELEASE_TITLE" --allow-empty

# Show the result
echo -e "\n${GREEN}✓ Release commit created:${NC}"
git log -1 --oneline

# Ask if user wants to push
echo -e "\n${YELLOW}Push to git_public/production? (y/n):${NC}"
read -r PUSH_CONFIRM

if [ "$PUSH_CONFIRM" = "y" ] || [ "$PUSH_CONFIRM" = "Y" ]; then
    echo -e "\n${GREEN}Pushing to git_public/production...${NC}"
    git push git_public production --force-with-lease
    echo -e "${GREEN}✓ Successfully pushed to git_public/production${NC}"
else
    echo -e "${YELLOW}Skipped push. Run 'git push git_public production --force-with-lease' manually when ready.${NC}"
fi

# Switch back to main
echo -e "\n${GREEN}Switching back to main branch...${NC}"
git checkout main

echo -e "\n${GREEN}=== Release complete! ===${NC}"
