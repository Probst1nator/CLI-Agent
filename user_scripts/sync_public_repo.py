#!/usr/bin/env python3
"""
Sync script for maintaining public repository from private development repo.

Three-stage workflow:
  1. main (private dev) → PR → public-sync (review for public readiness)
  2. public-sync → sync script → public-release (filtered version)
  3. public-release → git_public/main (public repository)

This script handles stage 2-3:
1. Deletes privileged tools (bash, writefile, etc.) from public version
2. Excludes internal development directories
3. Filters out secrets and logs
4. Creates a clean public-ready branch with safe tools only

Usage:
    # From public-sync branch:
    python user_scripts/sync_public_repo.py --dry-run     # Preview changes
    python user_scripts/sync_public_repo.py --prepare     # Prepare public branch
    python user_scripts/sync_public_repo.py --push        # Push to public remote

See docs/RELEASE_WORKFLOW.md for complete workflow.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


class PublicRepoSync:
    def __init__(self, repo_root: Path, config_path: Path):
        self.repo_root = repo_root
        self.config = self._load_config(config_path)
        self.public_branch = "public-release"

    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            return json.load(f)

    def _run_git(self, *args) -> subprocess.CompletedProcess:
        """Run git command in repo root."""
        return subprocess.run(
            ['git', *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True
        )

    def check_prerequisites(self, skip_prompts: bool = False) -> bool:
        """Verify git status and prerequisites."""
        print("🔍 Checking prerequisites...")

        # Check if we're in a git repo
        try:
            self._run_git('status')
        except subprocess.CalledProcessError:
            print("❌ Not a git repository")
            return False

        # Check for uncommitted changes
        result = self._run_git('status', '--porcelain')
        if result.stdout.strip():
            print("⚠️  Warning: Uncommitted changes detected:")
            print(result.stdout)
            if not skip_prompts:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return False
            else:
                print("⚠️  Continuing (--dry-run mode)")

        # Check if public remote exists
        result = self._run_git('remote', '-v')
        if 'git_public' not in result.stdout:
            print("❌ git_public remote not configured")
            print("Add with: git remote add git_public <public-repo-url>")
            return False

        print("✅ Prerequisites OK")
        return True

    def delete_privileged_tool(self, tool_path: Path) -> None:
        """Delete privileged tool from public repository."""
        if tool_path.exists():
            tool_path.unlink()
            print(f"  ✅ Deleted: {tool_path.name}")
        else:
            print(f"  ⏭️  Not found: {tool_path.name}")

    def prepare_public_branch(self, dry_run: bool = False) -> bool:
        """Prepare public branch with filtered content."""
        print(f"\n{'🔍 DRY RUN MODE' if dry_run else '🚀 PREPARING PUBLIC BRANCH'}")
        print("=" * 80)

        if not dry_run:
            # Verify public-sync branch exists
            try:
                self._run_git('rev-parse', '--verify', 'public-sync')
            except subprocess.CalledProcessError:
                print("❌ public-sync branch doesn't exist")
                print("Create it with: git checkout -b public-sync && git push git_private public-sync")
                return False

            # Create/checkout public branch from public-sync
            try:
                self._run_git('checkout', '-b', self.public_branch)
                print(f"✅ Created branch: {self.public_branch}")
            except subprocess.CalledProcessError:
                self._run_git('checkout', self.public_branch)
                self._run_git('merge', 'public-sync')
                print(f"✅ Updated branch: {self.public_branch}")

        # Delete privileged tools
        print("\n🗑️  Deleting privileged tools...")
        tools_dir = self.repo_root / "tools"

        for tool_name in self.config['tools']['privileged']['tools']:
            tool_path = tools_dir / tool_name
            if dry_run:
                if tool_path.exists():
                    print(f"  [DRY RUN] Would delete: {tool_name}")
            else:
                self.delete_privileged_tool(tool_path)

        # Remove excluded directories
        self._remove_excluded_content(dry_run)

        # Update .gitignore
        self._update_gitignore(dry_run)

        if not dry_run:
            # Commit changes
            self._run_git('add', '.')
            try:
                self._run_git(
                    'commit',
                    '-m',
                    'Prepare public release: remove privileged tools, filter internal dirs'
                )
                print("\n✅ Changes committed to public branch")
            except subprocess.CalledProcessError:
                print("\n⚠️  No changes to commit")

        return True

    def _remove_excluded_content(self, dry_run: bool):
        """Remove excluded directories and files."""
        print("\n🗑️  Processing excluded content...")

        for dir_pattern in self.config['tools']['excluded']['directories']:
            dir_path = self.repo_root / dir_pattern.rstrip('/')
            if dir_path.exists():
                if dry_run:
                    print(f"  [DRY RUN] Would exclude: {dir_pattern}")
                else:
                    # Use git rm to remove from index
                    try:
                        self._run_git('rm', '-rf', str(dir_path.relative_to(self.repo_root)))
                        print(f"  ✅ Excluded: {dir_pattern}")
                    except subprocess.CalledProcessError:
                        print(f"  ⚠️  Could not remove: {dir_pattern}")

    def _update_gitignore(self, dry_run: bool):
        """Add public-specific gitignore entries."""
        gitignore_additions = """
# Public repository exclusions
.env
*.log
deployment_logs/
saas_scripts/
docker_files/traefik/
"""

        gitignore_path = self.repo_root / ".gitignore"

        if dry_run:
            print("\n📝 Would update: .gitignore")
            return

        # Check if additions already present
        if gitignore_path.exists():
            with open(gitignore_path) as f:
                current = f.read()
            if 'Public repository exclusions' in current:
                print("\n⏭️  .gitignore already updated")
                return

        with open(gitignore_path, 'a') as f:
            f.write(gitignore_additions)
        print("\n✅ Updated: .gitignore")

    def push_to_public(self, force: bool = False) -> bool:
        """Push public branch to public remote."""
        print("\n🚀 Pushing to public repository...")

        # Confirm
        if not force:
            print("\n⚠️  This will push to the public repository!")
            print("Remote:", self._run_git('remote', 'get-url', 'git_public').stdout.strip())
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Aborted")
                return False

        # Push
        try:
            self._run_git('push', 'git_public', f'{self.public_branch}:main', '--force' if force else '')
            print("✅ Successfully pushed to public repository!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Push failed: {e.stderr}")
            return False

    def cleanup(self):
        """Clean up and return to original branch."""
        print("\n🧹 Cleaning up...")
        try:
            # Return to public-sync (the source branch for public releases)
            self._run_git('checkout', 'public-sync')
            print("✅ Returned to public-sync branch")
        except subprocess.CalledProcessError:
            print("⚠️  Could not return to public-sync branch")


def main():
    parser = argparse.ArgumentParser(
        description="Sync public repository (removes privileged tools, keeps safe tools only)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Preview changes without modifying anything"
    )
    parser.add_argument(
        '--prepare',
        action='store_true',
        help="Prepare public branch (delete privileged tools, filter internal dirs)"
    )
    parser.add_argument(
        '--push',
        action='store_true',
        help="Push public branch to remote (requires --prepare first)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force push without confirmation"
    )

    args = parser.parse_args()

    # Determine repository root
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / ".public-repo-config.json"

    # Create sync manager
    sync = PublicRepoSync(repo_root, config_path)

    # Check prerequisites
    if not sync.check_prerequisites(skip_prompts=args.dry_run):
        sys.exit(1)

    try:
        # Dry run
        if args.dry_run:
            sync.prepare_public_branch(dry_run=True)
            print("\n" + "=" * 80)
            print("✅ Dry run complete. Run with --prepare to apply changes.")
            return

        # Prepare branch
        if args.prepare:
            sync.prepare_public_branch(dry_run=False)
            print("\n" + "=" * 80)
            print("✅ Public branch prepared. Review changes, then run with --push")
            print(f"\nTo review: git diff public-sync..{sync.public_branch}")
            return

        # Push
        if args.push:
            # Check if public branch exists
            try:
                sync._run_git('rev-parse', '--verify', sync.public_branch)
            except subprocess.CalledProcessError:
                print(f"❌ Public branch '{sync.public_branch}' doesn't exist")
                print("Run with --prepare first")
                sys.exit(1)

            sync.push_to_public(force=args.force)
            return

        # No action specified
        parser.print_help()
        print("\n💡 Suggested workflow:")
        print("  1. Create PR: main → public-sync (review on GitHub)")
        print("  2. Merge PR and checkout public-sync")
        print("  3. python user_scripts/sync_public_repo.py --dry-run")
        print("  4. python user_scripts/sync_public_repo.py --prepare")
        print("  5. Review: git diff public-sync..public-release")
        print("  6. python user_scripts/sync_public_repo.py --push")
        print("\nSee docs/RELEASE_WORKFLOW.md for details")

    finally:
        if not args.dry_run and (args.prepare or args.push):
            sync.cleanup()


if __name__ == "__main__":
    main()
