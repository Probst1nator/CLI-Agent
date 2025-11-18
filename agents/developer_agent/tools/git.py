# tools/git.py
"""
Tool for git version control operations.
"""
import re
import subprocess


class git:
    """Perform git operations like status, diff, commit, push, branch, etc."""

    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'git'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "git",
            "description": (
                "Execute git commands for version control. "
                "Supports: status, diff, log, add, commit, push, pull, branch, checkout, etc. "
                "SAFETY: Always show git status and diff before committing. "
                "Use descriptive commit messages. Verify before force operations."
            ),
            "example": (
                "<git><command>status</command></git>\n"
                "<git><command>diff</command></git>\n"
                "<git><command>add</command><files>src/main.py</files></git>\n"
                "<git><command>commit</command><message>Add new feature</message></git>\n"
                "<git><command>push</command></git>\n"
                "<git><command>log</command><args>--oneline -10</args></git>"
            )
        }

    @staticmethod
    def _is_git_repo() -> tuple[bool, str]:
        """Check if current directory is a git repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return (result.returncode == 0, result.stderr.strip())
        except Exception as e:
            return (False, str(e))

    @staticmethod
    def run(content: str) -> str:
        """
        Execute a git command.

        Args:
            content: XML containing command and optional parameters.

        Returns:
            Command output or error message.
        """
        try:
            # Check if we're in a git repository
            is_repo, error_msg = git._is_git_repo()
            if not is_repo:
                return f"Error: Not a git repository. {error_msg}"

            # Parse XML
            command_match = re.search(r'<command>(.*?)</command>', content, re.DOTALL)

            if not command_match:
                return "Error: Invalid format. Input must contain a <command> tag."

            command = command_match.group(1).strip().lower()

            if not command:
                return "Error: Command cannot be empty."

            # Parse optional parameters
            files_match = re.search(r'<files>(.*?)</files>', content, re.DOTALL)
            message_match = re.search(r'<message>(.*?)</message>', content, re.DOTALL)
            args_match = re.search(r'<args>(.*?)</args>', content, re.DOTALL)
            branch_match = re.search(r'<branch>(.*?)</branch>', content, re.DOTALL)

            files = files_match.group(1).strip() if files_match else None
            message = message_match.group(1).strip() if message_match else None
            args = args_match.group(1).strip() if args_match else None
            branch = branch_match.group(1).strip() if branch_match else None

            # Build git command based on operation
            git_cmd = ['git', command]

            if command == 'status':
                git_cmd = ['git', 'status']

            elif command == 'diff':
                git_cmd = ['git', 'diff']
                if args:
                    git_cmd.extend(args.split())
                if files:
                    git_cmd.append(files)

            elif command == 'log':
                git_cmd = ['git', 'log']
                if args:
                    git_cmd.extend(args.split())
                else:
                    git_cmd.extend(['--oneline', '-10'])  # Default: last 10 commits

            elif command == 'add':
                if not files:
                    return "Error: 'add' command requires <files> tag (e.g., '.', 'file.py', 'src/*.py')"
                git_cmd = ['git', 'add'] + files.split()

            elif command == 'commit':
                if not message:
                    return "Error: 'commit' command requires <message> tag"
                git_cmd = ['git', 'commit', '-m', message]

            elif command == 'push':
                git_cmd = ['git', 'push']
                if args:
                    git_cmd.extend(args.split())

            elif command == 'pull':
                git_cmd = ['git', 'pull']
                if args:
                    git_cmd.extend(args.split())

            elif command == 'branch':
                git_cmd = ['git', 'branch']
                if branch:
                    git_cmd.append(branch)
                if args:
                    git_cmd.extend(args.split())

            elif command == 'checkout':
                if not branch and not files:
                    return "Error: 'checkout' command requires <branch> or <files> tag"
                git_cmd = ['git', 'checkout']
                if branch:
                    git_cmd.append(branch)
                if files:
                    git_cmd.extend(files.split())

            elif command in ['fetch', 'merge', 'rebase', 'reset', 'stash', 'tag', 'show', 'blame']:
                # Allow these common git commands
                git_cmd = ['git', command]
                if args:
                    git_cmd.extend(args.split())
                if files:
                    git_cmd.extend(files.split())

            else:
                return f"Error: Unsupported git command '{command}'. Supported: status, diff, log, add, commit, push, pull, branch, checkout, fetch, merge, rebase, reset, stash, tag, show, blame"

            # Safety check for dangerous operations
            dangerous_patterns = [
                '--force',
                '-f',
                '--hard',
                'push origin --delete',
            ]

            cmd_str = ' '.join(git_cmd)
            for pattern in dangerous_patterns:
                if pattern in cmd_str:
                    return f"Warning: Potentially destructive operation detected ('{pattern}'). Please confirm with user before executing: {cmd_str}"

            # Execute git command
            result = subprocess.run(
                git_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Format output
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"

            if result.returncode != 0:
                output = f"Git command failed (exit code {result.returncode}):\n{output}"

            return output.strip() if output.strip() else f"[Git {command} completed successfully]"

        except subprocess.TimeoutExpired:
            return "Error: Git command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing git command: {str(e)}"
