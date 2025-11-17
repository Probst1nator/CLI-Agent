# tools/github.py
"""
Tool for safe GitHub operations (PR, issues, status checks).
"""
import json
import re
import subprocess


class github:
    """Perform GitHub operations safely using gh CLI."""

    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'github'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "github",
            "description": (
                "Execute GitHub operations safely using gh CLI. "
                "Supports: create PR, list PRs, view PR, create issue, list issues, "
                "check PR status, get repo info, view workflows. "
                "SAFETY: All operations are read-only except PR/issue creation. "
                "Requires 'gh' CLI tool to be installed and authenticated."
            ),
            "example": (
                "<github><operation>pr-create</operation><title>Fix bug in auth</title><body>Fixes authentication issue</body></github>\n"
                "<github><operation>pr-list</operation></github>\n"
                "<github><operation>pr-status</operation><pr_number>123</pr_number></github>\n"
                "<github><operation>issue-create</operation><title>Bug report</title><body>Description</body></github>\n"
                "<github><operation>repo-info</operation></github>"
            )
        }

    @staticmethod
    def _check_gh_cli() -> tuple[bool, str]:
        """Check if gh CLI is installed and authenticated."""
        try:
            # Check if gh is installed
            result = subprocess.run(
                ['gh', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return (False, "GitHub CLI (gh) is not installed. Install from: https://cli.github.com/")

            # Check if authenticated
            result = subprocess.run(
                ['gh', 'auth', 'status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return (False, "GitHub CLI is not authenticated. Run: gh auth login")

            return (True, "GitHub CLI ready")

        except FileNotFoundError:
            return (False, "GitHub CLI (gh) is not installed. Install from: https://cli.github.com/")
        except Exception as e:
            return (False, f"Error checking gh CLI: {str(e)}")

    @staticmethod
    def run(content: str) -> str:
        """
        Execute a GitHub operation safely.

        Args:
            content: XML containing operation and parameters.

        Returns:
            Operation result or error message.
        """
        try:
            # Check gh CLI availability
            is_available, msg = github._check_gh_cli()
            if not is_available:
                return f"Error: {msg}"

            # Parse XML
            operation_match = re.search(r'<operation>(.*?)</operation>', content, re.DOTALL)

            if not operation_match:
                return "Error: Invalid format. Input must contain an <operation> tag."

            operation = operation_match.group(1).strip().lower()

            if not operation:
                return "Error: Operation cannot be empty."

            # Parse optional parameters
            title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
            body_match = re.search(r'<body>(.*?)</body>', content, re.DOTALL)
            pr_number_match = re.search(r'<pr_number>(.*?)</pr_number>', content, re.DOTALL)
            issue_number_match = re.search(r'<issue_number>(.*?)</issue_number>', content, re.DOTALL)
            base_match = re.search(r'<base>(.*?)</base>', content, re.DOTALL)
            head_match = re.search(r'<head>(.*?)</head>', content, re.DOTALL)
            labels_match = re.search(r'<labels>(.*?)</labels>', content, re.DOTALL)
            limit_match = re.search(r'<limit>(.*?)</limit>', content, re.DOTALL)

            title = title_match.group(1).strip() if title_match else None
            body = body_match.group(1).strip() if body_match else None
            pr_number = pr_number_match.group(1).strip() if pr_number_match else None
            issue_number = issue_number_match.group(1).strip() if issue_number_match else None
            base = base_match.group(1).strip() if base_match else None
            head = head_match.group(1).strip() if head_match else None
            labels = labels_match.group(1).strip() if labels_match else None
            limit = limit_match.group(1).strip() if limit_match else '10'

            # Execute operation
            gh_cmd = []

            if operation == 'pr-create':
                if not title:
                    return "Error: 'pr-create' requires <title> tag"

                gh_cmd = ['gh', 'pr', 'create', '--title', title]

                if body:
                    gh_cmd.extend(['--body', body])
                else:
                    gh_cmd.append('--body')
                    gh_cmd.append('')

                if base:
                    gh_cmd.extend(['--base', base])
                if head:
                    gh_cmd.extend(['--head', head])
                if labels:
                    gh_cmd.extend(['--label', labels])

            elif operation == 'pr-list':
                gh_cmd = ['gh', 'pr', 'list', '--limit', limit]

            elif operation == 'pr-view':
                if not pr_number:
                    return "Error: 'pr-view' requires <pr_number> tag"
                gh_cmd = ['gh', 'pr', 'view', pr_number]

            elif operation == 'pr-status':
                if not pr_number:
                    return "Error: 'pr-status' requires <pr_number> tag"
                gh_cmd = ['gh', 'pr', 'view', pr_number, '--json', 'state,statusCheckRollup,reviewDecision,mergeable']

            elif operation == 'pr-checks':
                if not pr_number:
                    return "Error: 'pr-checks' requires <pr_number> tag"
                gh_cmd = ['gh', 'pr', 'checks', pr_number]

            elif operation == 'issue-create':
                if not title:
                    return "Error: 'issue-create' requires <title> tag"

                gh_cmd = ['gh', 'issue', 'create', '--title', title]

                if body:
                    gh_cmd.extend(['--body', body])
                if labels:
                    gh_cmd.extend(['--label', labels])

            elif operation == 'issue-list':
                gh_cmd = ['gh', 'issue', 'list', '--limit', limit]

            elif operation == 'issue-view':
                if not issue_number:
                    return "Error: 'issue-view' requires <issue_number> tag"
                gh_cmd = ['gh', 'issue', 'view', issue_number]

            elif operation == 'repo-info':
                gh_cmd = ['gh', 'repo', 'view', '--json', 'name,owner,description,url,defaultBranch,isPrivate']

            elif operation == 'workflow-list':
                gh_cmd = ['gh', 'workflow', 'list']

            elif operation == 'workflow-view':
                gh_cmd = ['gh', 'run', 'list', '--limit', limit]

            elif operation == 'auth-status':
                gh_cmd = ['gh', 'auth', 'status']

            else:
                return f"Error: Unsupported operation '{operation}'. Supported: pr-create, pr-list, pr-view, pr-status, pr-checks, issue-create, issue-list, issue-view, repo-info, workflow-list, workflow-view, auth-status"

            # Execute command
            result = subprocess.run(
                gh_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Format output
            output = result.stdout
            if result.stderr:
                # GitHub CLI sometimes outputs info to stderr
                if result.returncode == 0:
                    output += f"\n{result.stderr}"
                else:
                    output = f"{result.stderr}\n{output}"

            if result.returncode != 0:
                return f"GitHub operation failed:\n{output.strip()}"

            # Try to parse JSON for pr-status and repo-info
            if operation in ['pr-status', 'repo-info']:
                try:
                    parsed = json.loads(output)
                    return json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    pass

            return output.strip() if output.strip() else f"[GitHub {operation} completed successfully]"

        except subprocess.TimeoutExpired:
            return "Error: GitHub command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing GitHub operation: {str(e)}"
