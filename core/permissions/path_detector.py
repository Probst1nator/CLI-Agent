# path_detector.py
"""
Path detection utility for XML blocks in CLI-Agent tools.
Detects file paths in XML tags containing 'path' substring and determines tool actions.
"""
import re
import os
from typing import Dict, List, Tuple, Optional, Any

class PathDetector:
    # Define permission groups for related tools
    PERMISSION_GROUPS = {
        'file_editing': ['editfile', 'writefile', 'appendfile'],
        'file_reading': ['readfile', 'viewfiles'],
        'code_execution': ['bash', 'python'],
        'file_management': ['bash', 'python']  # Only when they contain file operations
    }

    # Define common development tool patterns for smart quick actions
    COMMON_DEV_TOOLS = {
        'linting': {
            'patterns': ['ruff', 'flake8', 'pylint', 'black', 'isort', 'mypy'],
            'quick_key': 'r',  # Quick action key
            'display_name': 'linting tools (ruff, black, etc.)',
            'permission_scope': 'project'  # file, directory, project, global
        },
        'testing': {
            'patterns': ['pytest', 'python -m pytest', 'python -m unittest', 'nose', 'tox'],
            'quick_key': 't',
            'display_name': 'testing commands (pytest, etc.)',
            'permission_scope': 'project'
        },
        'building': {
            'patterns': ['npm install', 'npm run', 'yarn install', 'pip install', 'poetry install', 'make', 'cargo build'],
            'quick_key': 'b',
            'display_name': 'build tools (npm, pip, etc.)',
            'permission_scope': 'project'
        },
        'formatting': {
            'patterns': ['black', 'prettier', 'autopep8', 'yapf'],
            'quick_key': 'f',
            'display_name': 'code formatters',
            'permission_scope': 'directory'
        },
        'git': {
            'patterns': ['git add', 'git commit', 'git status', 'git diff', 'git push', 'git pull'],
            'quick_key': 'g',
            'display_name': 'git commands',
            'permission_scope': 'project'
        },
        'safe_commands': {
            'patterns': [
                'ls', 'ls -l', 'ls -la', 'ls -al', 'ls -lh', 'ls -R', 'ls -lr', 'ls -ltr',
                'pwd', 'whoami', 'date', 'uptime', 'hostname', 'uname', 'uname -a',
                'df -h', 'du -h', 'free -h', 'ps aux', 'top -n1', 'which',
                'cat /etc/os-release', 'lsb_release -a', 'env', 'printenv',
                'wc -l', 'head', 'tail', 'grep', 'find . -name', 'find . -type f',
                'tree', 'file', 'stat', 'id', 'groups'
            ],
            'quick_key': 'r',
            'display_name': 'read-only/safe commands (ls, pwd, etc.)',
            'permission_scope': 'project'
        },
        'search': {
            'patterns': ['grep', 'rg', 'ripgrep', 'find', 'ack', 'ag'],
            'quick_key': 's',
            'display_name': 'search tools (grep, find, etc.)',
            'permission_scope': 'global'
        }
    }

    # Define sensitive keywords that should trigger automatic blocking
    SENSITIVE_KEYWORDS = {
        'sudo',           # Environment files
        '.env',           # Environment files
        'password',       # Password-related content
        'api_key',        # API keys
        'secret',         # Secrets
        'token',          # Authentication tokens
        'private_key',    # Private keys
        'id_rsa',         # SSH keys
        'id_dsa',         # SSH keys
    }

    # Reverse mapping: tool -> groups it belongs to
    TOOL_TO_GROUPS = {}
    for group, tools in PERMISSION_GROUPS.items():
        for tool in tools:
            if tool not in TOOL_TO_GROUPS:
                TOOL_TO_GROUPS[tool] = []
            TOOL_TO_GROUPS[tool].append(group)

    @staticmethod
    def get_tool_groups(tool_name: str) -> List[str]:
        """Get all permission groups that a tool belongs to."""
        return PathDetector.TOOL_TO_GROUPS.get(tool_name, [])

    @staticmethod
    def get_group_tools(group_name: str) -> List[str]:
        """Get all tools that belong to a permission group."""
        return PathDetector.PERMISSION_GROUPS.get(group_name, [])

    @staticmethod
    def detect_dev_tool_category(command_pattern: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Detect what category of development tool this command represents.

        Args:
            command_pattern: The command pattern to analyze

        Returns:
            Tuple of (category_name, category_config) or (None, None) if no match
        """
        command_lower = command_pattern.lower().strip()

        for category, config in PathDetector.COMMON_DEV_TOOLS.items():
            # Special handling for safe commands with flexible pattern matching
            if category == 'safe_commands':
                if PathDetector._is_safe_command(command_lower):
                    return category, config
            else:
                # Regular substring matching for other categories
                for pattern in config['patterns']:
                    if pattern.lower() in command_lower:
                        return category, config

        return None, None

    @staticmethod
    def _is_safe_command(command_lower: str) -> bool:
        """
        Check if a command is a safe read-only command using flexible pattern matching.

        Args:
            command_lower: Command string in lowercase

        Returns:
            True if the command is considered safe
        """
        # Define safe command patterns with regex
        safe_patterns = [
            r'^ls(\s+(-[lartRhAFSG1]*\s*)*(\S+\s*)*)?$',  # ls with any common flags and optional paths
            r'^pwd\s*$',                                   # pwd
            r'^whoami\s*$',                               # whoami
            r'^date(\s+.*)?$',                            # date with optional args
            r'^uptime\s*$',                               # uptime
            r'^hostname\s*$',                             # hostname
            r'^uname(\s+-[a-zA-Z]*)?$',                   # uname with flags
            r'^df\s+-h(\s+.*)?$',                         # df -h
            r'^du\s+-h(\s+.*)?$',                         # du -h
            r'^free\s+-h\s*$',                            # free -h
            r'^ps(\s+aux)?\s*$',                          # ps or ps aux
            r'^top\s+-n1\s*$',                            # top -n1
            r'^which\s+\S+\s*$',                          # which command
            r'^cat\s+/etc/os-release\s*$',                # cat /etc/os-release
            r'^lsb_release\s+-a\s*$',                     # lsb_release -a
            r'^env\s*$',                                  # env
            r'^printenv(\s+\S+)?\s*$',                    # printenv
            r'^wc\s+-l(\s+.*)?$',                         # wc -l
            r'^head(\s+-\d+)?(\s+.*)?$',                  # head
            r'^tail(\s+-\d+)?(\s+.*)?$',                  # tail
            r'^grep(\s+.*)?$',                            # grep (read-only when not redirected)
            r'^find\s+\.(\s+.*)?$',                       # find . with any safe options
            r'^tree(\s+.*)?$',                            # tree
            r'^file\s+.*$',                               # file
            r'^stat\s+.*$',                               # stat
            r'^id\s*$',                                   # id
            r'^groups\s*$',                               # groups
        ]

        # Check if command matches any safe pattern
        for pattern in safe_patterns:
            if re.match(pattern, command_lower):
                return True

        return False

    @staticmethod
    def get_smart_quick_action(tool_name: str, command_pattern: str, file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a smart quick action suggestion for this command.

        Args:
            tool_name: Name of the tool
            command_pattern: The command being executed
            file_path: Optional file path being operated on

        Returns:
            Dictionary with quick action details or None if no smart action available
        """
        # Only provide quick actions for bash commands (where dev tools are typically run)
        if tool_name != 'bash':
            return None

        category, config = PathDetector.detect_dev_tool_category(command_pattern)
        if not category or not config:
            return None

        # Determine the permission scope based on file path and category config
        scope_path = None
        scope_description = ""

        if file_path:
            abs_path = os.path.abspath(file_path)

            if config['permission_scope'] == 'file':
                scope_path = abs_path
                scope_description = f"this file ({os.path.basename(abs_path)})"
            elif config['permission_scope'] == 'directory':
                scope_path = os.path.dirname(abs_path) + "/*"
                scope_description = f"this directory ({os.path.basename(os.path.dirname(abs_path))})"
            elif config['permission_scope'] == 'project':
                # Find project root (look for common project indicators)
                project_root = PathDetector._find_project_root(abs_path)
                if project_root:
                    scope_path = project_root + "/*"
                    scope_description = f"this project ({os.path.basename(project_root)})"
                else:
                    scope_path = os.path.dirname(abs_path) + "/*"
                    scope_description = f"this directory ({os.path.basename(os.path.dirname(abs_path))})"
        else:
            # No file path, use project scope if possible
            cwd = os.getcwd()
            if config['permission_scope'] == 'project':
                project_root = PathDetector._find_project_root(cwd)
                if project_root:
                    scope_path = project_root + "/*"
                    scope_description = f"this project ({os.path.basename(project_root)})"
                else:
                    scope_path = cwd + "/*"
                    scope_description = f"current directory ({os.path.basename(cwd)})"
            elif config['permission_scope'] == 'global':
                scope_path = category  # Use category name as global pattern
                scope_description = "globally"
            else:
                scope_path = cwd + "/*"
                scope_description = f"current directory ({os.path.basename(cwd)})"

        return {
            'category': category,
            'quick_key': config['quick_key'],
            'display_name': config['display_name'],
            'scope_path': scope_path,
            'scope_description': scope_description,
            'permission_scope': config['permission_scope']
        }

    @staticmethod
    def _find_project_root(start_path: str) -> Optional[str]:
        """
        Find the project root by looking for common project indicators.

        Args:
            start_path: Path to start searching from

        Returns:
            Project root path or None if not found
        """
        current = os.path.abspath(start_path)

        # Common project root indicators
        project_indicators = [
            '.git',
            'package.json',
            'pyproject.toml',
            'requirements.txt',
            'Cargo.toml',
            'pom.xml',
            'build.gradle',
            'Makefile',
            '.gitignore'
        ]

        # Walk up the directory tree
        while current != os.path.dirname(current):  # Not at root
            for indicator in project_indicators:
                if os.path.exists(os.path.join(current, indicator)):
                    return current
            current = os.path.dirname(current)

        return None

    @staticmethod
    def extract_paths_from_xml(xml_content: str, tool_name: str) -> List[Tuple[str, str]]:
        """
        Extract file paths from XML content by finding tags containing 'path' or from command content.

        Args:
            xml_content: XML content to parse
            tool_name: Name of the tool being executed

        Returns:
            List of (tag_name, file_path) tuples found in the XML
        """
        paths = []

        # Find all XML tags that contain 'path' in their name
        path_tag_pattern = r'<([^>]*path[^>]*)>(.*?)</\1>'
        matches = re.finditer(path_tag_pattern, xml_content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            tag_name = match.group(1).strip()
            content = match.group(2).strip()

            # Only consider non-empty content as potential file paths
            if content:
                # Resolve to absolute path
                abs_path = os.path.abspath(content)
                paths.append((tag_name, abs_path))

        # For bash and python tools, also extract file paths from command content
        if tool_name in ["bash", "python"] and not paths:
            command_paths = PathDetector._extract_paths_from_command_content(xml_content, tool_name)
            paths.extend(command_paths)

        return paths

    @staticmethod
    def _is_relevant_path(file_path: str) -> bool:
        """
        Determine if a file path is relevant to show to users.
        Filters out system cache files, bytecode, and other irrelevant paths.

        Args:
            file_path: The file path to evaluate

        Returns:
            True if the path is user-relevant
        """
        if not file_path:
            return False

        # Normalize path for checking
        path_lower = file_path.lower()

        # Filter out Python bytecode cache files
        irrelevant_patterns = [
            '.pyc',           # Compiled Python bytecode
            '.pyo',           # Optimized Python bytecode
            '__pycache__',    # Python cache directory
            '.git/',          # Git internal files
            '.venv/',         # Virtual environment
            'venv/',          # Virtual environment
            '.env/',          # Environment directory
            'node_modules/',  # Node.js modules
            '.cache/',        # General cache directories
            'site-packages/', # Python site packages
            'lib/python',     # Python library paths
            '/tmp/',          # Temporary files
            '/var/tmp/',      # Temporary files
            '.DS_Store',      # macOS metadata
            'Thumbs.db',      # Windows metadata
        ]

        # Check if path contains any irrelevant patterns
        for pattern in irrelevant_patterns:
            if pattern in path_lower:
                return False

        # Filter out very long paths that are likely system-generated
        if len(file_path) > 200:
            return False

        return True

    @staticmethod
    def _get_path_relevance_score(file_path: str) -> int:
        """
        Get a relevance score for a file path. Higher scores are more relevant.

        Args:
            file_path: The file path to score

        Returns:
            Integer relevance score (higher = more relevant)
        """
        if not file_path:
            return 0

        score = 0
        path_lower = file_path.lower()

        # Prefer user source code files
        source_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php', '.sh']
        if any(path_lower.endswith(ext) for ext in source_extensions):
            score += 10

        # Prefer files in current working directory or subdirectories
        cwd = os.getcwd().lower()
        if path_lower.startswith(cwd):
            score += 5

        # Prefer shorter, simpler paths (more likely to be user files)
        if len(file_path) < 100:
            score += 3

        # Prefer files with common user-relevant names
        relevant_patterns = ['main', 'index', 'app', 'src/', 'lib/', 'utils/', 'tools/', 'scripts/']
        if any(pattern in path_lower for pattern in relevant_patterns):
            score += 2

        return score

    @staticmethod
    def _extract_paths_from_command_content(xml_content: str, tool_name: str) -> List[Tuple[str, str]]:
        """
        Extract file paths from bash/python command content.

        Args:
            xml_content: XML content containing the command
            tool_name: Name of the tool (bash or python)

        Returns:
            List of (tag_name, file_path) tuples found in the command
        """
        paths = []

        # Remove XML tags to get the actual command content
        command_content = re.sub(r'<[^>]*>', '', xml_content).strip()

        if tool_name == "bash":
            all_paths = PathDetector._extract_bash_paths(command_content)
        elif tool_name == "python":
            all_paths = PathDetector._extract_python_paths(command_content)
        else:
            all_paths = []

        # Filter paths to only include relevant ones
        for tag_name, file_path in all_paths:
            if PathDetector._is_relevant_path(file_path):
                paths.append((tag_name, file_path))

        return paths

    @staticmethod
    def _extract_bash_paths(command_content: str) -> List[Tuple[str, str]]:
        """Extract file paths from bash command content - improved approach."""
        paths = []
        args = command_content.split()

        for arg in args:
            arg = arg.strip('\'"')  # Remove quotes

            # Skip if it's a flag/option
            if arg.startswith('-'):
                continue

            # Skip common glob patterns that aren't specific file paths
            if arg in ['*/', '*', '**', '*.txt', '*.py', '*.js', '**/*']:
                continue

            # Skip if it's just a glob pattern
            if arg.endswith('/*') or arg.endswith('*'):
                continue

            # Only consider it a path if:
            # 1. It contains / and doesn't look like a glob pattern
            # 2. It starts with ./ or ~/
            # 3. It's an absolute path
            if (('/' in arg and not any(glob_char in arg for glob_char in ['*', '?', '[', ']'])) or
                arg.startswith('./') or arg.startswith('~/') or arg.startswith('/')):
                abs_path = os.path.abspath(arg)
                paths.append(("bash_path", abs_path))

        return paths

    @staticmethod
    def _extract_python_paths(command_content: str) -> List[Tuple[str, str]]:
        """Extract file paths from python command content - simple approach."""
        paths = []

        # Look for quoted strings that contain /
        import re
        matches = re.findall(r'["\']([^"\']*[/\\][^"\']*)["\']', command_content)
        for match in matches:
            abs_path = os.path.abspath(match)
            paths.append(("python_path", abs_path))

        return paths

    @staticmethod
    def get_tool_action_description(tool_name: str, xml_content: str, paths: List[Tuple[str, str]]) -> str:
        """
        Generate a human-readable description of what the tool will do.

        Args:
            tool_name: Name of the tool
            xml_content: Full XML content
            paths: List of (tag_name, file_path) tuples

        Returns:
            Human-readable action description
        """
        if not paths:
            # For bash/python, try to show the actual command being executed
            if tool_name in ["bash", "python"]:
                # Extract the command content from XML
                content = xml_content.strip()
                # Remove the outer tool tags
                inner_content = re.sub(rf'^<{tool_name}[^>]*>|</{tool_name}>$', '', content, flags=re.MULTILINE).strip()
                if inner_content:
                    # Truncate long commands and show first line
                    first_line = inner_content.split('\n')[0]
                    if len(first_line) > 60:
                        first_line = first_line[:57] + "..."
                    return f"Execute {tool_name}: {first_line}"
            return f"Execute {tool_name} (no file paths detected)"

        # Tool-specific action descriptions
        if tool_name == "editfile":
            # Check for line-specific editing
            line_match = re.search(r'<line>(.*?)</line>', xml_content, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', xml_content, re.DOTALL)

            if line_match and content_match:
                line_num = line_match.group(1).strip()
                new_content = content_match.group(1)
                content_preview = new_content[:40] + "..." if len(new_content) > 40 else new_content
                return f"Edit line {line_num} → '{content_preview}'"
            else:
                # Traditional find/replace
                find_match = re.search(r'<find>(.*?)</find>', xml_content, re.DOTALL)
                replace_match = re.search(r'<replace>(.*?)</replace>', xml_content, re.DOTALL)

                find_preview = find_match.group(1)[:40] + "..." if find_match and len(find_match.group(1)) > 40 else find_match.group(1) if find_match else "?"
                replace_preview = replace_match.group(1)[:40] + "..." if replace_match and len(replace_match.group(1)) > 40 else replace_match.group(1) if replace_match else "?"

                return f"Replace '{find_preview}' with '{replace_preview}'"

        elif tool_name == "writefile":
            return "Write/overwrite file content"

        elif tool_name == "readfile":
            # Check for line-specific reading
            lines_match = re.search(r'<lines>(.*?)</lines>', xml_content, re.DOTALL)
            if lines_match:
                line_spec = lines_match.group(1).strip()
                return f"Read lines {line_spec}"
            else:
                return "Read file content"

        elif tool_name == "insertline":
            content_match = re.search(r'<content>(.*?)</content>', xml_content, re.DOTALL)
            after_match = re.search(r'<after>(.*?)</after>', xml_content, re.DOTALL)
            before_match = re.search(r'<before>(.*?)</before>', xml_content, re.DOTALL)

            content_preview = content_match.group(1)[:40] + "..." if content_match and len(content_match.group(1)) > 40 else content_match.group(1) if content_match else "?"

            if after_match:
                return f"Insert after line {after_match.group(1).strip()} → '{content_preview}'"
            elif before_match:
                return f"Insert before line {before_match.group(1).strip()} → '{content_preview}'"
            else:
                return f"Insert at end → '{content_preview}'"

        elif tool_name == "deletelines":
            line_match = re.search(r'<line>(.*?)</line>', xml_content, re.DOTALL)
            start_match = re.search(r'<start>(.*?)</start>', xml_content, re.DOTALL)
            end_match = re.search(r'<end>(.*?)</end>', xml_content, re.DOTALL)

            if line_match:
                return f"Delete line {line_match.group(1).strip()}"
            elif start_match and end_match:
                return f"Delete lines {start_match.group(1).strip()}-{end_match.group(1).strip()}"
            else:
                return "Delete lines"

        elif tool_name == "bash":
            # Check if bash commands involve file operations
            bash_content = xml_content.lower()
            if any(cmd in bash_content for cmd in ['rm ', 'delete', 'mv ', 'cp ', 'chmod', 'chown']):
                return "Execute potentially file-modifying bash commands"
            else:
                return "Execute bash commands"

        elif tool_name == "python":
            # Check for file operations in Python code
            python_content = xml_content.lower()
            if any(op in python_content for op in ['open(', 'write(', 'unlink(', 'remove(', 'rename(']):
                return "Execute Python code with potential file operations"
            else:
                return "Execute Python code"

        # Generic description for other tools
        return f"Execute {tool_name} on file(s)"

    @staticmethod
    def should_check_permissions(tool_name: str, paths: List[Tuple[str, str]]) -> bool:
        """
        Determine if permission checking should be applied based on tool and paths.

        Args:
            tool_name: Name of the tool
            paths: List of detected paths

        Returns:
            True if permissions should be checked
        """
        # Always check permissions for tools with detected paths
        if paths:
            return True

        # Check permissions for potentially destructive tools even without detected paths
        destructive_tools = ["bash", "python", "writefile", "editfile"]
        return tool_name in destructive_tools

    @staticmethod
    def contains_sensitive_keywords(content: str) -> Tuple[bool, List[str]]:
        """
        Check if content contains any sensitive keywords that should block execution.

        Args:
            content: The content to check (file paths, command text, etc.)

        Returns:
            Tuple of (has_sensitive_content, list_of_found_keywords)
        """
        content_lower = content.lower()
        found_keywords = []

        for keyword in PathDetector.SENSITIVE_KEYWORDS:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)

        return len(found_keywords) > 0, found_keywords

    @staticmethod
    def check_sensitive_content_in_xml(xml_content: str) -> Tuple[bool, List[str]]:
        """
        Check XML content for sensitive keywords in paths, commands, or content.

        Args:
            xml_content: XML content to analyze

        Returns:
            Tuple of (has_sensitive_content, list_of_found_keywords)
        """
        all_found_keywords = []

        # Check the entire XML content
        has_sensitive, keywords = PathDetector.contains_sensitive_keywords(xml_content)
        if has_sensitive:
            all_found_keywords.extend(keywords)

        return len(all_found_keywords) > 0, all_found_keywords

    @staticmethod
    def get_primary_path(paths: List[Tuple[str, str]]) -> Optional[str]:
        """
        Get the primary file path from a list of detected paths.
        Prioritizes user-relevant files over system/cache files.

        Args:
            paths: List of (tag_name, file_path) tuples

        Returns:
            Primary file path or None if no paths found
        """
        if not paths:
            return None

        # First filter to only relevant paths
        relevant_paths = [(tag, path) for tag, path in paths if PathDetector._is_relevant_path(path)]

        if not relevant_paths:
            # If no relevant paths found, fall back to original logic but still filter basic irrelevant ones
            relevant_paths = [(tag, path) for tag, path in paths if not path.lower().endswith('.pyc')]
            if not relevant_paths:
                relevant_paths = paths

        # Score paths by relevance and tag priority
        scored_paths = []
        priority_tags = ['filepath', 'path', 'file', 'filename']

        for tag_name, file_path in relevant_paths:
            score = PathDetector._get_path_relevance_score(file_path)

            # Boost score for priority tags
            if any(priority_tag in tag_name.lower() for priority_tag in priority_tags):
                score += 15

            scored_paths.append((score, file_path, tag_name))

        # Sort by score (highest first) and return the best path
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return scored_paths[0][1]

    @staticmethod
    def get_command_hash(xml_content: str) -> str:
        """
        Generate a hash/signature for the exact command content.
        This is used for exact command matching in permissions.
        """
        import hashlib
        # Normalize the content by removing extra whitespace
        normalized = re.sub(r'\s+', ' ', xml_content.strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def get_command_pattern(tool_name: str, xml_content: str) -> str:
        """
        Extract a readable pattern for the command that can be used for exact matching.

        Args:
            tool_name: Name of the tool
            xml_content: XML content of the command

        Returns:
            A pattern string representing this specific command
        """
        # For bash commands, extract the actual command
        if tool_name == "bash":
            # Remove XML tags and get the actual bash command
            clean_content = re.sub(r'<[^>]*>', '', xml_content).strip()
            # Truncate very long commands for display
            if len(clean_content) > 100:
                clean_content = clean_content[:97] + "..."
            return f"bash: {clean_content}"

        # For python commands, extract the python code
        elif tool_name == "python":
            clean_content = re.sub(r'<[^>]*>', '', xml_content).strip()
            # Get first meaningful line for pattern
            first_line = clean_content.split('\n')[0].strip()
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            return f"python: {first_line}"

        # For other tools, create more specific patterns
        elif tool_name == "editfile":
            # For editfile, show the find/replace operation
            find_match = re.search(r'<find>(.*?)</find>', xml_content, re.DOTALL)
            replace_match = re.search(r'<replace>(.*?)</replace>', xml_content, re.DOTALL)
            if find_match and replace_match:
                find_preview = find_match.group(1)[:30] + "..." if len(find_match.group(1)) > 30 else find_match.group(1)
                replace_preview = replace_match.group(1)[:30] + "..." if len(replace_match.group(1)) > 30 else replace_match.group(1)
                return f"{tool_name}: Replace '{find_preview}' with '{replace_preview}'"

        # For other tools without paths, use a content-based pattern
        else:
            clean_content = re.sub(r'<[^>]*>', '', xml_content).strip()
            if len(clean_content) > 60:
                clean_content = clean_content[:57] + "..."
            return f"{tool_name}: {clean_content}"

    @staticmethod
    def analyze_tool_block(tool_name: str, xml_content: str) -> Dict[str, Any]:
        """
        Analyze a tool block and extract relevant information for permission checking.

        Args:
            tool_name: Name of the tool
            xml_content: XML content of the tool block

        Returns:
            Dictionary with analysis results
        """
        paths = PathDetector.extract_paths_from_xml(xml_content, tool_name)
        primary_path = PathDetector.get_primary_path(paths)
        action_description = PathDetector.get_tool_action_description(tool_name, xml_content, paths)
        needs_permission = PathDetector.should_check_permissions(tool_name, paths)
        command_hash = PathDetector.get_command_hash(xml_content)
        command_pattern = PathDetector.get_command_pattern(tool_name, xml_content)

        return {
            'tool_name': tool_name,
            'paths': paths,
            'primary_path': primary_path,
            'action_description': action_description,
            'needs_permission': needs_permission,
            'xml_content': xml_content,
            'command_hash': command_hash,
            'command_pattern': command_pattern,
            'has_paths': len(paths) > 0
        }