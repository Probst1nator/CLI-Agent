from typing import Optional
"""
Minimal dev tool hints for CLI-Agent permission system.
Provides helpful suggestions when common development tools are detected.
NOT used for automatic permission granting - only for user guidance.
"""

# Common development tools by category
DEV_TOOL_PATTERNS = {
    'linting': ['ruff', 'flake8', 'pylint', 'black', 'isort', 'mypy', 'eslint', 'prettier'],
    'testing': ['pytest', 'unittest', 'nose', 'tox', 'jest', 'mocha', 'vitest'],
    'building': ['npm', 'yarn', 'pip', 'poetry', 'cargo', 'make', 'gradle', 'maven'],
    'git': ['git add', 'git commit', 'git push', 'git pull', 'git status', 'git diff'],
    'search': ['grep', 'rg', 'ripgrep', 'find', 'ack', 'ag'],
}


def get_tool_hint(command: Optional[str]) -> Optional[str]:
    """
    Get a helpful hint if the command matches a known dev tool category.

    This is ONLY used for displaying helpful suggestions to users,
    NOT for automatic permission detection.

    Args:
        command: Command string to analyze

    Returns:
        Helpful hint string or None if no match

    Examples:
        >>> get_tool_hint("ruff check .")
        "This looks like a linting tool. Consider adding: bash(ruff:*)"

        >>> get_tool_hint("pytest tests/")
        "This looks like a testing tool. Consider adding: bash(pytest:*)"
    """
    if not command:
        return None

    command_lower = command.lower().strip()

    for category, patterns in DEV_TOOL_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in command_lower:
                # Extract first word for suggestion
                first_word = command_lower.split()[0]
                return _get_hint_for_category(category, first_word)

    return None


def _get_hint_for_category(category: str, command_word: str) -> str:
    """Generate category-specific hint message."""
    hints = {
        'linting': f"This looks like a linting tool. Consider adding: bash({command_word}:*)",
        'testing': f"This looks like a testing tool. Consider adding: bash({command_word}:*)",
        'building': f"This looks like a build tool. Consider adding: bash({command_word}:*)",
        'git': "This is a git command. Consider adding: bash(git:*) for all git operations",
        'search': f"This is a search tool. Consider adding: bash({command_word}:*)",
    }
    return hints.get(category, f"Consider adding: bash({command_word}:*)")


def get_suggested_rules(tool_name: str, command: Optional[str],
                       file_path: Optional[str]) -> list[str]:
    """
    Get suggested permission rules based on tool usage.

    Args:
        tool_name: Tool being executed
        command: Command string if bash
        file_path: File path if applicable

    Returns:
        List of suggested permission rules

    Examples:
        >>> get_suggested_rules("bash", "npm install", None)
        ["bash(npm:*)", "bash(npm:*)  # All npm commands"]

        >>> get_suggested_rules("readfile", None, "./src/main.py")
        ["readfile(./src/**)", "readfile(**)  # All files"]
    """
    suggestions = []

    if tool_name == "bash" and command:
        # Suggest command-specific and wildcard rules
        cmd_word = command.split()[0]
        suggestions.append(f"bash({cmd_word}:*)")

        # Add hint if it's a known dev tool
        hint = get_tool_hint(command)
        if hint:
            suggestions.append(f"# {hint}")

    elif file_path:
        import os
        # Suggest directory pattern
        directory = os.path.dirname(file_path)
        if directory:
            suggestions.append(f"{tool_name}({directory}/**)")
        suggestions.append(f"{tool_name}(**)  # All files")

    else:
        # Suggest allowing all uses of this tool
        suggestions.append(tool_name)

    return suggestions
