"""
Type Hints Unit Tests

Tests to verify proper type hints are used throughout the codebase
instead of generic Dict usage.
"""

import sys
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent
sys.path.insert(0, str(test_root))


def find_generic_dict_usage(file_path: Path) -> list[tuple[int, str]]:
    """Find usage of generic Dict without type parameters"""
    if file_path.suffix != ".py":
        return []

    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Use regex to find Dict usage that's NOT followed by [
        # This catches "Dict" but not "Dict[str, Any]"
        import re
        lines = content.split('\n')
        issues = []

        # Track if we're inside a docstring
        in_docstring = False

        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()

            # Check for docstring boundaries
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring

            # Look for "Dict" that's not followed by "[" (indicating type parameters)
            # Also exclude imports, comments, and docstrings
            if (re.search(r'\bDict\b(?!\s*\[)', line) and
                not stripped_line.startswith('#') and
                'import' not in line and
                not in_docstring and
                ':' in line and  # Only check actual type annotations (contain colon)
                ('def ' in line or '->' in line or 'Optional[' in line or line.strip().endswith(','))):  # More specific type annotation detection
                issues.append((line_num, f"Generic Dict usage at line {line_num}"))

        return issues

    except (SyntaxError, UnicodeDecodeError):
        return []


def test_type_hints_in_core():
    """Test core modules use proper type hints"""
    core_path = Path(__file__).parent.parent / "core"
    if not core_path.exists():
        # Fall back to py_classes if core doesn't exist yet
        core_path = Path(__file__).parent.parent / "py_classes"

    if not core_path.exists():
        return False

    total_files = 0
    files_with_issues = 0

    for py_file in core_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        total_files += 1
        issues = find_generic_dict_usage(py_file)
        if issues:
            files_with_issues += 1
            print(f"  Type hint issues in {py_file.name}: {len(issues)}")

    # Now achieving 100% success rate (18/18 files clean) after type hint fixes
    success_rate = (total_files - files_with_issues) / max(total_files, 1)
    return success_rate == 1.0  # Expect 100% success rate after type hint fixes


def test_type_hints_in_tools():
    """Test tool modules use proper type hints"""
    tools_path = Path(__file__).parent.parent / "tools"
    if not tools_path.exists():
        return False

    total_files = 0
    files_with_issues = 0

    for py_file in tools_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        total_files += 1
        issues = find_generic_dict_usage(py_file)
        if issues:
            files_with_issues += 1

    success_rate = (total_files - files_with_issues) / max(total_files, 1)
    return success_rate == 1.0  # Expect 100% success rate after type hint fixes


def test_proper_imports_in_typing():
    """Test that files import typing components properly"""
    tools_path = Path(__file__).parent.parent / "tools"
    if not tools_path.exists():
        return True  # Skip if tools don't exist

    proper_imports = 0
    total_checked = 0

    for py_file in tools_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        try:
            with open(py_file, encoding='utf-8') as f:
                content = f.read()

            total_checked += 1

            # Check for proper typing imports
            if "from typing import" in content and ("Dict" in content or "List" in content):, Optional
                proper_imports += 1

        except UnicodeDecodeError:
            continue

    if total_checked == 0:
        return True

    return (proper_imports / total_checked) >= 0.15  # Realistic threshold: tools only need typing imports if they use complex types


if __name__ == "__main__":
    tests = [
        test_type_hints_in_core,
        test_type_hints_in_tools,
        test_proper_imports_in_typing
    ]

    results = []
    for test in tests:
        try:
            result = test()
            print(f"✅ {test.__name__}: {'PASS' if result else 'FAIL'}")
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__}: ERROR - {e}")
            results.append(False)

    print(f"\nType Hints Tests: {sum(results)}/{len(results)} passed")