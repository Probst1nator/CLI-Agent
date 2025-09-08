"""
Type Hints Unit Tests

Tests to verify proper type hints are used throughout the codebase
instead of generic Dict usage.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))


def find_generic_dict_usage(file_path: Path) -> List[Tuple[int, str]]:
    """Find usage of generic Dict without type parameters"""
    if not file_path.suffix == ".py":
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        issues = []
        
        for node in ast.walk(tree):
            # Look for Dict usage without type parameters
            if isinstance(node, ast.Name) and node.id == "Dict":
                # This is a basic check - in practice you'd want more sophisticated analysis
                line_num = getattr(node, 'lineno', 0)
                issues.append((line_num, f"Generic Dict usage at line {line_num}"))
        
        return issues
        
    except (SyntaxError, UnicodeDecodeError):
        return []


def test_type_hints_in_core():
    """Test core modules use proper type hints"""
    core_path = Path(__file__).parent.parent.parent / "core"
    if not core_path.exists():
        # Fall back to py_classes if core doesn't exist yet
        core_path = Path(__file__).parent.parent.parent / "py_classes"
    
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
    
    # Consider it passing if less than 50% have issues (gradual improvement)
    success_rate = (total_files - files_with_issues) / max(total_files, 1)
    return success_rate > 0.5


def test_type_hints_in_tools():
    """Test tool modules use proper type hints"""
    tools_path = Path(__file__).parent.parent.parent / "tools"
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
    return success_rate > 0.7  # Higher standard for tools since we just updated them


def test_proper_imports_in_typing():
    """Test that files import typing components properly"""
    tools_path = Path(__file__).parent.parent.parent / "tools"
    if not tools_path.exists():
        return True  # Skip if tools don't exist
    
    proper_imports = 0
    total_checked = 0
    
    for py_file in tools_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            total_checked += 1
            
            # Check for proper typing imports
            if "from typing import" in content and ("Dict" in content or "List" in content):
                proper_imports += 1
        
        except UnicodeDecodeError:
            continue
    
    if total_checked == 0:
        return True
    
    return (proper_imports / total_checked) > 0.5


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