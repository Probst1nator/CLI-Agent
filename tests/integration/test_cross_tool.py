"""
Cross-Tool Integration Tests

Tests that verify tools work together correctly and can share
infrastructure and resources appropriately.
"""

import sys
import os
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

from shared.path_resolver import setup_cli_agent_imports
setup_cli_agent_imports()


def test_all_tools_can_import_core():
    """Test that all tools can import core infrastructure"""
    try:
        from core import LlmRouter, Chat, Role, AIStrengths, g
        return True
    except ImportError:
        return False


def test_all_tools_can_import_shared():
    """Test that all tools can import shared utilities"""
    try:
        from shared import get_dia_model, AIFixPath  # extract_blocks might not be available
        return True
    except ImportError:
        return False


def test_path_resolver_consistency():
    """Test that PathResolver gives consistent results"""
    from shared.path_resolver import PathResolver
    
    root1 = PathResolver.find_cli_agent_root()
    root2 = PathResolver.find_cli_agent_root()
    
    return root1 == root2 and root1.exists()


if __name__ == "__main__":
    tests = [
        test_all_tools_can_import_core,
        test_all_tools_can_import_shared,
        test_path_resolver_consistency
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
    
    print(f"\nIntegration Tests: {sum(results)}/{len(results)} passed")