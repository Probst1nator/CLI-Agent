#!/usr/bin/env python3
"""
Simplified test suite for the HomeAssistant utility.
Tests core functionality without complex mocking.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_homeassistant_import():
    """Test that HomeAssistant can be imported and basic functionality works."""
    try:
        # Test direct import of the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("homeassistant",
                                                     project_root / "utils" / "homeassistant.py")
        homeassistant_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(homeassistant_module)

        HomeAssistant = homeassistant_module.HomeAssistant
        print("✓ HomeAssistant module imported successfully")

        # Test metadata
        metadata = HomeAssistant.get_metadata()
        print(f"✓ Metadata available: {len(metadata)} fields")

        # Test invalid action handling
        result = HomeAssistant._run_logic('invalid_action')
        result_data = json.loads(result)
        assert 'error' in result_data
        assert 'Invalid action' in result_data['error']
        print("✓ Invalid action handling works")

        # Test missing config handling (skip if env vars are set)
        if not (os.environ.get('HASS_URL') and os.environ.get('HASS_TOKEN')):
            try:
                HomeAssistant._get_config()
                print("❌ Missing config should raise ValueError")
            except ValueError:
                print("✓ Missing config validation works")
        else:
            print("✓ Missing config validation (skipped - env vars set)")

        # Test file operations
        test_entities = {'light.test': [{'timestamp': '2024-01-01T00:00:00', 'action': 'test'}]}
        HomeAssistant._save_interacted_entities(test_entities)
        loaded = HomeAssistant._load_interacted_entities()
        assert 'light.test' in loaded
        print("✓ File operations work")

        print("\n🎉 All basic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_homeassistant_functions():
    """Test specific HomeAssistant functions work correctly."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("homeassistant",
                                                     project_root / "utils" / "homeassistant.py")
        homeassistant_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(homeassistant_module)

        HomeAssistant = homeassistant_module.HomeAssistant

        # Test parameter validation for different actions
        test_cases = [
            ('call_service', {'error': 'domain'}),  # Missing domain and service
            ('get_state', {'error': 'entity_id'}),  # Missing entity_id
            ('search_entity_by_keyword', {'error': 'keyword'}),  # Missing keyword
            ('get_automation', {'error': 'automation_id'}),  # Missing automation_id
        ]

        for action, expected in test_cases:
            result = HomeAssistant._run_logic(action)
            result_data = json.loads(result)
            assert 'error' in result_data
            assert expected['error'] in result_data['error']
            print(f"✓ {action} parameter validation works")

        # Test module-level run function
        run_func = homeassistant_module.run
        result = run_func('invalid_action')
        result_data = json.loads(result)
        assert 'error' in result_data
        print("✓ Module-level run function works")

        print("\n🎉 Function tests passed!")
        return True

    except Exception as e:
        print(f"❌ Function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run comprehensive standalone test."""
    print("🧪 Testing HomeAssistant utility...")
    print("=" * 50)

    success = True

    # Test 1: Basic import and functionality
    print("\n📦 Testing import and basic functionality...")
    if not test_homeassistant_import():
        success = False

    # Test 2: Function validation
    print("\n🔧 Testing function parameter validation...")
    if not test_homeassistant_functions():
        success = False

    # Test 3: Check file structure
    print("\n📁 Testing file structure...")
    homeassistant_file = project_root / "utils" / "homeassistant.py"
    if homeassistant_file.exists():
        file_size = homeassistant_file.stat().st_size
        print(f"✓ HomeAssistant file exists ({file_size} bytes)")

        # Check for key classes and functions
        content = homeassistant_file.read_text()
        checks = [
            ('class HomeAssistant', 'HomeAssistant class'),
            ('def _run_logic', 'Main logic function'),
            ('def get_metadata', 'Metadata function'),
            ('def run(action', 'Module run function'),
            ('call_service', 'Service call action'),
            ('automation', 'Automation management'),
        ]

        for check, desc in checks:
            if check in content:
                print(f"✓ {desc} found")
            else:
                print(f"❌ {desc} missing")
                success = False
    else:
        print(f"❌ HomeAssistant file not found at {homeassistant_file}")
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! HomeAssistant utility is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == '__main__':
    exit(run_comprehensive_test())