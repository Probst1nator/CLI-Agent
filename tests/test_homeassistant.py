#!/usr/bin/env python3
"""
Comprehensive test suite for the HomeAssistant utility.
Tests all functionality with mocked Home Assistant API responses.
"""

import json
import os

# Mock the dependencies before importing
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock the missing dependencies
sys.modules['agent.utils_manager.util_base'] = MagicMock()
sys.modules['core.globals'] = MagicMock()

# Create mock globals
mock_globals = MagicMock()
mock_globals.CLIAGENT_PERSISTENT_STORAGE_PATH = tempfile.gettempdir()

# Mock more dependencies to avoid import errors
mock_modules = {
    'agent.utils_manager.util_base': MagicMock(),
    'core.globals': MagicMock(),
    'tools.searchweb': MagicMock(),
    'tools': MagicMock(),
    'py_classes': MagicMock(),
}

# Import after mocking
with patch.dict('sys.modules', mock_modules):
    # Import the module directly to avoid __init__.py issues
    import importlib.util
    spec = importlib.util.spec_from_file_location("homeassistant",
                                                 project_root / "utils" / "homeassistant.py")
    homeassistant_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(homeassistant_module)

    HomeAssistant = homeassistant_module.HomeAssistant

# Patch the globals reference
HomeAssistant._INTERACTED_ENTITIES_FILE = os.path.join(tempfile.gettempdir(), 'test_homeassistant_entities.json')
HomeAssistant._AUTOMATION_BACKUP_DIR = os.path.join(tempfile.gettempdir(), 'test_homeassistant_backups')


class TestHomeAssistant(unittest.TestCase):
    """Test suite for HomeAssistant utility class."""

    def setUp(self):
        """Set up test environment."""
        self.test_entities_file = HomeAssistant._INTERACTED_ENTITIES_FILE
        self.test_backup_dir = HomeAssistant._AUTOMATION_BACKUP_DIR

        # Clean up any existing test files
        if os.path.exists(self.test_entities_file):
            os.remove(self.test_entities_file)
        if os.path.exists(self.test_backup_dir):
            import shutil
            shutil.rmtree(self.test_backup_dir)

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_entities_file):
            os.remove(self.test_entities_file)
        if os.path.exists(self.test_backup_dir):
            import shutil
            shutil.rmtree(self.test_backup_dir)

    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_get_config(self):
        """Test configuration retrieval."""
        url, token = HomeAssistant._get_config()
        self.assertEqual(url, 'http://test:8123')
        self.assertEqual(token, 'test-token')

    def test_get_config_missing_vars(self):
        """Test configuration with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                HomeAssistant._get_config()
            self.assertIn("HASS_URL and HASS_TOKEN", str(context.exception))

    def test_load_save_interacted_entities(self):
        """Test loading and saving entity interaction history."""
        # Test with empty file
        entities = HomeAssistant._load_interacted_entities()
        self.assertEqual(entities, {})

        # Test saving and loading
        test_entities = {
            'light.test': [
                {
                    'timestamp': '2024-01-01T10:00:00',
                    'action': 'service_call',
                    'service': 'light.turn_on',
                    'details': 'Called light.turn_on'
                }
            ]
        }

        HomeAssistant._save_interacted_entities(test_entities)
        loaded_entities = HomeAssistant._load_interacted_entities()

        self.assertIn('light.test', loaded_entities)
        self.assertEqual(len(loaded_entities['light.test']), 1)
        self.assertEqual(loaded_entities['light.test'][0]['action'], 'service_call')

    def test_track_entity_interaction(self):
        """Test entity interaction tracking."""
        entity_id = 'light.living_room'

        # Track a service call
        HomeAssistant._track_entity_interaction(
            entity_id, 'service_call', 'light', 'turn_on'
        )

        entities = HomeAssistant._load_interacted_entities()
        self.assertIn(entity_id, entities)
        self.assertEqual(len(entities[entity_id]), 1)
        self.assertEqual(entities[entity_id][0]['action'], 'service_call')
        self.assertEqual(entities[entity_id][0]['service'], 'light.turn_on')

    @patch('requests.post')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_call_service_success(self, mock_post):
        """Test successful service call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '[]'
        mock_post.return_value = mock_response

        result = HomeAssistant._call_service('light', 'turn_on', 'light.living_room')
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')
        self.assertEqual(result_data['result']['service'], 'light.turn_on')

    @patch('requests.post')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_call_service_failure(self, mock_post):
        """Test failed service call."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_post.return_value = mock_response

        result = HomeAssistant._call_service('invalid', 'service', 'entity')
        result_data = json.loads(result)

        self.assertIn('error', result_data)
        self.assertIn('Failed to call service', result_data['error'])

    @patch('requests.get')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_get_state_single_entity(self, mock_get):
        """Test getting state of a single entity."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'entity_id': 'light.living_room',
            'state': 'on',
            'attributes': {'friendly_name': 'Living Room Light'}
        }
        mock_get.return_value = mock_response

        result = HomeAssistant._get_state('light.living_room')
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')
        self.assertEqual(result_data['result']['entity_id'], 'light.living_room')

    @patch('requests.get')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_get_state_all_entities(self, mock_get):
        """Test getting all entity states."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'entity_id': 'light.living_room',
                'state': 'on',
                'attributes': {'friendly_name': 'Living Room Light'}
            },
            {
                'entity_id': 'sensor.temperature',
                'state': '22.5',
                'attributes': {'friendly_name': 'Temperature Sensor'}
            }
        ]
        mock_get.return_value = mock_response

        result = HomeAssistant._get_state()
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')
        self.assertEqual(result_data['result']['entity_count'], 2)

    @patch('requests.get')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_search_entities_by_keyword(self, mock_get):
        """Test searching entities by keyword."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'entity_id': 'light.living_room',
                'state': 'on',
                'attributes': {'friendly_name': 'Living Room Light'}
            },
            {
                'entity_id': 'light.bedroom',
                'state': 'off',
                'attributes': {'friendly_name': 'Bedroom Light'}
            },
            {
                'entity_id': 'sensor.temperature',
                'state': '22.5',
                'attributes': {'friendly_name': 'Temperature Sensor'}
            }
        ]
        mock_get.return_value = mock_response

        result = HomeAssistant._search_entities_by_keyword('light')
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')
        self.assertEqual(result_data['result']['matches_found'], 2)
        self.assertIn('light.living_room', result_data['result']['matches'])

    def test_get_entity_usage_history_empty(self):
        """Test getting usage history for entity with no history."""
        result = HomeAssistant._get_entity_usage_history('light.nonexistent')
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'No history found')

    def test_get_entity_usage_history_with_data(self):
        """Test getting usage history for entity with history."""
        # Add some history
        HomeAssistant._track_entity_interaction('light.test', 'service_call', 'light', 'turn_on')

        result = HomeAssistant._get_entity_usage_history('light.test')
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')
        self.assertEqual(result_data['result']['history_entries'], 1)

    @patch('requests.get')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_manage_automations_list(self, mock_get):
        """Test listing automations."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'id': 'test_automation',
                'alias': 'Test Automation',
                'description': 'A test automation',
                'mode': 'single',
                'trigger': [{'platform': 'time'}],
                'action': [{'service': 'light.turn_on'}]
            }
        ]
        mock_get.return_value = mock_response

        result = HomeAssistant._manage_automations('list')
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')
        self.assertEqual(result_data['result']['automation_count'], 1)

    @patch('requests.post')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_manage_automations_create(self, mock_post):
        """Test creating an automation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.text = 'Created'
        mock_post.return_value = mock_response

        automation_config = {
            'id': 'test_automation',
            'alias': 'Test Automation',
            'trigger': [],
            'action': []
        }

        result = HomeAssistant._manage_automations('create', automation_config=automation_config)
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')

    @patch('requests.post')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_control_automation_enable(self, mock_post):
        """Test enabling an automation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'Success'
        mock_post.return_value = mock_response

        result = HomeAssistant._control_automation('enable', 'automation.test')
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')
        self.assertEqual(result_data['result']['action'], 'enable')

    @patch('requests.get')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_backup_automations(self, mock_get):
        """Test backing up automations."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'id': 'test_automation',
                'alias': 'Test Automation',
                'trigger': [],
                'action': []
            }
        ]
        mock_get.return_value = mock_response

        result = HomeAssistant._backup_restore_automations('backup', 'test_backup')
        result_data = json.loads(result)

        self.assertIn('result', result_data)
        self.assertEqual(result_data['result']['status'], 'Success')
        self.assertEqual(result_data['result']['backup_name'], 'test_backup')

    def test_run_logic_invalid_action(self):
        """Test run_logic with invalid action."""
        result = HomeAssistant._run_logic('invalid_action')
        result_data = json.loads(result)

        self.assertIn('error', result_data)
        self.assertIn('Invalid action', result_data['error'])

    def test_run_logic_call_service_missing_params(self):
        """Test run_logic call_service with missing parameters."""
        result = HomeAssistant._run_logic('call_service')
        result_data = json.loads(result)

        self.assertIn('error', result_data)
        self.assertIn('domain', result_data['error'])
        self.assertIn('service', result_data['error'])

    def test_run_logic_get_state_missing_entity(self):
        """Test run_logic get_state with missing entity_id."""
        result = HomeAssistant._run_logic('get_state')
        result_data = json.loads(result)

        self.assertIn('error', result_data)
        self.assertIn('entity_id', result_data['error'])

    def test_module_level_run_function(self):
        """Test module-level run function wrapper."""
        run_func = homeassistant_module.run

        result = run_func('invalid_action')
        result_data = json.loads(result)

        self.assertIn('error', result_data)
        self.assertIn('Invalid action', result_data['error'])

    @patch('requests.post')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_connection_error_handling(self, mock_post):
        """Test connection error handling."""
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError("Connection refused")

        result = HomeAssistant._call_service('light', 'turn_on', 'light.test')
        result_data = json.loads(result)

        self.assertIn('error', result_data)
        self.assertIn('Cannot connect to Home Assistant', result_data['error'])

    @patch('requests.post')
    @patch.dict(os.environ, {'HASS_URL': 'http://test:8123', 'HASS_TOKEN': 'test-token'})
    def test_timeout_error_handling(self, mock_post):
        """Test timeout error handling."""
        from requests.exceptions import Timeout
        mock_post.side_effect = Timeout("Request timed out")

        result = HomeAssistant._call_service('light', 'turn_on', 'light.test')
        result_data = json.loads(result)

        self.assertIn('error', result_data)
        self.assertIn('timed out', result_data['error'])


class TestHomeAssistantIntegration(unittest.TestCase):
    """Integration tests for HomeAssistant utility (requires actual HA instance)."""

    def setUp(self):
        """Check if integration tests should run."""
        self.skip_integration = not (os.environ.get('HASS_URL') and os.environ.get('HASS_TOKEN'))
        if self.skip_integration:
            self.skipTest("Integration tests require HASS_URL and HASS_TOKEN environment variables")

    def test_list_actual_devices(self):
        """Test listing devices from actual Home Assistant instance."""
        if self.skip_integration:
            return

        result = HomeAssistant._run_logic('list_devices')
        result_data = json.loads(result)

        # Should either succeed or fail with a clear error
        self.assertTrue('result' in result_data or 'error' in result_data)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHomeAssistant))
    suite.addTests(loader.loadTestsFromTestCase(TestHomeAssistantIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running HomeAssistant utility tests...")
    success = run_tests()
    exit(0 if success else 1)