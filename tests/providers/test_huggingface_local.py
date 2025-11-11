"""
Tests for HuggingFace Local Provider
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.chat import Chat, Role
from core.providers.cls_huggingface_local_interface import HuggingFaceLocalClient
from core.providers.huggingface_local_models import (
    SUPPORTED_MODELS,
    add_custom_model,
    get_model_info,
    get_model_key_from_id,
    get_model_keys,
)

# ============================================================================
# Test Configuration
# ============================================================================
# Model to test (change this to test different models)
TEST_MODEL_ID = "moonshotai/Moonlight-16B-A3B-Instruct"
TEST_MODEL_KEY = get_model_key_from_id(TEST_MODEL_ID)


class TestHuggingFaceLocalModels(unittest.TestCase):
    """Test model registry functionality."""

    def test_supported_models_exist(self):
        """Test that we have supported models defined."""
        self.assertGreater(len(SUPPORTED_MODELS), 0)
        self.assertIn(TEST_MODEL_KEY, get_model_keys())

    def test_get_model_info(self):
        """Test getting model information."""
        # Test by model_key
        model = get_model_info(TEST_MODEL_KEY)
        self.assertIsNotNone(model)
        self.assertEqual(model["model_id"], TEST_MODEL_ID)
        self.assertEqual(model["context_window"], 8000)
        self.assertTrue(model["trust_remote_code"])

        # Test getting all models
        all_models = get_model_info()
        self.assertIsInstance(all_models, list)
        self.assertGreater(len(all_models), 0)

        # Test non-existent model
        self.assertIsNone(get_model_info("nonexistent-model"))

    def test_add_custom_model(self):
        """Test adding custom models at runtime."""
        result = add_custom_model(
            model_id="test/model",
            context_window=4096,
            description="Test model"
        )
        self.assertTrue(result)

        # Verify it was added (key derived from ID)
        model_info = get_model_info("model")
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info["model_id"], "test/model")
        self.assertEqual(model_info["model_key"], "model")

        # Try adding duplicate
        result2 = add_custom_model(
            model_id="test/model",
            context_window=4096
        )
        self.assertFalse(result2)


class TestHuggingFaceLocalClient(unittest.TestCase):
    """Test HuggingFace Local Client functionality."""

    def test_models_dir_creation(self):
        """Test that models directory is created."""
        models_dir = HuggingFaceLocalClient.get_models_dir()
        self.assertIsInstance(models_dir, Path)
        # Directory should exist after calling get_models_dir()
        self.assertTrue(models_dir.exists())

    def test_is_model_downloaded(self):
        """Test checking if model is downloaded."""
        # Non-existent model should return False
        result = HuggingFaceLocalClient.is_model_downloaded("nonexistent/model")
        self.assertFalse(result)

    def test_chat_message_conversion(self):
        """Test conversion of Chat messages to model format."""
        # Create a test chat
        chat = Chat()
        chat.add_message(Role.SYSTEM, "You are a helpful assistant.")
        chat.add_message(Role.USER, "What is 2+2?")

        # Verify messages are properly formatted
        self.assertEqual(len(chat.messages), 2)
        self.assertEqual(chat.messages[0].role, Role.SYSTEM)
        self.assertEqual(chat.messages[0].content, "You are a helpful assistant.")
        self.assertEqual(chat.messages[1].role, Role.USER)
        self.assertEqual(chat.messages[1].content, "What is 2+2?")

        # Test role enum values match expected format
        self.assertEqual(Role.SYSTEM.value, "system")
        self.assertEqual(Role.USER.value, "user")
        self.assertEqual(Role.ASSISTANT.value, "assistant")


class TestMoonlightModelSetup(unittest.TestCase):
    """Test model setup for configured test model."""

    def test_model_config(self):
        """Verify test model configuration is correct."""
        model = get_model_info(TEST_MODEL_KEY)

        # Verify key properties
        self.assertEqual(model["model_id"], TEST_MODEL_ID)
        self.assertEqual(model["model_key"], TEST_MODEL_KEY)
        self.assertEqual(model["context_window"], 8000)
        self.assertTrue(model["trust_remote_code"], f"{TEST_MODEL_KEY} requires trust_remote_code=True")
        self.assertFalse(model["requires_auth"], f"{TEST_MODEL_KEY} should not require authentication")

        # Verify capabilities
        from core.lm_capabilities import LMCapabilities
        self.assertIn(LMCapabilities.LOCAL, model["capabilities"])

    def test_model_in_registry(self):
        """Verify test model is registered in the model list."""
        model_keys = get_model_keys()
        self.assertIn(TEST_MODEL_KEY, model_keys)

    @patch('core.providers.cls_huggingface_local_interface.HuggingFaceLocalClient.load_model')
    @patch('core.providers.cls_huggingface_local_interface.HuggingFaceLocalClient.get_valid_model')
    def test_model_generation_setup(self, mock_get_valid_model, mock_load_model):
        """Test that generation is set up correctly for test model."""
        # Mock the model loading
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value=MagicMock())

        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_get_valid_model.return_value = (Path("/fake/path"), TEST_MODEL_ID)

        # Create a simple chat
        chat = Chat()
        chat.add_message(Role.USER, "Test message")

        # Attempt generation setup (will fail at actual generation, but we can test setup)
        try:
            HuggingFaceLocalClient.generate_response(
                chat,
                model_key=TEST_MODEL_KEY,
                temperature=0.7
            )
        except Exception:
            # Expected to fail since we're mocking, but setup should have run
            pass

        # Verify load_model was called with correct model_id
        mock_load_model.assert_called()
        call_args = mock_load_model.call_args
        self.assertEqual(call_args[0][0], TEST_MODEL_ID)


def run_integration_test():
    """
    Integration test that actually downloads and tests the configured model.
    WARNING: This will download large model files!
    Only run if you have enough disk space and bandwidth.
    """
    print("\n" + "="*80)
    print(f"INTEGRATION TEST - {TEST_MODEL_KEY}")
    print("="*80)
    print(f"\nWARNING: This test will download the {TEST_MODEL_ID} model")
    print("Size: ~32GB")
    print("Location: ~/.cli-agent/models/")
    print("\nTo run this test, execute: python tests/providers/test_huggingface_local.py --integration")
    print("="*80 + "\n")

    # Check if we should run integration test
    if "--integration" not in sys.argv:
        print("Skipping integration test. Use --integration flag to run.")
        return

    print("\nStarting integration test...")

    try:
        # Create a test chat
        chat = Chat()
        chat.add_message(Role.SYSTEM, "You are a helpful assistant.")
        chat.add_message(Role.USER, "What is 2+2? Answer briefly.")

        print(f"\nGenerating response with {TEST_MODEL_KEY}...")
        print("(First run will download the model - this may take a while)\n")

        # Generate response
        stream = HuggingFaceLocalClient.generate_response(
            chat,
            model_key=TEST_MODEL_KEY,
            temperature=0.7
        )

        # Collect response
        response = ""
        for chunk in stream:
            if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                token = chunk.message.content
                response += token
                print(token, end="", flush=True)

        print("\n\n" + "="*80)
        print(f"SUCCESS! {TEST_MODEL_KEY} generated response:")
        print("-"*80)
        print(response)
        print("="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n\nERROR: Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestHuggingFaceLocalModels))
    suite.addTests(loader.loadTestsFromTestCase(TestHuggingFaceLocalClient))
    suite.addTests(loader.loadTestsFromTestCase(TestMoonlightModelSetup))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run integration test if requested
    if "--integration" in sys.argv:
        print("\n" + "="*80)
        integration_result = run_integration_test()
        if not integration_result:
            sys.exit(1)

    # Exit with error code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)

    print("\n✅ All tests passed!")
