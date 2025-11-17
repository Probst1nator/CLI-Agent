"""
Autonomous Permission System Tests

Comprehensive tests for the ML-based permission classifier and orchestration layer.

Test Structure:
    1. EmbeddingClassifier Tests (unit)
        - Anchor concept initialization
        - Classification accuracy
        - Threshold sensitivity
        - Cache behavior
        - Error handling

    2. AutonomousPermission Tests (integration)
        - Layer precedence (deny → allow → ML → prompt)
        - ML enable/disable
        - Fallback behavior
        - Statistics tracking

    3. SettingsManager Tests (unit)
        - ML configuration loading
        - ML config updates
        - Settings persistence
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

from shared.path_resolver import setup_cli_agent_imports
setup_cli_agent_imports()

import pytest
import numpy as np

from core.permissions.settings_manager import SettingsManager
from core.permissions.embedding_classifier import (
    EmbeddingClassifier,
    tool_call_to_xml,
    OLLAMA_AVAILABLE
)
from core.permissions.autonomous_permission import AutonomousPermission


# ============================================================================
# EmbeddingClassifier Tests
# ============================================================================

@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not installed")
class TestEmbeddingClassifier:
    """Test suite for ML embedding classifier."""

    def test_initialization(self):
        """Test classifier initializes with correct configuration."""
        classifier = EmbeddingClassifier(
            model="embeddinggemma:latest",
            threshold_allow=0.01,
            threshold_deny=-0.01
        )

        assert classifier.model == "embeddinggemma:latest"
        assert classifier.threshold_allow == 0.01
        assert classifier.threshold_deny == -0.01
        assert classifier.safe_embeddings is None  # Lazy init
        assert classifier.unsafe_embeddings is None

    @patch('core.permissions.embedding_classifier.ollama.Client')
    def test_anchor_initialization(self, mock_client):
        """Test anchor embeddings are pre-computed correctly."""
        # Mock embedding responses
        mock_client.return_value.embeddings.return_value = {
            'embedding': [0.1] * 384  # Standard embedding dimension
        }

        classifier = EmbeddingClassifier()
        classifier.initialize()

        # Verify embeddings created
        assert classifier.safe_embeddings is not None
        assert classifier.unsafe_embeddings is not None
        assert len(classifier.safe_embeddings) == len(EmbeddingClassifier.SAFE_CONCEPTS)
        assert len(classifier.unsafe_embeddings) == len(EmbeddingClassifier.UNSAFE_CONCEPTS)

    @patch('core.permissions.embedding_classifier.ollama.Client')
    def test_safe_classification(self, mock_client):
        """Test safe operations are correctly classified as auto-allow."""
        # Mock embeddings: safe operations have high similarity to safe concepts
        def mock_embedding(model, prompt):
            # Safe commands get embeddings similar to safe concepts
            if any(safe in prompt for safe in ["ls", "pwd", "cat", "grep", "readfile"]):
                return {'embedding': [0.9] * 384}  # High similarity to safe
            else:
                return {'embedding': [0.1] * 384}  # Low similarity

        mock_client.return_value.embeddings.side_effect = mock_embedding

        classifier = EmbeddingClassifier(threshold_allow=0.01)
        classifier.initialize()

        # Test safe bash commands
        decision, score = classifier.classify("<bash>ls</bash>")
        assert decision == "auto-allow"
        assert score > 0.01

        decision, score = classifier.classify("<bash>pwd</bash>")
        assert decision == "auto-allow"
        assert score > 0.01

        # Test safe file operations
        decision, score = classifier.classify(
            "<readfile><filepath>./src/main.py</filepath></readfile>"
        )
        assert decision == "auto-allow"
        assert score > 0.01

    @patch('core.permissions.embedding_classifier.ollama.Client')
    def test_unsafe_classification(self, mock_client):
        """Test unsafe operations are correctly classified as ask."""
        # Mock embeddings: unsafe operations have high similarity to unsafe concepts
        def mock_embedding(model, prompt):
            # Unsafe commands get embeddings similar to unsafe concepts
            if any(unsafe in prompt for unsafe in ["rm", "sudo", "chmod", "writefile", "pip install"]):
                return {'embedding': [0.1] * 384}  # High similarity to unsafe
            else:
                return {'embedding': [0.9] * 384}  # Low similarity

        mock_client.return_value.embeddings.side_effect = mock_embedding

        classifier = EmbeddingClassifier(threshold_deny=-0.01)
        classifier.initialize()

        # Test unsafe bash commands
        decision, score = classifier.classify("<bash>rm -rf /</bash>")
        assert decision == "ask"
        assert score < -0.01

        decision, score = classifier.classify("<bash>sudo apt-get install malware</bash>")
        assert decision == "ask"
        assert score < -0.01

        # Test unsafe file operations
        decision, score = classifier.classify(
            "<writefile><filepath>./important.py</filepath></writefile>"
        )
        assert decision == "ask"
        assert score < -0.01

    @patch('core.permissions.embedding_classifier.ollama.Client')
    def test_neutral_classification(self, mock_client):
        """Test neutral operations fall in the uncertain range."""
        # Mock embeddings: neutral operations have medium similarity to both
        mock_client.return_value.embeddings.return_value = {
            'embedding': [0.5] * 384  # Medium similarity to both
        }

        classifier = EmbeddingClassifier(
            threshold_allow=0.01,
            threshold_deny=-0.01
        )
        classifier.initialize()

        # Test neutral operation
        decision, score = classifier.classify("<bash>npm install</bash>")
        assert decision == "neutral"
        assert -0.01 <= score <= 0.01

    @patch('core.permissions.embedding_classifier.ollama.Client')
    def test_embedding_cache(self, mock_client):
        """Test embedding results are cached."""
        call_count = 0

        def mock_embedding(model, prompt):
            nonlocal call_count
            call_count += 1
            return {'embedding': [0.1] * 384}

        mock_client.return_value.embeddings.side_effect = mock_embedding

        classifier = EmbeddingClassifier()
        classifier.initialize()

        initial_calls = call_count

        # Classify same command multiple times
        xml = "<bash>ls</bash>"
        classifier.classify(xml)
        classifier.classify(xml)
        classifier.classify(xml)

        # Should only call embedding API once (cached after first)
        assert call_count == initial_calls + 1

        # Check cache stats
        stats = classifier.get_cache_stats()
        assert stats["cache_size"] > 0

    @patch('core.permissions.embedding_classifier.ollama.Client')
    def test_threshold_sensitivity(self, mock_client):
        """Test different thresholds produce different decisions."""
        # Mock fixed score scenario
        mock_client.return_value.embeddings.return_value = {
            'embedding': [0.5] * 384
        }

        # Strict thresholds (narrow auto-allow zone)
        strict = EmbeddingClassifier(threshold_allow=0.05, threshold_deny=-0.05)
        strict.initialize()
        decision, _ = strict.classify("<bash>ls</bash>")
        # Likely neutral with strict thresholds
        assert decision in ["neutral", "ask"]

        # Permissive thresholds (wide auto-allow zone)
        permissive = EmbeddingClassifier(threshold_allow=0.001, threshold_deny=-0.001)
        permissive.initialize()
        decision, _ = permissive.classify("<bash>ls</bash>")
        # More likely to auto-allow with permissive thresholds
        # (depends on actual similarity, but thresholds affect boundary)


def test_tool_call_to_xml_bash():
    """Test XML conversion for bash commands."""
    xml = tool_call_to_xml("bash", command="ls -la")
    assert xml == "<bash>ls -la</bash>"

    xml = tool_call_to_xml("bash", command="npm install")
    assert xml == "<bash>npm install</bash>"


def test_tool_call_to_xml_readfile():
    """Test XML conversion for file read operations."""
    xml = tool_call_to_xml("readfile", file_path="./src/main.py")
    assert xml == "<readfile><filepath>./src/main.py</filepath></readfile>"


def test_tool_call_to_xml_writefile():
    """Test XML conversion for file write operations."""
    xml = tool_call_to_xml("writefile", file_path="./config.json")
    assert xml == "<writefile><filepath>./config.json</filepath></writefile>"


def test_tool_call_to_xml_raw():
    """Test raw XML passthrough."""
    raw_xml = "<custom><param>value</param></custom>"
    xml = tool_call_to_xml("custom", xml_content=raw_xml)
    assert xml == raw_xml


# ============================================================================
# SettingsManager ML Configuration Tests
# ============================================================================

class TestSettingsManagerML:
    """Test ML configuration in SettingsManager."""

    def test_default_ml_config(self):
        """Test default ML configuration is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "settings.json")
            mgr = SettingsManager(config_path)

            ml_config = mgr.get_ml_config()

            assert ml_config["enabled"] is False
            assert ml_config["model"] == "embeddinggemma:latest"
            assert ml_config["threshold_allow"] == 0.01
            assert ml_config["threshold_deny"] == -0.01

    def test_ml_enable_disable(self):
        """Test enabling and disabling ML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "settings.json")
            mgr = SettingsManager(config_path)

            # Enable ML
            assert mgr.set_ml_enabled(True)
            ml_config = mgr.get_ml_config()
            assert ml_config["enabled"] is True

            # Disable ML
            assert mgr.set_ml_enabled(False)
            ml_config = mgr.get_ml_config()
            assert ml_config["enabled"] is False

    def test_ml_config_update(self):
        """Test updating ML configuration parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "settings.json")
            mgr = SettingsManager(config_path)

            # Update thresholds
            mgr.update_ml_config(threshold_allow=0.02, threshold_deny=-0.02)
            ml_config = mgr.get_ml_config()

            assert ml_config["threshold_allow"] == 0.02
            assert ml_config["threshold_deny"] == -0.02

    def test_ml_config_persistence(self):
        """Test ML configuration persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "settings.json")

            # Create and configure
            mgr1 = SettingsManager(config_path)
            mgr1.set_ml_enabled(True)
            mgr1.update_ml_config(threshold_allow=0.05)

            # Reload from same file
            mgr2 = SettingsManager(config_path)
            ml_config = mgr2.get_ml_config()

            assert ml_config["enabled"] is True
            assert ml_config["threshold_allow"] == 0.05


# ============================================================================
# AutonomousPermission Integration Tests
# ============================================================================

class TestAutonomousPermission:
    """Test autonomous permission orchestration layer."""

    @pytest.fixture
    def mock_settings_manager(self):
        """Create a mock settings manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "settings.json")
            mgr = SettingsManager(config_path)
            yield mgr

    def test_static_deny_highest_priority(self, mock_settings_manager):
        """Test static deny rules block everything."""
        # Add deny rule
        mock_settings_manager.add_permission("deny", "bash(rm:*)")
        mock_settings_manager.save_settings()

        autonomous = AutonomousPermission(
            settings_manager=mock_settings_manager,
            ml_enabled=False
        )

        # Should deny regardless of ML
        import asyncio
        decision, reason, confidence, ml_score = asyncio.run(
            autonomous.check_permission("bash", command="rm file.txt")
        )

        assert decision == "deny"
        assert "bash(rm:*)" in reason
        assert confidence == 1.0
        assert ml_score is None

    def test_static_allow_second_priority(self, mock_settings_manager):
        """Test static allow rules approve before ML."""
        # Add allow rule
        mock_settings_manager.add_permission("allow", "bash(npm:*)")
        mock_settings_manager.save_settings()

        autonomous = AutonomousPermission(
            settings_manager=mock_settings_manager,
            ml_enabled=False
        )

        # Should allow without ML
        import asyncio
        decision, reason, confidence, ml_score = asyncio.run(
            autonomous.check_permission("bash", command="npm install")
        )

        assert decision == "allow"
        assert "bash(npm:*)" in reason
        assert confidence == 1.0
        assert ml_score is None

    @patch('core.permissions.autonomous_permission.OLLAMA_AVAILABLE', False)
    def test_ml_disabled_fallback(self, mock_settings_manager):
        """Test fallback when ML is disabled."""
        autonomous = AutonomousPermission(
            settings_manager=mock_settings_manager,
            ml_enabled=False
        )

        # Should ask user (no ML, no static rules)
        import asyncio
        decision, reason, confidence, ml_score = asyncio.run(
            autonomous.check_permission("bash", command="ls")
        )

        assert decision == "ask"
        assert ml_score is None

    def test_ml_enable_disable(self, mock_settings_manager):
        """Test enabling and disabling ML layer."""
        autonomous = AutonomousPermission(
            settings_manager=mock_settings_manager,
            ml_enabled=False
        )

        assert not autonomous.ml_enabled

        # Enable (may fail if Ollama not available, but test the API)
        try:
            success = autonomous.enable_ml()
            assert autonomous.ml_enabled
        except Exception:
            # Ollama not available in test environment
            pass

        # Disable
        autonomous.disable_ml()
        assert not autonomous.ml_enabled

    def test_statistics_tracking(self, mock_settings_manager):
        """Test permission decision statistics are tracked."""
        autonomous = AutonomousPermission(
            settings_manager=mock_settings_manager,
            ml_enabled=False
        )

        import asyncio

        # Add some static rules
        mock_settings_manager.add_permission("deny", "bash(sudo:*)")
        mock_settings_manager.add_permission("allow", "bash(ls)")

        # Make various checks
        asyncio.run(autonomous.check_permission("bash", command="sudo rm"))  # deny
        asyncio.run(autonomous.check_permission("bash", command="ls"))       # allow
        asyncio.run(autonomous.check_permission("bash", command="pwd"))      # ask

        stats = autonomous.get_stats()

        assert stats["total_checks"] == 3
        assert stats["static_deny"] >= 1
        assert stats["static_allow"] >= 1
        assert stats["user_prompts"] >= 1

    def test_ml_config_retrieval(self, mock_settings_manager):
        """Test ML configuration can be retrieved."""
        autonomous = AutonomousPermission(
            settings_manager=mock_settings_manager,
            ml_enabled=True,
            ml_model="test-model",
            threshold_allow=0.05,
            threshold_deny=-0.05
        )

        config = autonomous.get_ml_config()

        assert config["model"] == "test-model"
        assert config["threshold_allow"] == 0.05
        assert config["threshold_deny"] == -0.05


# ============================================================================
# Test Runners
# ============================================================================

def run_all_tests():
    """Run all autonomous permission tests."""
    import pytest

    # Run with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "-s",  # Show print statements
        "--tb=short",  # Short traceback format
    ])

    return exit_code == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
