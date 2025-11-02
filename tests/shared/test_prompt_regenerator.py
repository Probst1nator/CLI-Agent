"""
Unit tests for prompt regenerator module.

Tests the LLM-powered prompt generation system that analyzes recent study files
and generates contextual prompts for empty chats.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Mark all tests in this module as using anyio (since anyio is available)
pytestmark = pytest.mark.anyio

from shared.utils.prompt_regenerator import PromptRegenerator, regenerate_prompts


@pytest.fixture
def temp_studium_dir():
    """Create a temporary Studium directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        studium_path = Path(tmpdir) / "Studium"
        studium_path.mkdir()

        # Create some test files with different modification times
        test_files = [
            ("KIM3/EinführungInSimulationsverfahren/Vorlesung1.pdf", 1),
            ("KIM3/EinführungInSimulationsverfahren/Vorlesung2.pdf", 5),
            ("KIM3/MaschinellesLernen/Assignment1.ipynb", 10),
            ("KIM3/MaschinellesLernen/Notes.txt", 15),
            ("Physik/Quantenmechanik/Lecture.pdf", 20),
        ]

        for file_path, days_ago in test_files:
            full_path = studium_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f"Test content for {file_path}")

            # Set modification time
            mtime = (datetime.now() - timedelta(days=days_ago)).timestamp()
            full_path.touch()
            import os
            os.utime(full_path, (mtime, mtime))

        yield studium_path


@pytest.fixture
def temp_prompts_file():
    """Create a temporary prompts file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        prompts_path = Path(f.name)

    yield prompts_path

    # Cleanup
    if prompts_path.exists():
        prompts_path.unlink()


class TestPromptRegenerator:
    """Test suite for PromptRegenerator class."""

    def test_initialization(self, temp_studium_dir, temp_prompts_file):
        """Test that regenerator initializes with correct parameters."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=20
        )

        assert regenerator.studium_dir == temp_studium_dir
        assert regenerator.prompts_file == temp_prompts_file
        assert regenerator.num_files == 3
        assert regenerator.time_frame_days == 20
        assert len(regenerator.default_prompts) == 5

    def test_find_recent_files(self, temp_studium_dir, temp_prompts_file):
        """Test finding recently modified files."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=25  # Should find 4 files (1, 5, 10, 15, 20 days ago)
        )

        files = regenerator.find_recent_files()

        assert len(files) == 5  # All test files are within 25 days
        assert all(isinstance(f, tuple) for f in files)
        assert all(isinstance(f[0], Path) for f in files)
        assert all(isinstance(f[1], datetime) for f in files)

    def test_find_recent_files_with_short_timeframe(self, temp_studium_dir, temp_prompts_file):
        """Test finding files with a short time frame."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=5,
            time_frame_days=7  # Should find 2 files (1 and 5 days ago)
        )

        files = regenerator.find_recent_files()

        assert len(files) == 2

    def test_find_recent_files_missing_directory(self, temp_prompts_file):
        """Test handling of missing Studium directory."""
        regenerator = PromptRegenerator(
            studium_dir="/nonexistent/path",
            prompts_file=str(temp_prompts_file),
            num_files=5,
            time_frame_days=30
        )

        files = regenerator.find_recent_files()

        assert files == []

    def test_select_random_files(self, temp_studium_dir, temp_prompts_file):
        """Test random file selection."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        # Create 5 dummy files
        all_files = [
            (Path(f"/path/file{i}.txt"), datetime.now())
            for i in range(5)
        ]

        selected = regenerator.select_random_files(all_files)

        assert len(selected) == 3
        assert all(f in all_files for f in selected)

    def test_select_random_files_fewer_than_requested(self, temp_studium_dir, temp_prompts_file):
        """Test selection when fewer files exist than requested."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=10,
            time_frame_days=30
        )

        all_files = [(Path(f"/path/file{i}.txt"), datetime.now()) for i in range(3)]

        selected = regenerator.select_random_files(all_files)

        assert len(selected) == 3  # Returns all available files

    def test_build_llm_prompt(self, temp_studium_dir, temp_prompts_file):
        """Test LLM prompt generation."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        files = [
            (Path("/path/file1.pdf"), datetime(2025, 1, 15)),
            (Path("/path/file2.txt"), datetime(2025, 1, 20)),
        ]

        prompt = regenerator.build_llm_prompt(files)

        assert "generate 5 contextual prompt suggestions" in prompt
        assert "file1.pdf" in prompt
        assert "file2.txt" in prompt
        assert "2025-01-15" in prompt
        assert "2025-01-20" in prompt
        assert "JSON object" in prompt
        assert str(temp_studium_dir) in prompt

    async def test_generate_prompts_with_llm_success(self, temp_studium_dir, temp_prompts_file):
        """Test successful LLM prompt generation."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        files = [(Path("/path/file.pdf"), datetime.now())]

        # Mock LLM response
        mock_response = '''```json
{
  "prompts": [
    "Explain quantum mechanics concepts",
    "Create practice problems for ML",
    "Summarize recent lecture notes",
    "Generate study questions",
    "Connect topics from different courses"
  ]
}
```'''

        with patch('core.llm_router.LlmRouter') as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.generate_completion = AsyncMock(return_value=mock_response)

            prompts = await regenerator.generate_prompts_with_llm(files)

        assert prompts is not None
        assert len(prompts) == 5
        assert prompts[0] == "Explain quantum mechanics concepts"
        assert prompts[4] == "Connect topics from different courses"

    async def test_generate_prompts_with_llm_json_without_code_block(self, temp_studium_dir, temp_prompts_file):
        """Test parsing JSON response without code block markers."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        files = [(Path("/path/file.pdf"), datetime.now())]

        # Mock LLM response without code blocks
        mock_response = '''{"prompts": ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]}'''

        with patch('core.llm_router.LlmRouter') as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.generate_completion = AsyncMock(return_value=mock_response)

            prompts = await regenerator.generate_prompts_with_llm(files)

        assert prompts is not None
        assert len(prompts) == 5

    async def test_generate_prompts_with_llm_failure(self, temp_studium_dir, temp_prompts_file):
        """Test handling of LLM generation failure."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        files = [(Path("/path/file.pdf"), datetime.now())]

        with patch('core.llm_router.LlmRouter') as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.generate_completion = AsyncMock(side_effect=Exception("LLM error"))

            prompts = await regenerator.generate_prompts_with_llm(files)

        assert prompts is None

    async def test_generate_prompts_with_llm_invalid_json(self, temp_studium_dir, temp_prompts_file):
        """Test handling of invalid JSON from LLM."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        files = [(Path("/path/file.pdf"), datetime.now())]

        # Mock invalid JSON response
        mock_response = "This is not valid JSON"

        with patch('core.llm_router.LlmRouter') as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.generate_completion = AsyncMock(return_value=mock_response)

            prompts = await regenerator.generate_prompts_with_llm(files)

        assert prompts is None

    def test_write_prompts_file(self, temp_studium_dir, temp_prompts_file):
        """Test writing prompts to file."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]

        success = regenerator.write_prompts_file(prompts)

        assert success is True
        assert temp_prompts_file.exists()

        # Verify content
        with open(temp_prompts_file) as f:
            data = json.load(f)

        assert data["prompts"] == prompts

    def test_write_prompts_file_creates_parent_dirs(self, temp_studium_dir):
        """Test that write_prompts_file creates parent directories."""
        prompts_file = Path(temp_studium_dir) / "nested" / "dir" / "prompts.json"

        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(prompts_file),
            num_files=3,
            time_frame_days=30
        )

        prompts = ["Test prompt"]
        success = regenerator.write_prompts_file(prompts)

        assert success is True
        assert prompts_file.exists()
        assert prompts_file.parent.exists()

    async def test_regenerate_full_flow_with_files(self, temp_studium_dir, temp_prompts_file):
        """Test complete regeneration flow with actual files."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        # Mock LLM response
        mock_response = '''{"prompts": ["P1", "P2", "P3", "P4", "P5"]}'''

        with patch('core.llm_router.LlmRouter') as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.generate_completion = AsyncMock(return_value=mock_response)

            result = await regenerator.regenerate()

        assert result["status"] == "success"
        assert len(result["prompts"]) == 5
        assert result["files_analyzed"] > 0
        assert "elapsed_seconds" in result
        assert temp_prompts_file.exists()

    async def test_regenerate_no_files_found(self, temp_prompts_file):
        """Test regeneration when no files are found."""
        regenerator = PromptRegenerator(
            studium_dir="/nonexistent/path",
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        result = await regenerator.regenerate()

        assert result["status"] == "partial"
        assert result["prompts"] == regenerator.default_prompts
        assert result["files_analyzed"] == 0
        assert "No files found" in result["message"]

    async def test_regenerate_llm_failure_fallback(self, temp_studium_dir, temp_prompts_file):
        """Test that regeneration falls back to defaults when LLM fails."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        with patch('core.llm_router.LlmRouter') as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.generate_completion = AsyncMock(side_effect=Exception("LLM error"))

            result = await regenerator.regenerate()

        assert result["status"] == "partial"
        assert result["prompts"] == regenerator.default_prompts
        assert "LLM generation failed" in result["message"]


class TestConvenienceFunction:
    """Test the convenience function for API access."""

    async def test_regenerate_prompts_function(self, temp_studium_dir, temp_prompts_file):
        """Test the regenerate_prompts convenience function."""
        # Mock LLM response
        mock_response = '''{"prompts": ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5"]}'''

        with patch('core.llm_router.LlmRouter') as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.generate_completion = AsyncMock(return_value=mock_response)

            with patch('shared.utils.prompt_regenerator.PromptRegenerator') as MockRegenerator:
                mock_regen = MockRegenerator.return_value
                mock_regen.regenerate = AsyncMock(return_value={
                    "status": "success",
                    "prompts": ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5"],
                    "files_analyzed": 3
                })

                result = await regenerate_prompts(
                    studium_dir=str(temp_studium_dir),
                    num_files=3,
                    time_frame_days=30
                )

        assert result["status"] == "success"
        assert len(result["prompts"]) == 5


# Assign test level for hierarchical testing
test_level = 3  # Shared utilities tests

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
