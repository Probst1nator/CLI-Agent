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

from shared.utils.prompt_regenerator import PromptRegenerator


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

    def test_find_recent_files_excludes_venv(self, temp_studium_dir, temp_prompts_file):
        """Test that virtual environment directories are automatically excluded."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=5,
            time_frame_days=30
        )

        # Create a virtual environment directory with pyvenv.cfg
        venv_dir = temp_studium_dir / "KIM3" / "my_project" / "venv"
        venv_dir.mkdir(parents=True, exist_ok=True)

        # Create pyvenv.cfg file (marker for Python virtual environments)
        pyvenv_cfg = venv_dir / "pyvenv.cfg"
        pyvenv_cfg.write_text("home = /usr/bin\nversion = 3.12.3")

        # Create files inside the venv that should be excluded
        venv_lib = venv_dir / "lib" / "python3.12" / "site-packages"
        venv_lib.mkdir(parents=True, exist_ok=True)
        (venv_lib / "module.py").write_text("# venv module")

        # Create a normal file outside venv that should be included
        normal_file = temp_studium_dir / "KIM3" / "my_project" / "main.py"
        normal_file.parent.mkdir(parents=True, exist_ok=True)
        normal_file.write_text("# normal file")

        # Set modification times to ensure they're recent
        import os
        recent_time = datetime.now().timestamp()
        os.utime(pyvenv_cfg, (recent_time, recent_time))
        os.utime(venv_lib / "module.py", (recent_time, recent_time))
        os.utime(normal_file, (recent_time, recent_time))

        files = regenerator.find_recent_files()
        file_paths = [str(f[0]) for f in files]

        # Verify that normal file is included
        assert any("main.py" in path for path in file_paths), "Normal file should be included"

        # Verify that venv files are excluded (pyvenv.cfg and module.py)
        assert not any("pyvenv.cfg" in path for path in file_paths), "pyvenv.cfg should be excluded"
        assert not any("site-packages" in path for path in file_paths), "venv files should be excluded"

    def test_bm25_index_building(self, temp_studium_dir, temp_prompts_file):
        """Test BM25 index building and caching."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        # Build index
        result = regenerator.build_or_load_bm25_index()

        assert result is not None
        bm25_index, file_paths = result
        assert len(file_paths) > 0
        assert all(isinstance(p, Path) for p in file_paths)

    def test_bm25_search(self, temp_studium_dir, temp_prompts_file):
        """Test BM25 search functionality."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        # Search for a term that appears in test files
        results = regenerator.search_files_with_bm25(
            query="Test content",
            top_n=5,
            return_top_recent=3
        )

        assert results is not None
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Each result should be (Path, str)
        assert all(isinstance(r[0], Path) and isinstance(r[1], str) for r in results)

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

        assert "generate 10 contextual prompt suggestions" in prompt  # Changed from 5 to 10
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

        # Mock LLM response - now with paths included
        mock_response = f'''```json
{{
  "prompts": [
    "Explain quantum mechanics concepts {temp_studium_dir}",
    "Create practice problems for ML {temp_studium_dir}",
    "Summarize recent lecture notes {temp_studium_dir}",
    "Generate study questions {temp_studium_dir}",
    "Connect topics from different courses {temp_studium_dir}"
  ]
}}
```'''

        with patch('core.llm_router.LlmRouter') as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.generate_completion = AsyncMock(return_value=mock_response)

            prompts = await regenerator.generate_prompts_with_llm(files)

        assert prompts is not None
        assert len(prompts) == 5
        # Paths are now automatically added, so check for the base text
        assert "Explain quantum mechanics concepts" in prompts[0]
        assert str(temp_studium_dir) in prompts[0]
        assert "Connect topics from different courses" in prompts[4]

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

    def test_extract_relevant_content_small_file(self, temp_studium_dir, temp_prompts_file):
        """Test excerpt extraction for small files (returns full content)."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        small_content = "This is a small file with test content."
        result = regenerator.extract_relevant_content(
            small_content,
            query="test",
            max_chars=5000
        )

        # Should return full content since it's under max_chars
        assert result == small_content

    def test_extract_relevant_content_large_file(self, temp_studium_dir, temp_prompts_file):
        """Test excerpt extraction for large files using BM25."""
        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        # Create large content with relevant paragraphs
        large_content = "\n\n".join([
            "This is paragraph 1 about random topics.",
            "This is paragraph 2 about machine learning and neural networks.",
            "This is paragraph 3 about quantum physics.",
            "This is paragraph 4 about machine learning algorithms and training.",
            "This is paragraph 5 about chemistry.",
        ] * 200)  # Make it large enough to trigger excerpt extraction

        result = regenerator.extract_relevant_content(
            large_content,
            query="machine learning",
            max_chars=5000,
            num_excerpts=2
        )

        # Should contain relevant paragraphs
        assert "machine learning" in result.lower()
        assert len(result) < len(large_content)
        assert "[...]" in result  # Should have excerpt separators

    async def test_bm25_integration_with_custom_bias(self, temp_studium_dir, temp_prompts_file):
        """Test full BM25 integration when custom_bias_text is provided."""
        # Create files with specific content
        ml_file = temp_studium_dir / "machine_learning.txt"
        ml_file.write_text("Deep learning neural networks backpropagation gradient descent")

        physics_file = temp_studium_dir / "physics.txt"
        physics_file.write_text("Quantum mechanics wave function Schrödinger equation")

        regenerator = PromptRegenerator(
            studium_dir=str(temp_studium_dir),
            prompts_file=str(temp_prompts_file),
            num_files=3,
            time_frame_days=30
        )

        # Search for machine learning
        results = regenerator.search_files_with_bm25(
            query="neural networks machine learning",
            top_n=10,
            return_top_recent=2
        )

        assert results is not None
        assert len(results) > 0
        # ML file should be in results
        result_paths = [str(r[0]) for r in results]
        assert any("machine_learning.txt" in p for p in result_paths)

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


# Assign test level for hierarchical testing
test_level = 3  # Shared utilities tests

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
