"""
Memory file utilities for CLI-Agent.

Provides functions to save large content to memory files instead of
overwhelming the context window or using in-memory resource managers.
"""
from datetime import datetime
from pathlib import Path


def save_to_memory_file(content: str, source_name: str) -> Path:
    """
    Save large content to a memory file.

    Args:
        content: The text content to save
        source_name: Source identifier (used in filename)

    Returns:
        Path to the created memory file
    """
    memory_dir = Path.home() / ".cli-agent" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp and sanitized source name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name)
    memory_file = memory_dir / f"{timestamp}_{safe_source}.txt"

    # Write content
    memory_file.write_text(content, encoding='utf-8')
    return memory_file
