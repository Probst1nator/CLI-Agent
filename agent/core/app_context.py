"""
Application Context for CLI-Agent.

This module provides a centralized context object that replaces the global 
state singleton pattern, enabling dependency injection and better testability.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from core.chat import Chat
    from core.llm_router import LlmRouter
    from agent.notebook.computational_notebook import ComputationalNotebook
    from agent.playbook.playbook_manager import PlaybookManager
    from agent.utils_manager.utils_manager import UtilsManager

@dataclass
class LlmConfig:
    """LLM configuration settings with proper typing."""
    model_key: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = True
    provider: str = ""
    context_window: Optional[int] = None

@dataclass
class AgentSettings:
    """Agent behavior settings."""
    debug_mode: bool = False
    verbose_debug: bool = False
    auto_execute: bool = False
    voice_mode: bool = False
    local_mode: bool = False
    online_mode: bool = False
    full_output_mode: bool = False
    monte_carlo_mode: bool = False
    fast_mode: bool = False

class AppContext:
    """
    Central application context that manages all shared state.
    
    This class replaces the global singleton pattern with explicit
    dependency injection, making the code more testable and maintainable.
    """
    
    def __init__(self) -> None:
        # Core components (lazy-loaded)
        self._notebook: Optional[ComputationalNotebook] = None
        self._llm_router: Optional[LlmRouter] = None
        self._playbook_manager: Optional[PlaybookManager] = None
        self._utils_manager: Optional[UtilsManager] = None
        
        # Configuration
        self.llm_config: LlmConfig = LlmConfig()
        self.settings: AgentSettings = AgentSettings()
        
        # Runtime state
        self.current_chat: Optional[Chat] = None
        self.base64_images: List[str] = field(default_factory=list)
        self.tool_classes: Dict[str, Any] = field(default_factory=dict)
        
        # File and path management
        self.project_root: Path = Path.cwd()
        self.config_dir: Path = Path.home() / ".cli-agent"
        self.temp_files: List[Path] = field(default_factory=list)
        
        # Async runtime
        self._locks: Dict[str, asyncio.Lock] = field(default_factory=dict)
        
        # Logging
        self.logger: logging.Logger = logging.getLogger("cli_agent")
        
    def get_notebook(self) -> ComputationalNotebook:
        """Get computational notebook instance (lazy-loaded)."""
        if self._notebook is None:
            from agent.notebook.computational_notebook import ComputationalNotebook
            self._notebook = ComputationalNotebook()
        return self._notebook
        
    def get_llm_router(self) -> LlmRouter:
        """Get LLM router instance (lazy-loaded)."""
        if self._llm_router is None:
            from core.llm_router import LlmRouter
            self._llm_router = LlmRouter()
        return self._llm_router
        
    def get_playbook_manager(self) -> PlaybookManager:
        """Get playbook manager instance (lazy-loaded)."""
        if self._playbook_manager is None:
            from agent.playbook.playbook_manager import PlaybookManager
            self._playbook_manager = PlaybookManager()
        return self._playbook_manager
        
    def get_utils_manager(self) -> UtilsManager:
        """Get utils manager instance (lazy-loaded)."""
        if self._utils_manager is None:
            from agent.utils_manager.utils_manager import UtilsManager
            self._utils_manager = UtilsManager()
        return self._utils_manager
        
    def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a named async lock."""
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]
        
    def update_llm_config(self, **kwargs: Any) -> None:
        """Update LLM configuration with type checking."""
        for key, value in kwargs.items():
            if hasattr(self.llm_config, key):
                setattr(self.llm_config, key, value)
            else:
                self.logger.warning(f"Unknown LLM config parameter: {key}")
                
    def update_settings(self, **kwargs: Any) -> None:
        """Update agent settings with type checking."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
            else:
                self.logger.warning(f"Unknown setting: {key}")
                
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self.temp_files.clear()
        
    def __enter__(self) -> AppContext:
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup_temp_files()