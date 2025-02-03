# File: globals.py
import os
from typing import List

class Globals:
    PROJ_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJ_VSCODE_DIR_PATH = os.path.join(PROJ_DIR_PATH, '.vscode')
    PROJ_ENV_FILE_PATH = os.path.join(PROJ_DIR_PATH, '.env')
    PROJ_AGENTIC_PATH = os.path.join(PROJ_DIR_PATH, 'agentic')
    PROJ_SANDBOX_PATH = os.path.join(PROJ_VSCODE_DIR_PATH, 'sandbox')
    PROJ_AGENTIC_SANDBOX_PATH = os.path.join(PROJ_AGENTIC_PATH, 'sandbox')
    PROJ_AGENTIC_SANDBOX_BACKUP_PATH = os.path.join(PROJ_AGENTIC_PATH, 'sandbox_backup')
    CURRENT_WORKING_DIR_PATH = os.getcwd()
    CURRENT_MOST_INTELLIGENT_MODEL_KEY: str = "gpt4-o"
    DEBUG_LOGGING: bool = False  # Global debug logging flag
    
    RECENT_ACTIONS: List[str] = []

    os.makedirs(PROJ_VSCODE_DIR_PATH, exist_ok=True)
    PROJ_CONFIG_FILE_PATH = os.path.join(PROJ_VSCODE_DIR_PATH, 'cli-agent.json')
    PROJ_MEMORY_FILE_PATH = os.path.join(PROJ_VSCODE_DIR_PATH, 'agent_memory.json')

    @classmethod
    def get_path(cls, path_name: str) -> str:
        """Get a project path by name."""
        return getattr(cls, f"{path_name.upper()}_PATH", cls.PROJ_DIR_PATH)

    @classmethod
    def remember_recent_action(cls, action: str) -> None:
        """Remember a recent action"""
        cls.RECENT_ACTIONS.append(action)
        with open(cls.PROJ_MEMORY_FILE_PATH, "w") as f:
            f.write("\n".join(cls.RECENT_ACTIONS))
    
    @classmethod
    def get_recent_actions(cls) -> List[str]:
        """Get recent actions"""
        with open(cls.PROJ_MEMORY_FILE_PATH, "r") as f:
            return f.read().split("\n")

g = Globals()