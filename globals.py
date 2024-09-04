# File: globals.py
import os
from pathlib import Path

class Globals:
    PROJ_DIR_PATH = str(Path(__file__).resolve().parent)
    PROJ_VSCODE_DIR_PATH = str(Path(PROJ_DIR_PATH) / '.vscode')
    PROJ_ENV_FILE_PATH = str(Path(PROJ_DIR_PATH) / '.env')
    PROJ_SANDBOX_PATH = str(Path(PROJ_DIR_PATH) / 'sandbox')
    CURRENT_WORKING_DIR_PATH = os.getcwd()  # New property for current working directory

    os.makedirs(PROJ_VSCODE_DIR_PATH, exist_ok=True)
    PROJ_CONFIG_FILE_PATH = str(Path(PROJ_VSCODE_DIR_PATH) / 'cli-agent.json')

    @classmethod
    def get_path(cls, path_name: str) -> str:
        """Get a project path by name."""
        return getattr(cls, f"{path_name.upper()}_PATH", cls.PROJ_DIR_PATH)

g = Globals()