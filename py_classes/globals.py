# File: globals.py
import os

class Globals:
    PROJ_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    PROJ_VSCODE_DIR_PATH = os.path.join(PROJ_DIR_PATH, '.vscode')
    PROJ_ENV_FILE_PATH = os.path.join(PROJ_DIR_PATH, '.env')
    PROJ_AGENTIC_PATH = os.path.join(PROJ_DIR_PATH, 'agentic')
    PROJ_AGENTIC_SANDBOX_PATH = os.path.join(PROJ_AGENTIC_PATH, 'sandbox')
    PROJ_AGENTIC_SANDBOX_BACKUP_PATH = os.path.join(PROJ_AGENTIC_PATH, 'sandbox_backup')
    CURRENT_WORKING_DIR_PATH = os.getcwd()  # New property for current working directory

    os.makedirs(PROJ_VSCODE_DIR_PATH, exist_ok=True)
    PROJ_CONFIG_FILE_PATH = os.path.join(PROJ_VSCODE_DIR_PATH, 'cli-agent.json')

    @classmethod
    def get_path(cls, path_name: str) -> str:
        """Get a project path by name."""
        return getattr(cls, f"{path_name.upper()}_PATH", cls.PROJ_DIR_PATH)

g = Globals()