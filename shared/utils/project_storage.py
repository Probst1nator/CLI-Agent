"""
Project Storage Manager

Manages per-project storage in centralized ~/.cli-agent/projects/ directory.
Each project gets its own subdirectory identified by sanitized absolute path.

Structure:
    ~/.cli-agent/
        projects/
            home_user_repos_myproject/
                metadata.json           # Project info
                chat_history/           # Archived chats
                recent_chat.json        # Active chat
"""

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def sanitize_path(absolute_path: str) -> str:
    """
    Convert an absolute path to a safe directory name.

    Examples:
        ~/repos/CLI-Agent -> home_user_repos_CLI-Agent
        /var/www/my-site -> var_www_my-site
        C:\\Users\\Name\\Project -> C_Users_Name_Project

    Args:
        absolute_path: Absolute path to sanitize

    Returns:
        Sanitized directory name safe for filesystems
    """
    # Convert to absolute path if not already
    abs_path = os.path.abspath(absolute_path)

    # Remove leading slash/drive letter separator
    if abs_path.startswith('/'):
        sanitized = abs_path[1:]  # Remove leading /
    elif len(abs_path) >= 2 and abs_path[1] == ':':
        # Windows drive letter (C:\ -> C_)
        sanitized = abs_path[0] + abs_path[2:]
    else:
        sanitized = abs_path

    # Replace path separators with underscores
    sanitized = sanitized.replace(os.sep, '_')
    sanitized = sanitized.replace('/', '_')  # Handle both separators
    sanitized = sanitized.replace('\\', '_')

    # Replace other unsafe characters with underscores
    # Keep alphanumeric, dash, underscore, and period
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', sanitized)

    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Ensure not empty
    if not sanitized:
        sanitized = 'unknown_project'

    return sanitized


def get_project_storage_path(cwd: Optional[str] = None) -> Path:
    """
    Get the centralized storage path for a project.

    Args:
        cwd: Current working directory (defaults to os.getcwd())

    Returns:
        Path to project storage directory in ~/.cli-agent/projects/{sanitized}/
    """
    if cwd is None:
        cwd = os.getcwd()

    sanitized = sanitize_path(cwd)
    base_path = Path.home() / '.cli-agent' / 'projects'
    project_path = base_path / sanitized

    return project_path


def ensure_project_directory(cwd: Optional[str] = None, auto_migrate: bool = True) -> Path:
    """
    Ensure project directory exists and has metadata.
    Optionally migrates from local ./.cli-agent/ if found.

    Args:
        cwd: Current working directory (defaults to os.getcwd())
        auto_migrate: If True, automatically migrate ./.cli-agent/ if found

    Returns:
        Path to project storage directory
    """
    if cwd is None:
        cwd = os.getcwd()

    project_path = get_project_storage_path(cwd)

    # Check if we need to migrate from local storage
    if auto_migrate:
        local_storage = Path(cwd) / '.cli-agent'
        if local_storage.exists() and local_storage.is_dir():
            migrate_local_to_global(cwd)

    # Create project directory structure
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / 'chat_history').mkdir(exist_ok=True)

    # Ensure metadata exists
    metadata_path = project_path / 'metadata.json'
    if not metadata_path.exists():
        # Create initial metadata
        metadata = {
            'absolute_path': os.path.abspath(cwd),
            'project_name': os.path.basename(cwd),
            'created_at': time.time(),
            'last_accessed': time.time(),
            'chat_count': 0,
            'version': '1.0'
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Created project metadata for {cwd}")

    # Update last accessed time
    update_project_metadata(cwd, last_accessed=time.time())

    return project_path


def load_project_metadata(cwd: Optional[str] = None) -> dict[str, Any]:
    """
    Load project metadata.

    Args:
        cwd: Current working directory (defaults to os.getcwd())

    Returns:
        Dictionary containing project metadata
    """
    if cwd is None:
        cwd = os.getcwd()

    project_path = get_project_storage_path(cwd)
    metadata_path = project_path / 'metadata.json'

    if not metadata_path.exists():
        # Return default metadata if not exists
        return {
            'absolute_path': os.path.abspath(cwd),
            'project_name': os.path.basename(cwd),
            'created_at': time.time(),
            'last_accessed': time.time(),
            'chat_count': 0,
            'version': '1.0'
        }

    try:
        with open(metadata_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load project metadata: {e}")
        return {}


def update_project_metadata(cwd: Optional[str] = None, **kwargs) -> None:
    """
    Update project metadata with provided key-value pairs.

    Args:
        cwd: Current working directory (defaults to os.getcwd())
        **kwargs: Metadata fields to update
    """
    if cwd is None:
        cwd = os.getcwd()

    project_path = get_project_storage_path(cwd)
    metadata_path = project_path / 'metadata.json'

    # Load existing metadata
    metadata = load_project_metadata(cwd)

    # Update with new values
    metadata.update(kwargs)

    # Ensure directory exists
    project_path.mkdir(parents=True, exist_ok=True)

    # Write updated metadata
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to update project metadata: {e}")


def migrate_local_to_global(cwd: Optional[str] = None) -> bool:
    """
    Migrate local ./.cli-agent/ directory to centralized global storage.

    Args:
        cwd: Current working directory (defaults to os.getcwd())

    Returns:
        True if migration successful or not needed, False on error
    """
    if cwd is None:
        cwd = os.getcwd()

    local_storage = Path(cwd) / '.cli-agent'

    # Check if local storage exists
    if not local_storage.exists():
        logger.debug(f"No local .cli-agent found in {cwd}, skipping migration")
        return True

    # Check if it's a file (the wrapper script) rather than a directory
    if not local_storage.is_dir():
        logger.debug(f".cli-agent in {cwd} is a file, not a directory - skipping migration")
        return True

    project_path = get_project_storage_path(cwd)

    logger.info(f"Migrating local .cli-agent from {cwd} to {project_path}")

    try:
        # Ensure target directory exists
        project_path.mkdir(parents=True, exist_ok=True)

        # Migrate each file/directory
        migrated_items = []
        for item in local_storage.iterdir():
            if item.name.startswith('.'):
                # Skip hidden files/markers
                continue

            target = project_path / item.name

            if item.is_file():
                # Copy file
                shutil.copy2(item, target)
                logger.debug(f"Copied {item.name}")
                migrated_items.append(item.name)
            elif item.is_dir():
                # Copy directory tree
                if target.exists():
                    # Merge directories (e.g., chat_history)
                    for subitem in item.rglob('*'):
                        if subitem.is_file():
                            rel_path = subitem.relative_to(item)
                            target_file = target / rel_path
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(subitem, target_file)
                else:
                    shutil.copytree(item, target)
                logger.debug(f"Copied directory {item.name}")
                migrated_items.append(item.name)

        # Create metadata with migration info
        metadata = {
            'absolute_path': os.path.abspath(cwd),
            'project_name': os.path.basename(cwd),
            'created_at': time.time(),
            'last_accessed': time.time(),
            'migrated_at': time.time(),
            'migrated_from': str(local_storage),
            'migrated_items': migrated_items,
            'chat_count': len(list((project_path / 'chat_history').glob('*.json'))) if (project_path / 'chat_history').exists() else 0,
            'version': '1.0'
        }

        with open(project_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Remove old .cli-agent directory after successful migration
        shutil.rmtree(local_storage)
        logger.info(f"Successfully migrated {len(migrated_items)} items from {local_storage} to {project_path} and removed old directory")
        return True

    except Exception as e:
        logger.error(f"Failed to migrate local storage: {e}", exc_info=True)
        return False


def list_all_projects() -> list[dict[str, Any]]:
    """
    List all projects in centralized storage.

    Returns:
        List of project metadata dictionaries
    """
    base_path = Path.home() / '.cli-agent' / 'projects'

    if not base_path.exists():
        return []

    projects = []
    for project_dir in base_path.iterdir():
        if not project_dir.is_dir():
            continue

        metadata_path = project_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata['storage_path'] = str(project_dir)
                    projects.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {project_dir}: {e}")

    # Sort by last accessed (most recent first)
    projects.sort(key=lambda p: p.get('last_accessed', 0), reverse=True)

    return projects
