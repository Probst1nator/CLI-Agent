"""
Tools API endpoints for CLI-Agent.

Provides tool discovery and configuration endpoints.
"""

import importlib
import logging
from pathlib import Path
from typing import Any

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class UploadSecurity:
    """Dual-mode upload security with environment-based configuration"""

    def __init__(self):
        import os

        self.is_containerized = os.getenv("CLI_AGENT_MODE") == "containerized"

        # Stricter limits for SaaS mode
        if self.is_containerized:
            self.max_size = 10 * 1024 * 1024  # 10MB for SaaS
            self.max_files_per_request = 5
        else:
            self.max_size = 20 * 1024 * 1024  # 20MB for localhost
            self.max_files_per_request = 20

        self.allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.pdf', '.txt', '.md'}

        # Different storage paths based on mode
        if self.is_containerized:
            # In container: use workspace volume
            self.upload_dir = Path('/workspace/.cli-agent/uploads')
        else:
            # Localhost: use home directory
            self.upload_dir = Path.home() / '.cli-agent' / 'uploads'

        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_filename(self, filename: str) -> str:
        """Validate and sanitize filename, return UUID-based safe name"""
        import uuid

        # Strip directory components
        safe_name = Path(filename).name

        # Check for path traversal
        if '..' in safe_name or '/' in safe_name or '\\' in safe_name:
            raise ValueError("Invalid filename: path traversal detected")

        # Check extension
        ext = Path(safe_name).suffix.lower()
        if ext not in self.allowed_extensions:
            raise ValueError(f"File type {ext} not allowed. Allowed: {', '.join(self.allowed_extensions)}")

        # Generate UUID-based name to prevent collisions
        unique_name = f"{uuid.uuid4().hex}{ext}"
        return unique_name

    async def save_upload(self, field, user_uuid: str = None) -> dict:
        """Save uploaded file with validation"""
        import time

        filename = field.filename
        if not filename:
            raise ValueError("No filename provided")

        # Validate and generate safe filename
        safe_filename = self.validate_filename(filename)

        # For SaaS mode, organize by user UUID
        if self.is_containerized and user_uuid:
            user_dir = self.upload_dir / user_uuid[:8]  # First 8 chars of UUID
            user_dir.mkdir(exist_ok=True)
            file_path = user_dir / safe_filename
        else:
            file_path = self.upload_dir / safe_filename

        # Write with size limit
        size = 0
        try:
            with open(file_path, 'wb') as f:
                while True:
                    chunk = await field.read_chunk(8192)
                    if not chunk:
                        break

                    size += len(chunk)
                    if size > self.max_size:
                        f.close()
                        file_path.unlink()
                        raise ValueError(
                            f"File exceeds {self.max_size // (1024*1024)}MB limit"
                        )

                    f.write(chunk)
        except Exception as e:
            # Cleanup on error
            if file_path.exists():
                file_path.unlink()
            raise

        # Optional: MIME type verification (requires python-magic)
        try:
            import magic
            mime = magic.from_file(str(file_path), mime=True)

            # Basic MIME validation
            expected_mimes = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.webp': 'image/webp',
                '.pdf': 'application/pdf',
                '.txt': 'text/plain',
                '.md': 'text/plain'
            }

            ext = Path(safe_filename).suffix.lower()
            if mime != expected_mimes.get(ext):
                logger.warning(f"MIME mismatch: {mime} != {expected_mimes.get(ext)}")
                # Only fail in containerized mode for security
                if self.is_containerized:
                    file_path.unlink()
                    raise ValueError(f"File content type mismatch")

        except ImportError:
            # python-magic not installed, skip MIME check
            logger.debug("python-magic not available, skipping MIME verification")

        return {
            'path': str(file_path),
            'original_name': filename,
            'safe_name': safe_filename,
            'size': size,
            'uploaded_at': time.time()
        }


# Global instance
_upload_security = None


def get_upload_security():
    """Get or create upload security instance"""
    global _upload_security
    if _upload_security is None:
        _upload_security = UploadSecurity()
    return _upload_security


async def get_tools(request: Any) -> Any:
    """GET /api/tools

    Discover and return available tools for the frontend tool manager.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        tools_dir = Path(__file__).parent.parent / "tools"
        tools_data = []

        tool_directories = [
            (tools_dir, "tools"),
            (tools_dir / "optional", "tools.optional")
        ]

        for directory, module_prefix in tool_directories:
            if not directory.exists():
                continue

            for file in directory.glob("*.py"):
                if file.name.startswith("_") or file.stem == "__init__":
                    continue

                try:
                    module_name = f"{module_prefix}.{file.stem}"
                    module = importlib.import_module(module_name)

                    tool_class_name = file.stem
                    if hasattr(module, tool_class_name):
                        tool_class = getattr(module, tool_class_name)
                        if hasattr(tool_class, 'get_delim') and hasattr(tool_class, 'get_tool_info'):
                            try:
                                tool_info = tool_class.get_tool_info()
                                tool_delim = tool_class.get_delim()

                                tools_data.append({
                                    'name': tool_class_name,
                                    'delim': tool_delim,
                                    'info': tool_info,
                                    'displayName': tool_info.get('name', tool_class_name),
                                    'description': tool_info.get('description', ''),
                                    'example': tool_info.get('example', '')
                                })
                            except Exception as tool_error:
                                logger.debug(f"Could not get info for tool {tool_class_name}: {tool_error}")
                except Exception as import_error:
                    logger.debug(f"Could not import tool from {file}: {import_error}")

        tools_data.sort(key=lambda x: x['name'])

        logger.info(f"Served tools data: {len(tools_data)} tools discovered")

        return web.json_response({
            'status': 'success',
            'tools': tools_data,
            'total_count': len(tools_data)
        })

    except Exception as e:
        logger.error(f"Failed to get tools data: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': f'Failed to retrieve tools data: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def upload_file(request: Any) -> Any:
    """POST /upload

    Handle file uploads with security validation (dual-mode: localhost/SaaS).
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        security = get_upload_security()
        reader = await request.multipart()
        uploaded_files = []

        # Get user UUID from header (SaaS mode)
        user_uuid = request.headers.get('X-User-UUID')

        file_count = 0

        while True:
            field = await reader.next()
            if field is None:
                break

            if field.name == 'file':
                # Check per-request file limit
                file_count += 1
                if file_count > security.max_files_per_request:
                    return web.json_response({
                        'status': 'error',
                        'error': f'Upload limit exceeded ({security.max_files_per_request} files per request)',
                        'code': 'LIMIT_EXCEEDED'
                    }, status=429)

                try:
                    upload_info = await security.save_upload(field, user_uuid=user_uuid)
                    uploaded_files.append(upload_info)
                    logger.info(f"File uploaded: {upload_info['original_name']} -> {upload_info['safe_name']} ({upload_info['size']} bytes)")

                except ValueError as e:
                    logger.warning(f"Upload validation failed: {e}")
                    return web.json_response({
                        'status': 'error',
                        'error': str(e),
                        'code': 'VALIDATION_FAILED'
                    }, status=400)

        # Add to session images
        if hasattr(g, 'current_images'):
            g.current_images.extend([u['path'] for u in uploaded_files])

        return web.json_response({
            'status': 'success',
            'files': uploaded_files,
            'count': len(uploaded_files),
            'message': f'Successfully uploaded {len(uploaded_files)} file(s)'
        })

    except Exception as e:
        logger.error(f"File upload failed: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'UPLOAD_FAILED'
        }, status=500)


async def update_disabled_tools(request: Any) -> Any:
    """POST /api/tools/disabled

    Update the list of disabled tools in global state.
    This allows the frontend tool manager to control which tools are available to the agent.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        data = await request.json()
        disabled_tools = data.get('disabled_tools', [])

        # Validate that disabled_tools is a list
        if not isinstance(disabled_tools, list):
            return web.json_response({
                'status': 'error',
                'error': 'disabled_tools must be a list',
                'code': 'INVALID_INPUT'
            }, status=400)

        # Store in global state
        g.disabled_tools = disabled_tools

        logger.info(f"Updated disabled tools: {disabled_tools}")

        # Update active chat's system prompt with new disabled tools
        try:
            from main import update_system_prompt
            update_system_prompt()
        except Exception as e:
            logger.warning(f"Failed to update system prompt: {e}")

        return web.json_response({
            'status': 'success',
            'disabled_tools': disabled_tools,
            'message': f'Updated disabled tools list ({len(disabled_tools)} tools disabled)'
        })

    except Exception as e:
        logger.error(f"Failed to update disabled tools: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


def setup_routes(app: Any):
    """Register all tools API routes"""
    app.router.add_get('/api/tools', get_tools)
    app.router.add_post('/upload', upload_file)
    app.router.add_post('/api/tools/disabled', update_disabled_tools)
    logger.info("Tools API routes registered")
