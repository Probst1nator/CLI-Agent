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

    Handle file uploads from the frontend.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        import tempfile

        from core.globals import g

        reader = await request.multipart()
        uploaded_files = []

        while True:
            field = await reader.next()
            if field is None:
                break

            if field.name == 'file':
                filename = field.filename
                if not filename:
                    continue

                file_path = Path(tempfile.gettempdir()) / filename

                with open(file_path, 'wb') as f:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)

                uploaded_files.append(str(file_path.absolute()))
                logger.info(f"File uploaded: {filename} -> {file_path}")

        if hasattr(g, 'current_images'):
            g.current_images.extend(uploaded_files)

        return web.json_response({
            'status': 'success',
            'files': uploaded_files,
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
