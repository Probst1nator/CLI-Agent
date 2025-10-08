"""
Agent State API

Provides endpoints for getting and synchronizing agent state (voice_mode, auto_execute, etc.)
with the frontend.
"""

import logging
from typing import Any

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_routes(app: Any) -> None:
    """Register agent state API routes"""
    if not AIOHTTP_AVAILABLE:
        return

    app.router.add_get('/api/agent/state', get_agent_state)


async def get_agent_state(request: Any) -> Any:
    """GET /api/agent/state

    Get current agent state including:
    - voice_mode: Full voice mode (mic + TTS)
    - tts_only_mode: TTS-only mode (no mic)
    - auto_execute: YOLO mode status
    - online_mode: Online features enabled
    - local_mode: Force local models

    Returns: {
        "status": "success",
        "state": {
            "voice_mode": bool,
            "tts_only_mode": bool,
            "auto_execute": bool,
            "online_mode": bool,
            "local_mode": bool
        }
    }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        state = {
            'voice_mode': getattr(g, 'voice_mode', False),
            'tts_only_mode': getattr(g, 'tts_only_mode', False),
            'auto_execute': getattr(g, 'auto_execute', False),
            'online_mode': getattr(g, 'online_mode', False),
            'local_mode': getattr(g, 'local_mode', False)
        }

        return web.json_response({
            'status': 'success',
            'state': state
        })

    except Exception as e:
        logger.error(f"Failed to get agent state: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e)
        }, status=500)
