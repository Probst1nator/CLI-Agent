"""
Profile API endpoints for CLI-Agent.

Provides LLM profile management endpoints.
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


async def get_profiles(request: Any) -> Any:
    """GET /api/profiles

    Get all LLM profiles and active profile.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.llm_profiles import get_profile_manager

        manager = get_profile_manager()
        profiles = manager.list_profiles()
        active_profile = manager.get_active_profile()

        profiles_data = [
            {
                'id': profile.id,
                'override': profile.override,
                'router': profile.router,
                'planner': profile.planner,
                'fallbacks': profile.fallbacks,
                'is_active': profile.id == active_profile
            }
            for profile in profiles
        ]

        return web.json_response({
            'status': 'success',
            'profiles': profiles_data,
            'active_profile': active_profile
        })

    except Exception as e:
        logger.error(f"Failed to get profiles data: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': f'Failed to retrieve profiles data: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def get_llms(request: Any) -> Any:
    """GET /api/llms

    Get all available LLM models and current configuration.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.llm import Llm
        from core.globals import g

        available_llms = Llm.get_available_llms(include_dynamic=True)

        llm_data = []
        for llm in available_llms:
            provider_name = llm.provider.__class__.__name__.replace('API', '')

            model_entry = {
                'id': llm.model_key,
                'name': llm.model_key,
                'provider': provider_name,
                'context_window': llm.context_window,
                'capabilities': [str(capability) for capability in llm.strengths] if llm.strengths else [],
                'status': 'available'
            }
            llm_data.append(model_entry)

        llm_data.sort(key=lambda x: (x['provider'], x['name']))

        g.load_llm_roles()
        llm_config = g.get_llm_config(force_reread=True)
        selected_llms = g.SELECTED_LLMS
        planner_llm = g.planner_llm
        router_llm = g.router_llm
        override_llm = g.override_llm

        logger.info(f"Served LLM data: {len(llm_data)} models from {len(set(m['provider'] for m in llm_data))} providers")

        return web.json_response({
            'status': 'success',
            'models': llm_data,
            'total_count': len(llm_data),
            'current_configuration': {
                'selected_llms': selected_llms,
                'planner_llm': planner_llm,
                'router_llm': router_llm,
                'override_llm': override_llm
            }
        })

    except Exception as e:
        logger.error(f"Failed to get LLM data: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': f'Failed to retrieve LLM data: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }, status=500)


def setup_routes(app: Any):
    """Register all profile API routes"""
    app.router.add_get('/api/profiles', get_profiles)
    app.router.add_get('/api/llms', get_llms)
    logger.info("Profile API routes registered")
