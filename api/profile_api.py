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

        # Get both single model (backward compat) and multi-model arrays
        planner_llm = g.planner_llm
        router_llm = g.router_llm
        override_llm = g.override_llm

        planner_llms = g.planner_llms_list
        router_llms = g.router_llms_list
        override_llms = g.override_llms_list

        # Debug logging to trace profile load issue
        logger.info(f"🔍 API /llms - planner_llm: {planner_llm}, planner_llms: {planner_llms}")
        logger.info(f"🔍 API /llms - router_llm: {router_llm}, router_llms: {router_llms}")
        logger.info(f"🔍 API /llms - override_llm: {override_llm}, override_llms: {override_llms}")
        logger.info(f"🔍 API /llms - selected_llms/fallbacks: {selected_llms}")

        logger.info(f"Served LLM data: {len(llm_data)} models from {len(set(m['provider'] for m in llm_data))} providers")

        return web.json_response({
            'status': 'success',
            'models': llm_data,
            'total_count': len(llm_data),
            'current_configuration': {
                'selected_llms': selected_llms,
                # Backward compatibility: single model properties
                'planner_llm': planner_llm,
                'router_llm': router_llm,
                'override_llm': override_llm,
                # New: multi-model arrays for dynamic fallback support
                'planner_llms': planner_llms,
                'router_llms': router_llms,
                'override_llms': override_llms
            }
        })

    except Exception as e:
        logger.error(f"Failed to get LLM data: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': f'Failed to retrieve LLM data: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def update_llm_config(request: Any) -> Any:
    """POST /api/llms/config

    Update LLM role configuration with multi-model support.

    Request body:
    {
        "role": "planner" | "router" | "override" | "fallbacks",
        "models": ["model1", "model2", ...]  // Array for multi-model fallback chain
    }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        data = await request.json()
        role = data.get('role')
        models = data.get('models', [])

        if not role:
            return web.json_response({
                'status': 'error',
                'error': 'Missing required field: role',
                'code': 'INVALID_REQUEST'
            }, status=400)

        if not isinstance(models, list):
            return web.json_response({
                'status': 'error',
                'error': 'models must be an array',
                'code': 'INVALID_REQUEST'
            }, status=400)

        # Update the role configuration
        g.update_llm_role(role=role, model_keys=models if models else None)

        logger.info(f"Updated LLM role '{role}' to models: {models}")

        return web.json_response({
            'status': 'success',
            'message': f'LLM role {role} updated successfully',
            'role': role,
            'models': models
        })

    except ValueError as e:
        logger.error(f"Invalid LLM role update: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INVALID_ROLE'
        }, status=400)
    except Exception as e:
        logger.error(f"Failed to update LLM config: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': f'Failed to update LLM config: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }, status=500)


def setup_routes(app: Any):
    """Register all profile API routes"""
    app.router.add_get('/api/profiles', get_profiles)
    app.router.add_get('/api/llms', get_llms)
    app.router.add_post('/api/llms/config', update_llm_config)
    logger.info("Profile API routes registered")
