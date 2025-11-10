"""
Profile API endpoints for CLI-Agent.

Provides LLM profile management endpoints.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web

AIOHTTP_AVAILABLE = False
if not TYPE_CHECKING:
    try:
        from aiohttp import web

        AIOHTTP_AVAILABLE = True
    except ImportError:
        web = None  # type: ignore

logger = logging.getLogger(__name__)


async def get_profiles(request: Any) -> Any:
    """GET /api/profiles

    Get all LLM profiles.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.llm_profiles import get_profile_manager

        manager = get_profile_manager()
        profiles = manager.list_profiles()

        profiles_data = [
            {
                "id": profile.id,
                "override": profile.override,
                "router": profile.router,
                "planner": profile.planner,
                "fallbacks": profile.fallbacks,
            }
            for profile in profiles
        ]

        return web.json_response({"status": "success", "profiles": profiles_data})

    except Exception as e:
        logger.error(f"Failed to get profiles data: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "error",
                "error": f"Failed to retrieve profiles data: {str(e)}",
                "code": "INTERNAL_ERROR",
            },
            status=500,
        )


async def get_llms(request: Any) -> Any:
    """GET /api/llms

    Get all available LLM models and current configuration.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g
        from core.llm import Llm

        available_llms = Llm.get_available_llms(include_dynamic=True)

        llm_data = []
        for llm in available_llms:
            provider_name = llm.provider.__class__.__name__.replace("API", "")

            model_entry = {
                "id": llm.model_key,
                "name": llm.model_key,
                "provider": provider_name,
                "context_window": llm.context_window,
                "capabilities": [str(capability) for capability in llm.capabilities]
                if llm.capabilities
                else [],
                "status": "available",
            }
            llm_data.append(model_entry)

        llm_data.sort(key=lambda x: (x["provider"], x["name"]))

        g.load_llm_roles()
        llm_config = g.get_llm_config(force_reread=True)
        selected_llms = g.SELECTED_LLMS

        # Get both single model (backward compat) and multi-model arrays
        planner_llm = g.planner_llm
        router_llm = g.router_llm
        override_llm = g.override_llm
        autocomplete_llm = g.autocomplete_llm

        planner_llms = g.planner_llms_list
        router_llms = g.router_llms_list
        override_llms = g.override_llms_list
        autocomplete_llms = g.autocomplete_llms_list

        # Debug logging to trace profile load issue
        logger.info(
            f"🔍 API /llms - planner_llm: {planner_llm}, planner_llms: {planner_llms}"
        )
        logger.info(
            f"🔍 API /llms - router_llm: {router_llm}, router_llms: {router_llms}"
        )
        logger.info(
            f"🔍 API /llms - override_llm: {override_llm}, override_llms: {override_llms}"
        )
        logger.info(
            f"🔍 API /llms - autocomplete_llm: {autocomplete_llm}, autocomplete_llms: {autocomplete_llms}"
        )
        logger.info(f"🔍 API /llms - selected_llms/fallbacks: {selected_llms}")

        logger.info(
            f"Served LLM data: {len(llm_data)} models from {len(set(m['provider'] for m in llm_data))} providers"
        )

        return web.json_response(
            {
                "status": "success",
                "models": llm_data,
                "total_count": len(llm_data),
                "current_configuration": {
                    "selected_llms": selected_llms,
                    # Backward compatibility: single model properties
                    "planner_llm": planner_llm,
                    "router_llm": router_llm,
                    "override_llm": override_llm,
                    "autocomplete_llm": autocomplete_llm,
                    # New: multi-model arrays for dynamic fallback support
                    "planner_llms": planner_llms,
                    "router_llms": router_llms,
                    "override_llms": override_llms,
                    "autocomplete_llms": autocomplete_llms,
                },
            }
        )

    except Exception as e:
        logger.error(f"Failed to get LLM data: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "error",
                "error": f"Failed to retrieve LLM data: {str(e)}",
                "code": "INTERNAL_ERROR",
            },
            status=500,
        )


async def update_llm_config(request: Any) -> Any:
    """POST /api/llms/config

    Update LLM role configuration with multi-model support.

    Request body:
    {
        "role": "planner" | "router" | "override" | "autocomplete" | "fallbacks",
        "models": ["model1", "model2", ...]  // Array for multi-model fallback chain
    }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        data = await request.json()
        role = data.get("role")
        models = data.get("models", [])

        if not role:
            return web.json_response(
                {
                    "status": "error",
                    "error": "Missing required field: role",
                    "code": "INVALID_REQUEST",
                },
                status=400,
            )

        if not isinstance(models, list):
            return web.json_response(
                {
                    "status": "error",
                    "error": "models must be an array",
                    "code": "INVALID_REQUEST",
                },
                status=400,
            )

        # Update the role configuration
        g.update_llm_role(role=role, model_keys=models if models else None)

        logger.info(f"Updated LLM role '{role}' to models: {models}")

        # NOTE: Auto-save is handled by frontend, not backend
        # Frontend tracks currently loaded profile and saves to it

        return web.json_response(
            {
                "status": "success",
                "message": f"LLM role {role} updated successfully",
                "role": role,
                "models": models,
            }
        )

    except ValueError as e:
        logger.error(f"Invalid LLM role update: {e}")
        return web.json_response(
            {"status": "error", "error": str(e), "code": "INVALID_ROLE"}, status=400
        )
    except Exception as e:
        logger.error(f"Failed to update LLM config: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "error",
                "error": f"Failed to update LLM config: {str(e)}",
                "code": "INTERNAL_ERROR",
            },
            status=500,
        )


async def save_profile(request: web.Request) -> web.Response:
    """Save a profile via HTTP POST (fallback when WebSocket unavailable)"""
    try:
        data = await request.json()
        profile_id = data.get("profile_id")
        config = data.get("config")

        if not profile_id:
            return web.json_response(
                {
                    "status": "error",
                    "error": "profile_id is required",
                    "code": "MISSING_PROFILE_ID",
                },
                status=400,
            )

        if not config:
            return web.json_response(
                {
                    "status": "error",
                    "error": "config is required",
                    "code": "MISSING_CONFIG",
                },
                status=400,
            )

        # Get context from request
        context = request.app.get("context")
        if not context:
            return web.json_response(
                {
                    "status": "error",
                    "error": "Application context not available",
                    "code": "NO_CONTEXT",
                },
                status=500,
            )

        # Execute profile save command
        from agent.commands.profile_commands import ProfileSaveCommand

        command = ProfileSaveCommand(context)

        # Format args like WebSocket command: "/profile_save 1"
        args = profile_id
        result = command.execute(args)

        logger.info(f"Profile {profile_id} saved via HTTP")

        return web.json_response(
            {
                "status": "success",
                "message": f"Profile {profile_id} saved",
                "profile_id": profile_id,
            }
        )

    except Exception as e:
        logger.error(f"Failed to save profile via HTTP: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "error",
                "error": f"Failed to save profile: {str(e)}",
                "code": "SAVE_FAILED",
            },
            status=500,
        )


async def delete_profile(request: web.Request) -> web.Response:
    """Delete a profile via HTTP DELETE (fallback when WebSocket unavailable)"""
    try:
        profile_id = request.match_info.get("profile_id")

        if not profile_id:
            return web.json_response(
                {
                    "status": "error",
                    "error": "profile_id is required",
                    "code": "MISSING_PROFILE_ID",
                },
                status=400,
            )

        # Get context from request
        context = request.app.get("context")
        if not context:
            return web.json_response(
                {
                    "status": "error",
                    "error": "Application context not available",
                    "code": "NO_CONTEXT",
                },
                status=500,
            )

        # Execute profile delete command
        from agent.commands.profile_commands import ProfileDeleteCommand

        command = ProfileDeleteCommand(context)

        # Format args like WebSocket command: "/profile_delete 1"
        args = profile_id
        result = command.execute(args)

        logger.info(f"Profile {profile_id} deleted via HTTP")

        return web.json_response(
            {
                "status": "success",
                "message": f"Profile {profile_id} deleted",
                "profile_id": profile_id,
            }
        )

    except Exception as e:
        logger.error(f"Failed to delete profile via HTTP: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "error",
                "error": f"Failed to delete profile: {str(e)}",
                "code": "DELETE_FAILED",
            },
            status=500,
        )


def setup_routes(app: Any):
    """Register all profile API routes"""
    app.router.add_get("/api/profiles", get_profiles)
    app.router.add_get("/api/llms", get_llms)
    app.router.add_post("/api/llms/config", update_llm_config)
    app.router.add_post("/api/profiles/save", save_profile)
    app.router.add_delete("/api/profiles/{profile_id}", delete_profile)
    logger.info("Profile API routes registered")
