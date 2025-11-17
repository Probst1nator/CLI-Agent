"""
Permissions API endpoints for CLI-Agent.

Provides permission management endpoints for the settings UI.
Permissions are stored per-profile in LLM profiles.
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


async def get_permissions(request: Any) -> Any:
    """GET /api/permissions

    Get current active profile's permission settings.

    Returns:
        {
            "status": "success",
            "permissions": {
                "allow": [...],
                "deny": [...],
                "ask": [...]
            },
            "profile_id": "1"
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g
        from core.llm_profiles import get_profile_manager

        # Get active profile
        profile_id = g.get_active_profile_id()

        if not profile_id:
            # No active profile, return current settings from SettingsManager
            from core.permissions.settings_manager import SettingsManager

            mgr = SettingsManager()
            permissions = mgr.get_all_permissions()

            return web.json_response({
                "status": "success",
                "permissions": permissions,
                "profile_id": None
            })

        # Get permissions from active profile
        profile_manager = get_profile_manager()
        profile = profile_manager.get_profile(profile_id)

        if not profile:
            logger.warning(f"Active profile '{profile_id}' not found")
            return web.json_response({
                "status": "error",
                "error": f"Profile '{profile_id}' not found",
                "code": "PROFILE_NOT_FOUND"
            }, status=404)

        return web.json_response({
            "status": "success",
            "permissions": profile.permissions,
            "profile_id": profile_id
        })

    except Exception as e:
        logger.error(f"Failed to get permissions: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": f"Failed to retrieve permissions: {str(e)}",
            "code": "INTERNAL_ERROR"
        }, status=500)


async def update_permissions(request: Any) -> Any:
    """POST /api/permissions

    Update permissions.
    If an active profile exists, update it. Otherwise, just update settings.json.

    Request body:
        {
            "allow": ["bash(npm:*)", "readfile(./src/**)"],
            "deny": ["bash(sudo:*)", "readfile(.env)"],
            "ask": ["bash(git push:*)"]
        }

    Returns:
        {
            "status": "success",
            "message": "Permissions updated"
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g
        from core.llm_profiles import get_profile_manager
        from core.permissions.settings_manager import SettingsManager

        data = await request.json()

        # Validate input
        allow = data.get("allow", [])
        deny = data.get("deny", [])
        ask = data.get("ask", [])

        if not isinstance(allow, list) or not isinstance(deny, list) or not isinstance(ask, list):
            return web.json_response({
                "status": "error",
                "error": "allow, deny, and ask must be arrays",
                "code": "INVALID_INPUT"
            }, status=400)

        # Get active profile (if any)
        profile_id = g.get_active_profile_id()

        if profile_id:
            # Profile is active - update the profile
            profile_manager = get_profile_manager()
            profile = profile_manager.get_profile(profile_id)

            if profile:
                # Update permissions in profile
                profile.update_permissions(allow, deny, ask)

                # Save profile
                profile_manager.save_profiles()

                # Apply permissions immediately
                profile.apply_permissions()

                logger.info(f"Updated permissions for profile '{profile_id}'")

                return web.json_response({
                    "status": "success",
                    "message": f"Permissions updated for profile {profile_id}"
                })

        # No active profile or profile not found - update settings.json directly
        mgr = SettingsManager()
        mgr.settings["permissions"] = {
            "allow": allow.copy() if allow else [],
            "deny": deny.copy() if deny else [],
            "ask": ask.copy() if ask else []
        }
        mgr.save_settings()

        logger.info("Updated permissions (no active profile)")

        return web.json_response({
            "status": "success",
            "message": "Permissions updated"
        })

    except Exception as e:
        logger.error(f"Failed to update permissions: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": f"Failed to update permissions: {str(e)}",
            "code": "INTERNAL_ERROR"
        }, status=500)


async def reset_permissions(request: Any) -> Any:
    """POST /api/permissions/reset

    Reset permissions to default template.
    If an active profile exists, update it. Otherwise, just update settings.json.

    Returns:
        {
            "status": "success",
            "message": "Permissions reset to default template",
            "permissions": {...}
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        import json
        from pathlib import Path

        from core.globals import g
        from core.llm_profiles import get_profile_manager
        from core.permissions.settings_manager import SettingsManager

        # Load default template
        template_path = Path(__file__).parent.parent / "templates" / "default_settings.json"

        if not template_path.exists():
            return web.json_response({
                "status": "error",
                "error": "Default template not found",
                "code": "TEMPLATE_NOT_FOUND"
            }, status=500)

        with open(template_path) as f:
            template = json.load(f)

        default_permissions = template.get("permissions", {
            "allow": [],
            "deny": [],
            "ask": []
        })

        # Get active profile (if any)
        profile_id = g.get_active_profile_id()

        if profile_id:
            # Profile is active - update the profile
            profile_manager = get_profile_manager()
            profile = profile_manager.get_profile(profile_id)

            if profile:
                # Reset permissions in profile
                profile.update_permissions(
                    default_permissions.get("allow", []),
                    default_permissions.get("deny", []),
                    default_permissions.get("ask", [])
                )

                # Save profile
                profile_manager.save_profiles()

                # Apply permissions immediately
                profile.apply_permissions()

                logger.info(f"Reset permissions for profile '{profile_id}' to default template")

                return web.json_response({
                    "status": "success",
                    "message": f"Permissions reset to default template (profile {profile_id})",
                    "permissions": profile.permissions
                })

        # No active profile or profile not found - update settings.json directly
        mgr = SettingsManager()
        mgr.settings["permissions"] = {
            "allow": default_permissions.get("allow", []).copy(),
            "deny": default_permissions.get("deny", []).copy(),
            "ask": default_permissions.get("ask", []).copy()
        }
        mgr.save_settings()

        logger.info("Reset permissions to default template (no active profile)")

        return web.json_response({
            "status": "success",
            "message": "Permissions reset to default template",
            "permissions": mgr.settings["permissions"]
        })

    except Exception as e:
        logger.error(f"Failed to reset permissions: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": f"Failed to reset permissions: {str(e)}",
            "code": "INTERNAL_ERROR"
        }, status=500)


async def save_as_default_permissions(request: Any) -> Any:
    """POST /api/permissions/save-as-default

    Save current permissions as the default template.
    Gets current permissions from active profile or settings.json,
    then saves them to core/permissions/default_settings.json.

    Returns:
        {
            "status": "success",
            "message": "Current permissions saved as default template",
            "permissions": {...}
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        import json
        from pathlib import Path

        from core.globals import g
        from core.llm_profiles import get_profile_manager
        from core.permissions.settings_manager import SettingsManager

        # Get current permissions
        profile_id = g.get_active_profile_id()
        current_permissions = None

        if profile_id:
            # Get from active profile
            profile_manager = get_profile_manager()
            profile = profile_manager.get_profile(profile_id)

            if profile:
                current_permissions = profile.permissions
                logger.info(f"Getting permissions from profile '{profile_id}' to save as default")
            else:
                logger.warning(f"Active profile '{profile_id}' not found, using settings.json")

        if not current_permissions:
            # Get from settings.json
            mgr = SettingsManager()
            current_permissions = mgr.get_all_permissions()
            logger.info("Getting permissions from settings.json to save as default")

        # Load current template to preserve other settings
        template_path = Path(__file__).parent.parent / "core" / "permissions" / "default_settings.json"

        if template_path.exists():
            with open(template_path) as f:
                template = json.load(f)
        else:
            # Create minimal template if it doesn't exist
            template = {}

        # Update permissions in template
        template["permissions"] = current_permissions

        # Save updated template
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)

        logger.info(f"Saved current permissions as default template to {template_path}")

        return web.json_response({
            "status": "success",
            "message": "Current permissions saved as default template",
            "permissions": current_permissions
        })

    except Exception as e:
        logger.error(f"Failed to save as default: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": f"Failed to save as default: {str(e)}",
            "code": "INTERNAL_ERROR"
        }, status=500)


async def add_permission_rule(request: Any) -> Any:
    """POST /api/permissions/add-rule

    Add a single permission rule to the current permissions.
    If an active profile exists, update it. Otherwise, update settings.json.

    Request body:
        {
            "permission_type": "allow" | "deny" | "ask",
            "rule": "bash(npm:*)" | "readfile(./src/**)" | tool name, etc.
        }

    Returns:
        {
            "status": "success",
            "message": "Permission rule added",
            "permissions": {...}
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g
        from core.llm_profiles import get_profile_manager
        from core.permissions.settings_manager import SettingsManager

        data = await request.json()

        # Validate input
        permission_type = data.get("permission_type")
        rule = data.get("rule")

        if not permission_type or permission_type not in ["allow", "deny", "ask"]:
            return web.json_response({
                "status": "error",
                "error": "permission_type must be 'allow', 'deny', or 'ask'",
                "code": "INVALID_INPUT"
            }, status=400)

        if not rule or not isinstance(rule, str):
            return web.json_response({
                "status": "error",
                "error": "rule must be a non-empty string",
                "code": "INVALID_INPUT"
            }, status=400)

        # Get active profile (if any)
        profile_id = g.get_active_profile_id()

        if profile_id:
            # Profile is active - update the profile
            profile_manager = get_profile_manager()
            profile = profile_manager.get_profile(profile_id)

            if profile:
                # Get current permissions
                current_perms = profile.permissions.copy()

                # Add the new rule to the appropriate list (if not already there)
                if rule not in current_perms.get(permission_type, []):
                    current_perms.setdefault(permission_type, []).append(rule)

                    # Update profile
                    profile.update_permissions(
                        current_perms.get("allow", []),
                        current_perms.get("deny", []),
                        current_perms.get("ask", [])
                    )

                    # Save profile
                    profile_manager.save_profiles()

                    # Apply permissions immediately
                    profile.apply_permissions()

                    logger.info(f"Added permission rule '{rule}' ({permission_type}) to profile '{profile_id}'")

                return web.json_response({
                    "status": "success",
                    "message": f"Permission rule added to profile {profile_id}",
                    "permissions": profile.permissions
                })

        # No active profile or profile not found - update settings.json directly
        mgr = SettingsManager()
        mgr.add_permission(permission_type, rule)
        mgr.save_settings()

        logger.info(f"Added permission rule '{rule}' ({permission_type}) to settings.json")

        return web.json_response({
            "status": "success",
            "message": "Permission rule added",
            "permissions": mgr.get_all_permissions()
        })

    except Exception as e:
        logger.error(f"Failed to add permission rule: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": f"Failed to add permission rule: {str(e)}",
            "code": "INTERNAL_ERROR"
        }, status=500)


async def get_ml_config(request: Any) -> Any:
    """GET /api/permissions/ml-config

    Get ML configuration.

    Returns:
        {
            "status": "success",
            "ml_config": {
                "enabled": false,
                "model": "embeddinggemma:latest",
                "threshold_allow": 0.01,
                "threshold_deny": -0.01,
                ...
            }
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.permissions.settings_manager import SettingsManager

        mgr = SettingsManager()
        ml_config = mgr.get_ml_config()

        return web.json_response({
            "status": "success",
            "ml_config": ml_config
        })

    except Exception as e:
        logger.error(f"Failed to get ML config: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "INTERNAL_ERROR"
        }, status=500)


async def update_ml_config(request: Any) -> Any:
    """POST /api/permissions/ml-config

    Update ML configuration.

    Request body:
        {
            "enabled": true,
            "threshold_allow": 0.02,
            "threshold_deny": -0.02,
            ...
        }

    Returns:
        {
            "status": "success",
            "message": "ML config updated",
            "ml_config": {...}
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.permissions.settings_manager import SettingsManager

        data = await request.json()

        mgr = SettingsManager()

        # Special handling for enabled flag
        if "enabled" in data:
            mgr.set_ml_enabled(data["enabled"])
            # Remove from data to avoid duplicate update
            data = {k: v for k, v in data.items() if k != "enabled"}

        # Update other config parameters
        if data:
            mgr.update_ml_config(**data)

        return web.json_response({
            "status": "success",
            "message": "ML configuration updated",
            "ml_config": mgr.get_ml_config()
        })

    except Exception as e:
        logger.error(f"Failed to update ML config: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "INTERNAL_ERROR"
        }, status=500)


async def get_ml_stats(request: Any) -> Any:
    """GET /api/permissions/ml-stats

    Get ML permission statistics.

    Returns:
        {
            "status": "success",
            "stats": {
                "total_checks": 156,
                "static_deny": 12,
                "ml_auto_allow": 89,
                ...
            }
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.permissions.settings_manager import SettingsManager

        mgr = SettingsManager()
        ml_config = mgr.get_ml_config()

        if not ml_config.get("enabled", False):
            return web.json_response({
                "status": "success",
                "stats": None,
                "message": "ML not enabled"
            })

        # Try to get stats from autonomous permission instance
        # Note: In production, this should be a singleton/global instance
        try:
            from core.permissions.autonomous_permission import AutonomousPermission

            autonomous = AutonomousPermission(
                settings_manager=mgr,
                ml_enabled=True,
                ml_model=ml_config.get("model", "embeddinggemma:latest"),
                threshold_allow=ml_config.get("threshold_allow", 0.01),
                threshold_deny=ml_config.get("threshold_deny", -0.01)
            )

            stats = autonomous.get_stats()

            return web.json_response({
                "status": "success",
                "stats": stats
            })

        except Exception as e:
            logger.warning(f"Could not get ML stats: {e}")
            return web.json_response({
                "status": "success",
                "stats": None,
                "message": "Statistics not available (ML not initialized)"
            })

    except Exception as e:
        logger.error(f"Failed to get ML stats: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "INTERNAL_ERROR"
        }, status=500)


def setup_routes(app: Any):
    """Register all permissions API routes"""
    app.router.add_get("/api/permissions", get_permissions)
    app.router.add_post("/api/permissions", update_permissions)
    app.router.add_post("/api/permissions/add-rule", add_permission_rule)
    app.router.add_post("/api/permissions/reset", reset_permissions)
    app.router.add_post("/api/permissions/save-as-default", save_as_default_permissions)
    # ML endpoints
    app.router.add_get("/api/permissions/ml-config", get_ml_config)
    app.router.add_post("/api/permissions/ml-config", update_ml_config)
    app.router.add_get("/api/permissions/ml-stats", get_ml_stats)
    logger.info("Permissions API routes registered (including ML endpoints)")
