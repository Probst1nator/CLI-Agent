"""
LLM Profile Management System

Allows saving and loading different LLM configurations as profiles.
Each profile contains: override, autocomplete, router, planner, and fallback models.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional


class LLMProfile:
    """Represents a single LLM configuration profile."""

    def __init__(
        self,
        id: str,
        override: Optional[list[str]] = None,
        autocomplete: Optional[list[str]] = None,
        router: Optional[list[str]] = None,
        planner: Optional[list[str]] = None,
        fallbacks: Optional[list[str]] = None,
        permissions: Optional[dict[str, list[str]]] = None
    ):
        self.id = id
        self.override = override or []
        self.autocomplete = autocomplete or []
        self.router = router or []
        self.planner = planner or []
        self.fallbacks = fallbacks or []
        self.permissions = permissions or {
            "allow": [],
            "deny": [],
            "ask": []
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "override": self.override,
            "autocomplete": self.autocomplete,
            "router": self.router,
            "planner": self.planner,
            "fallbacks": self.fallbacks,
            "permissions": self.permissions
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'LLMProfile':
        """Create profile from dictionary. Handles both old (single value) and new (array) formats."""
        def normalize_to_list(value):
            """Convert single value or None to list."""
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return [value] if value else []

        # Load permissions with default structure if missing
        permissions = data.get("permissions", {
            "allow": [],
            "deny": [],
            "ask": []
        })

        # Ensure permissions has all required keys
        if not isinstance(permissions, dict):
            permissions = {"allow": [], "deny": [], "ask": []}
        else:
            permissions.setdefault("allow", [])
            permissions.setdefault("deny", [])
            permissions.setdefault("ask", [])

        return cls(
            id=data.get("id", ""),
            override=normalize_to_list(data.get("override")),
            autocomplete=normalize_to_list(data.get("autocomplete")),
            router=normalize_to_list(data.get("router")),
            planner=normalize_to_list(data.get("planner")),
            fallbacks=normalize_to_list(data.get("fallbacks")),
            permissions=permissions
        )

    def is_empty(self) -> bool:
        """Check if this profile is empty/corrupted (all None/empty values)."""
        return (
            len(self.override) == 0 and
            len(self.autocomplete) == 0 and
            len(self.router) == 0 and
            len(self.planner) == 0 and
            len(self.fallbacks) == 0
        )

    def apply_permissions(self) -> None:
        """Apply this profile's permissions to the SettingsManager."""
        from core.permissions.settings_manager import SettingsManager

        mgr = SettingsManager()

        # Update permissions in settings manager
        mgr.settings["permissions"] = {
            "allow": self.permissions.get("allow", []).copy(),
            "deny": self.permissions.get("deny", []).copy(),
            "ask": self.permissions.get("ask", []).copy()
        }

        # Save to disk
        mgr.save_settings()
        logging.debug(f"Applied permissions from profile '{self.id}'")

    def update_permissions(self, allow: list[str], deny: list[str], ask: list[str]) -> None:
        """Update this profile's permissions."""
        self.permissions = {
            "allow": allow.copy() if allow else [],
            "deny": deny.copy() if deny else [],
            "ask": ask.copy() if ask else []
        }
        logging.debug(f"Updated permissions for profile '{self.id}'")

    def apply(self) -> None:
        """Apply this profile to the current LLM configuration and permissions."""
        from core.globals import g

        if self.is_empty():
            raise ValueError(f"Cannot apply empty profile '{self.id}' - all configuration values are None/empty")

        # Pass values directly - update_llm_role() handles None and empty lists correctly
        # Don't use ternary operators because empty lists [] are falsy but valid!
        g.update_llm_role('override', model_keys=self.override)
        g.update_llm_role('autocomplete', model_keys=self.autocomplete)
        g.update_llm_role('router', model_keys=self.router)
        g.update_llm_role('planner', model_keys=self.planner)
        g.update_llm_role('fallbacks', model_keys=self.fallbacks)

        # Apply permissions
        self.apply_permissions()

        # Set this profile as active so changes persist
        g.set_active_profile_id(self.id)

        logging.info(f"Applied LLM profile: {self.id}")

    @classmethod
    def from_current_config(cls, id: str) -> 'LLMProfile':
        """Create a profile from the current LLM configuration and permissions."""
        from core.globals import g
        from core.permissions.settings_manager import SettingsManager

        # Capture current permissions
        mgr = SettingsManager()
        current_permissions = mgr.get_all_permissions()

        return cls(
            id=id,
            override=g.override_llms_list.copy() if g.override_llms_list else [],
            autocomplete=g.autocomplete_llms_list.copy() if g.autocomplete_llms_list else [],
            router=g.router_llms_list.copy() if g.router_llms_list else [],
            planner=g.planner_llms_list.copy() if g.planner_llms_list else [],
            fallbacks=g.FALLBACK_MODELS.copy() if g.FALLBACK_MODELS else [],
            permissions={
                "allow": current_permissions.get("allow", []).copy(),
                "deny": current_permissions.get("deny", []).copy(),
                "ask": current_permissions.get("ask", []).copy()
            }
        )


class LLMProfileManager:
    """Manages LLM profile storage and retrieval."""

    def __init__(self, profiles_file: Optional[Path] = None):
        """Initialize the profile manager."""
        if profiles_file:
            self.profiles_file = profiles_file
        else:
            # Default location: ~/.cli-agent/llm_profiles.json
            try:
                from core.globals import g
                config_dir = Path(g.LLM_CONFIG_PATH).parent
            except (ImportError, AttributeError):
                config_dir = Path.home() / ".cli-agent"

            config_dir.mkdir(parents=True, exist_ok=True)
            self.profiles_file = config_dir / "llm_profiles.json"

        self._profiles: dict[str, LLMProfile] = {}
        self.load_profiles()

    def load_profiles(self) -> None:
        """Load all profiles from disk."""
        if not self.profiles_file.exists():
            self._profiles = {}
            return

        try:
            with open(self.profiles_file) as f:
                data = json.load(f)

            self._profiles = {
                id: LLMProfile.from_dict(profile_data)
                for id, profile_data in data.get("profiles", {}).items()
            }

            logging.debug(f"Loaded {len(self._profiles)} LLM profiles from {self.profiles_file}")

        except Exception as e:
            logging.error(f"Failed to load LLM profiles: {e}")
            self._profiles = {}

    def save_profiles(self) -> None:
        """Save all profiles to disk."""
        try:
            data = {
                "profiles": {
                    id: profile.to_dict()
                    for id, profile in self._profiles.items()
                }
            }

            # Ensure directory exists
            self.profiles_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2)

            logging.debug(f"Saved {len(self._profiles)} LLM profiles to {self.profiles_file}")

        except Exception as e:
            logging.error(f"Failed to save LLM profiles: {e}")

    def create_profile(self, id: str, from_current: bool = True) -> LLMProfile:
        """Create a new profile."""
        if id in self._profiles:
            raise ValueError(f"Profile '{id}' already exists")

        if from_current:
            profile = LLMProfile.from_current_config(id)
        else:
            profile = LLMProfile(id=id)

        self._profiles[id] = profile
        self.save_profiles()

        logging.info(f"Created new LLM profile: {id}")
        return profile

    def update_profile(self, id: str, from_current: bool = True) -> LLMProfile:
        """Update an existing profile."""
        if id not in self._profiles:
            raise ValueError(f"Profile '{id}' does not exist")

        if from_current:
            new_profile = LLMProfile.from_current_config(id)
            self._profiles[id] = new_profile

        self.save_profiles()

        logging.info(f"Updated LLM profile: {id}")
        return self._profiles[id]

    def delete_profile(self, id: str) -> None:
        """Delete a profile."""
        if id not in self._profiles:
            raise ValueError(f"Profile '{id}' does not exist")

        del self._profiles[id]
        self.save_profiles()

        logging.info(f"Deleted LLM profile: {id}")

    def get_profile(self, id: str) -> Optional[LLMProfile]:
        """Get a profile by id."""
        return self._profiles.get(id)

    def list_profiles(self) -> list[LLMProfile]:
        """List all available profiles."""
        return list(self._profiles.values())

    def apply_profile(self, id: str) -> None:
        """Apply a profile to the current configuration (runtime only, not persisted)."""
        if id not in self._profiles:
            raise ValueError(f"Profile '{id}' does not exist")

        profile = self._profiles[id]

        # Check if profile is empty/corrupted before applying
        if profile.is_empty():
            logging.warning(f"Profile '{id}' is empty/corrupted and cannot be applied")
            raise ValueError(f"Profile '{id}' is empty/corrupted - all configuration values are None/empty")

        profile.apply()

        logging.info(f"Applied LLM profile: {id}")

    def detect_corrupted_profiles(self) -> list[str]:
        """Detect profiles that are empty/corrupted."""
        corrupted = []
        for profile_id, profile in self._profiles.items():
            if profile.is_empty():
                corrupted.append(profile_id)
        return corrupted

    def cleanup_corrupted_profiles(self, delete: bool = False) -> list[str]:
        """
        Detect and optionally delete corrupted profiles.

        Args:
            delete: If True, delete corrupted profiles. If False, just return their IDs.

        Returns:
            List of corrupted profile IDs that were detected (and deleted if delete=True)
        """
        corrupted = self.detect_corrupted_profiles()

        if corrupted:
            logging.warning(f"Found {len(corrupted)} corrupted profile(s): {', '.join(corrupted)}")

            if delete:
                for profile_id in corrupted:
                    logging.info(f"Deleting corrupted profile: {profile_id}")
                    del self._profiles[profile_id]

                self.save_profiles()
                logging.info(f"Deleted {len(corrupted)} corrupted profile(s)")

        return corrupted

    def update_active_profile_role(self, role_name: str, model_keys: Optional[list[str]]) -> None:
        """
        Update a specific role in the currently active profile.

        This is called automatically when users change LLM roles via commands like
        /override_llm, /router_llm, etc. to ensure changes persist across restarts.

        Args:
            role_name: The role to update ('override', 'router', 'planner', 'autocomplete', 'fallbacks')
            model_keys: List of model keys (or None to clear the role)
        """
        from core.globals import g

        profile_id = g.get_active_profile_id()
        if not profile_id:
            # No active profile, nothing to update
            logging.debug(f"No active profile to update for role '{role_name}'")
            return

        if profile_id not in self._profiles:
            logging.warning(f"Active profile '{profile_id}' not found in profile manager")
            return

        profile = self._profiles[profile_id]

        # Update the appropriate field
        if role_name == 'override':
            profile.override = model_keys or []
        elif role_name == 'autocomplete':
            profile.autocomplete = model_keys or []
        elif role_name == 'router':
            profile.router = model_keys or []
        elif role_name == 'planner':
            profile.planner = model_keys or []
        elif role_name == 'fallbacks':
            profile.fallbacks = model_keys or []
        else:
            logging.error(f"Unknown role name: {role_name}")
            return

        # Save the updated profile
        self.save_profiles()
        logging.debug(f"Updated profile '{profile_id}' role '{role_name}' to: {model_keys}")

    def delete_corrupted_profiles_on_startup(self) -> list[str]:
        """
        FATAL ERROR handler: Detect and DELETE corrupted profiles at startup.

        Corrupted profiles should never exist in production. If found, this indicates
        a system bug and the corrupted profiles are immediately deleted.

        Returns:
            List of profile IDs that were deleted
        """
        corrupted = self.detect_corrupted_profiles()

        if corrupted:
            for profile_id in corrupted:
                logging.critical(f"❌ FATAL ERROR: Corrupted profile '{profile_id}' detected (all values None/empty)")
                logging.info(f"🗑️  Deleting corrupted profile: {profile_id}")
                del self._profiles[profile_id]

            self.save_profiles()
            logging.critical(f"❌ Deleted {len(corrupted)} corrupted profile(s) - system initialized with blank slate")

        return corrupted


# Global profile manager instance
_profile_manager: Optional[LLMProfileManager] = None


def get_profile_manager() -> LLMProfileManager:
    """Get or create the global profile manager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = LLMProfileManager()
    return _profile_manager
