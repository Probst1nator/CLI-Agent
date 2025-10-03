"""
LLM Profile Management System

Allows saving and loading different LLM configurations as profiles.
Each profile contains: override, router, planner, and fallback models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any


class LLMProfile:
    """Represents a single LLM configuration profile."""

    def __init__(
        self,
        id: str,
        override: Optional[str] = None,
        router: Optional[str] = None,
        planner: Optional[str] = None,
        fallbacks: Optional[List[str]] = None
    ):
        self.id = id
        self.override = override
        self.router = router
        self.planner = planner
        self.fallbacks = fallbacks or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "override": self.override,
            "router": self.router,
            "planner": self.planner,
            "fallbacks": self.fallbacks
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMProfile':
        """Create profile from dictionary."""
        return cls(
            id=data.get("id", ""),
            override=data.get("override"),
            router=data.get("router"),
            planner=data.get("planner"),
            fallbacks=data.get("fallbacks", [])
        )

    def apply(self) -> None:
        """Apply this profile to the current LLM configuration."""
        from core.globals import g

        g.update_llm_role('override', self.override)
        g.update_llm_role('router', self.router)
        g.update_llm_role('planner', self.planner)
        g.update_llm_role('fallbacks', model_keys=self.fallbacks)

        logging.info(f"Applied LLM profile: {self.id}")

    @classmethod
    def from_current_config(cls, id: str) -> 'LLMProfile':
        """Create a profile from the current LLM configuration."""
        from core.globals import g

        return cls(
            id=id,
            override=g.override_llm,
            router=g.router_llm,
            planner=g.planner_llm,
            fallbacks=g.FALLBACK_MODELS.copy()
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

        self._profiles: Dict[str, LLMProfile] = {}
        self._active_profile: Optional[str] = None
        self.load_profiles()

    def load_profiles(self) -> None:
        """Load all profiles from disk."""
        if not self.profiles_file.exists():
            self._profiles = {}
            self._active_profile = None
            return

        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)

            self._profiles = {
                id: LLMProfile.from_dict(profile_data)
                for id, profile_data in data.get("profiles", {}).items()
            }
            self._active_profile = data.get("active_profile")

            logging.debug(f"Loaded {len(self._profiles)} LLM profiles from {self.profiles_file}")

        except Exception as e:
            logging.error(f"Failed to load LLM profiles: {e}")
            self._profiles = {}
            self._active_profile = None

    def save_profiles(self) -> None:
        """Save all profiles to disk."""
        try:
            data = {
                "profiles": {
                    id: profile.to_dict()
                    for id, profile in self._profiles.items()
                },
                "active_profile": self._active_profile
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

        if self._active_profile == id:
            self._active_profile = None

        self.save_profiles()

        logging.info(f"Deleted LLM profile: {id}")

    def get_profile(self, id: str) -> Optional[LLMProfile]:
        """Get a profile by id."""
        return self._profiles.get(id)

    def list_profiles(self) -> List[LLMProfile]:
        """List all available profiles."""
        return list(self._profiles.values())

    def apply_profile(self, id: str, set_active: bool = True) -> None:
        """Apply a profile to the current configuration."""
        if id not in self._profiles:
            raise ValueError(f"Profile '{id}' does not exist")

        profile = self._profiles[id]
        profile.apply()

        if set_active:
            self._active_profile = id
            self.save_profiles()

        logging.info(f"Applied LLM profile: {id}")

    def get_active_profile(self) -> Optional[str]:
        """Get the name of the currently active profile."""
        return self._active_profile

    def clear_active_profile(self) -> None:
        """Clear the active profile marker."""
        self._active_profile = None
        self.save_profiles()


# Global profile manager instance
_profile_manager: Optional[LLMProfileManager] = None


def get_profile_manager() -> LLMProfileManager:
    """Get or create the global profile manager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = LLMProfileManager()
    return _profile_manager
