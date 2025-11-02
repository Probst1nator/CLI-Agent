"""
Rate limit configuration and initialization for different providers and models.
"""

import logging

from infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker import (
    LimitType,
    enhanced_rate_limit_tracker,
)

logger = logging.getLogger(__name__)

class RateLimitConfig:
    """Manages rate limit configuration for different models and providers."""

    # Known model configurations
    MODEL_CONFIGS = {
        # Gemini Models - Free Tier (optimistic estimates - let API tell us actual limits)
        'gemini-1.5-pro': {
            LimitType.RPM: 60,  # Increased from 2 to be more permissive initially
            LimitType.TPM: 100000,  # Increased from 32000
            LimitType.RPD: 1000  # Increased from 50
        },
        'gemini-1.5-flash': {
            LimitType.RPM: 100,  # Increased from 15
            LimitType.TPM: 2000000,  # Increased from 1M
            LimitType.RPD: 5000  # Increased from 1500
        },
        'gemini-2.5-flash': {
            LimitType.RPM: 100,  # Increased from 15
            LimitType.TPM: 2000000,  # Increased from 1M
            LimitType.RPD: 5000  # Increased from 1500
        },
        'gemini-1.5-pro-latest': {
            LimitType.RPM: 60,  # Increased from 2
            LimitType.TPM: 100000,  # Increased from 32000
            LimitType.RPD: 1000  # Increased from 50
        },
        'gemini-1.5-flash-latest': {
            LimitType.RPM: 100,  # Increased from 15
            LimitType.TPM: 2000000,  # Increased from 1M
            LimitType.RPD: 5000  # Increased from 1500
        },
        # Additional Gemini model variants
        'gemini-2.5-flash-preview-05-20': {
            LimitType.RPM: 100,  # Increased from 15
            LimitType.TPM: 2000000,  # Increased from 1M
            LimitType.RPD: 5000  # Increased from 1500
        },
        # NEW: Add missing model configurations
        'gemini-2.5-flash-lite-preview-06-17': {
            LimitType.RPM: 100,
            LimitType.TPM: 2000000,
            LimitType.RPD: 5000
        },
        'gemini-2.0-flash': {
            LimitType.RPM: 100,
            LimitType.TPM: 2000000,
            LimitType.RPD: 5000
        },
        'gemini-2.5-pro': {
            LimitType.RPM: 60,
            LimitType.TPM: 100000,
            LimitType.RPD: 1000
        },
        'gemini-flash': {
            LimitType.RPM: 100,  # Increased from 15
            LimitType.TPM: 2000000,  # Increased from 1M
            LimitType.RPD: 5000  # Increased from 1500
        },
        'gemini-pro': {
            LimitType.RPM: 60,  # Increased from 2
            LimitType.TPM: 100000,  # Increased from 32000
            LimitType.RPD: 1000  # Increased from 50
        },

        # Gemini Models - Paid Tier 1 (with billing enabled)
        'gemini-1.5-pro-paid': {
            LimitType.RPM: 360,
            LimitType.TPM: 4000000,
            LimitType.RPD: 10000
        },
        'gemini-1.5-flash-paid': {
            LimitType.RPM: 1000,
            LimitType.TPM: 4000000,
            LimitType.RPD: 10000
        },

        # Groq Models (generous defaults)
        'groq-default': {
            LimitType.RPM: 500,  # Groq is generally fast and generous
            LimitType.TPM: 1000000,
            LimitType.RPD: 10000
        },

        # Ollama Models (typically local, high limits)
        'ollama-default': {
            LimitType.RPM: 1000,  # Increased - local models should be very permissive
            LimitType.TPM: 1000000,  # Increased
        },

        # Generic defaults for unknown models - MUCH more permissive
        'unknown-free': {
            LimitType.RPM: 100,  # Increased from 10
            LimitType.TPM: 500000,  # Increased from 10000
            LimitType.RPD: 2000  # Increased from 100
        },
        'unknown-paid': {
            LimitType.RPM: 500,  # Increased from 100
            LimitType.TPM: 2000000,  # Increased from 100000
            LimitType.RPD: 10000  # Increased from 1000
        }
    }

    @classmethod
    def initialize_model_limits(cls, model_key: str, is_paid: bool = False, provider: str = "unknown", lazy: bool = True):
        """
        Initialize rate limits for a model if not already configured.

        Args:
            model_key: The model identifier
            is_paid: Whether the user has a paid account
            provider: The provider name (for fallback defaults)
            lazy: If True, only initialize if the model has been rate limited before.
                  If False, initialize proactively (original behavior)
        """
        # Check if already configured
        status = enhanced_rate_limit_tracker.get_status_summary(model_key)
        if status["status"] != "no_limits_configured":
            logger.debug(f"Rate limits already configured for {model_key}")
            return

        # LAZY INITIALIZATION: Only initialize limits after first rate limit error
        # This prevents preemptive blocking of models that may have generous limits
        if lazy:
            logger.debug(f"Lazy initialization: skipping preemptive rate limits for {model_key}")
            return

        # Try to find specific configuration
        config = None

        # Exact match first
        if model_key in cls.MODEL_CONFIGS:
            config = cls.MODEL_CONFIGS[model_key]
        # Try with paid suffix if paid account
        elif is_paid and f"{model_key}-paid" in cls.MODEL_CONFIGS:
            config = cls.MODEL_CONFIGS[f"{model_key}-paid"]
        # Pattern matching for Gemini models
        elif provider.lower() in ["gemini", "google"]:
            if "flash" in model_key.lower():
                config = cls.MODEL_CONFIGS['gemini-flash']  # Use flash defaults
                logger.debug(f"Using Gemini Flash defaults for {model_key}")
            elif "pro" in model_key.lower():
                config = cls.MODEL_CONFIGS['gemini-pro']  # Use pro defaults
                logger.debug(f"Using Gemini Pro defaults for {model_key}")
            else:
                # Default to flash for unknown Gemini models
                config = cls.MODEL_CONFIGS['gemini-flash']
                logger.debug(f"Using Gemini Flash defaults for unknown Gemini model: {model_key}")
        # Pattern matching for Groq models
        elif provider.lower() in ["groq", "groqapi"]:
            config = cls.MODEL_CONFIGS['groq-default']
            logger.debug(f"Using Groq defaults for {model_key}")
        # Provider defaults
        elif provider.lower() == "ollama":
            config = cls.MODEL_CONFIGS['ollama-default']
        # Generic fallbacks
        elif is_paid:
            config = cls.MODEL_CONFIGS['unknown-paid']
        else:
            config = cls.MODEL_CONFIGS['unknown-free']

        if config:
            enhanced_rate_limit_tracker.set_model_limits(model_key, config)
            logger.info(f"Initialized rate limits for {model_key}: {config}")
        else:
            logger.warning(f"No rate limit configuration found for {model_key}")

    @classmethod
    def configure_gemini_defaults(cls, is_paid: bool = False):
        """Configure default limits for all Gemini models."""
        gemini_models = [
            'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.5-flash',
            'gemini-1.5-pro-latest', 'gemini-1.5-flash-latest',
            'gemini-pro', 'gemini-flash'  # Legacy names
        ]

        for model in gemini_models:
            cls.initialize_model_limits(model, is_paid=is_paid, provider="gemini")

    @classmethod
    def configure_ollama_defaults(cls):
        """Configure default limits for Ollama models."""
        # Ollama models are typically local, so we use generous defaults
        # Individual models will be configured as they're encountered
        pass

    @classmethod
    def auto_detect_tier(cls, model_key: str, error_message: str = "") -> bool:
        """
        Try to auto-detect if user has paid tier based on error messages.

        Args:
            model_key: The model that had an error
            error_message: The error message from the API

        Returns:
            True if paid tier is detected, False otherwise
        """
        # This is a heuristic approach - could be improved with more data
        error_lower = error_message.lower()

        # High RPM in error suggests paid tier
        if "300 rpm" in error_lower or "1000 rpm" in error_lower:
            return True

        # High TPM suggests paid tier
        if "4000000 tpm" in error_lower or "4m tpm" in error_lower:
            return True

        # High daily limits suggest paid tier
        if "10000 rpd" in error_lower or "10k rpd" in error_lower:
            return True

        # Default to free tier
        return False

    @classmethod
    def initialize_limits_from_error(cls, model_key: str, provider: str = "unknown", error_message: str = ""):
        """
        Initialize rate limits for a model that just hit a rate limit.
        This is called reactively after the first rate limit error.

        Args:
            model_key: The model that had an error
            provider: The provider name
            error_message: The error message from the API (for tier detection)
        """
        # Auto-detect tier if possible
        is_paid = cls.auto_detect_tier(model_key, error_message) if error_message else False

        # Force initialization with lazy=False to actually create the limits
        cls.initialize_model_limits(model_key, is_paid=is_paid, provider=provider, lazy=False)
        logger.info(f"Reactively initialized rate limits for {model_key} after rate limit error - paid tier: {is_paid}")

    @classmethod
    def update_limits_from_error(cls, model_key: str, error_message: str):
        """
        Update model limits based on information in error message.

        Args:
            model_key: The model that had an error
            error_message: The error message from the API
        """
        # Auto-detect tier and reconfigure if needed
        is_paid = cls.auto_detect_tier(model_key, error_message)

        # Get current status
        status = enhanced_rate_limit_tracker.get_status_summary(model_key)

        # If no limits configured or if we detected a tier upgrade
        if status["status"] == "no_limits_configured" or is_paid:
            cls.initialize_model_limits(model_key, is_paid=is_paid, provider="gemini", lazy=False)
            logger.info(f"Updated {model_key} limits based on error - paid tier: {is_paid}")

# Initialize common models on import
def initialize_common_models():
    """Initialize rate limits for commonly used models."""
    config = RateLimitConfig()

    # Configure Gemini defaults (assume free tier initially)
    config.configure_gemini_defaults(is_paid=False)

    # Configure Ollama defaults
    config.configure_ollama_defaults()

# Auto-initialize on import - but with lazy mode to prevent preemptive blocking
try:
    # Note: We skip auto-initialization to prevent preemptive blocking
    # Models will be initialized lazily when they first encounter rate limits
    logger.info("Rate limiting system ready - using lazy initialization to prevent preemptive blocking")
except Exception as e:
    logger.warning(f"Failed to initialize rate limiting system: {e}")