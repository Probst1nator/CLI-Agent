"""
Registry of supported HuggingFace models for local inference.

Each model entry contains:
- model_id: Full HuggingFace repository ID (model_key is auto-derived from this)
- context_window: Maximum context length in tokens
- capabilities: List of model capabilities (vision, uncensored, etc.)
- requires_auth: Whether HF_TOKEN is needed (for gated models)
- trust_remote_code: Whether custom code execution is required
- description: Brief description of the model

Note: model_key is automatically derived as the text after the last "/" in model_id
"""

from core.lm_capabilities import LMCapabilities


def get_model_key_from_id(model_id: str) -> str:
    """
    Derive model key from model ID by taking text after last "/".

    Args:
        model_id: Full HuggingFace model ID (e.g., "moonshotai/Moonlight-16B-A3B-Instruct")

    Returns:
        str: Model key (e.g., "Moonlight-16B-A3B-Instruct")
    """
    return model_id.split("/")[-1]


# Registry of supported models
SUPPORTED_MODELS = [
    {
        "model_id": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "model_key": "Kimi-Linear-48B-A3B-Instruct",
        "context_window": 8000,
        "capabilities": [LMCapabilities.LOCAL],
        "requires_auth": False,
        "trust_remote_code": True,
        "description": "Kimi 48B Linear MoE model (3B active params)",
    },
    {
        "model_id": "moonshotai/Moonlight-16B-A3B-Instruct",
        "model_key": "Moonlight-16B-A3B-Instruct",
        "context_window": 8000,
        "capabilities": [LMCapabilities.LOCAL],
        "requires_auth": False,
        "trust_remote_code": True,
        "description": "Moonlight 16B MoE model (3B active params) - Trained with Muon optimizer",
    },
]


def get_model_info(model_key: str = None) -> dict | list[dict]:
    """
    Get information about a specific model or all models.

    Args:
        model_key: Optional model key to look up. If None, returns all models.

    Returns:
        dict: Model information (with model_key added) if model_key provided
        list[dict]: List of all models (with model_keys added) if model_key is None
    """
    if model_key is None:
        # Return all models with model_key added
        return [
            {**model, "model_key": get_model_key_from_id(model["model_id"])}
            for model in SUPPORTED_MODELS
        ]

    # Try exact match with derived model_key
    for model in SUPPORTED_MODELS:
        derived_key = get_model_key_from_id(model["model_id"])
        if derived_key == model_key:
            return {**model, "model_key": derived_key}

    # Try matching by model_id
    for model in SUPPORTED_MODELS:
        if model["model_id"] == model_key:
            return {**model, "model_key": get_model_key_from_id(model["model_id"])}

    # Try case-insensitive partial match
    model_key_lower = model_key.lower()
    for model in SUPPORTED_MODELS:
        derived_key = get_model_key_from_id(model["model_id"])
        if model_key_lower in derived_key.lower():
            return {**model, "model_key": derived_key}

    return None


def get_model_keys() -> list[str]:
    """
    Get list of all available model keys (derived from model IDs).

    Returns:
        list[str]: List of model keys
    """
    return [get_model_key_from_id(model["model_id"]) for model in SUPPORTED_MODELS]


def get_model_ids() -> list[str]:
    """
    Get list of all HuggingFace model IDs.

    Returns:
        list[str]: List of model IDs
    """
    return [model["model_id"] for model in SUPPORTED_MODELS]


def add_custom_model(
    model_id: str,
    context_window: int = 8192,
    capabilities: list = None,
    requires_auth: bool = False,
    trust_remote_code: bool = True,
    description: str = "",
):
    """
    Add a custom model to the registry at runtime.
    Model key will be auto-derived from model_id.

    Args:
        model_id: HuggingFace model ID (e.g., "org/model-name")
        context_window: Maximum context length
        capabilities: List of LMCapabilities
        requires_auth: Whether HF_TOKEN is needed
        trust_remote_code: Whether custom code is needed
        description: Model description

    Returns:
        bool: True if added, False if already exists
    """
    if capabilities is None:
        capabilities = [LMCapabilities.LOCAL]

    custom_model = {
        "model_id": model_id,
        "context_window": context_window,
        "capabilities": capabilities,
        "requires_auth": requires_auth,
        "trust_remote_code": trust_remote_code,
        "description": description,
    }

    # Check if already exists (by model_id or derived key)
    model_key = get_model_key_from_id(model_id)
    if get_model_info(model_key) is None:
        SUPPORTED_MODELS.append(custom_model)
        return True
    return False
