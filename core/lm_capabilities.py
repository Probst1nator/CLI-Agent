from enum import Enum


class LMCapabilities(Enum):
    """Enum class to represent AI model capabilities."""

    UNCENSORED = 1
    VISION = 2
    LOCAL = 3
    LITE = 4
    PRO = 5
