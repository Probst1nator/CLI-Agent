
from enum import Enum


class AIStrengths(Enum):
    """Enum class to represent AI model strengths."""
    UNCENSORED = 6
    REASONING = 5
    CODE = 4
    GUARD = 3
    GENERAL = 2
    FAST = 1
    LOCAL = 8
    VISION = 9
    BALANCED = 10