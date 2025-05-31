
from enum import Enum


class AIStrengths(Enum):
    """Enum class to represent AI model strengths."""
    UNCENSORED = 6
    REASONING = 5
    CODE = 4
    GUARD = 3
    GENERAL = 2
    SMALL = 1
    LOCAL = 8
    ONLINE = 9
    VISION = 10
    STRONG = 11