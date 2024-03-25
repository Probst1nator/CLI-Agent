from enum import Enum


class AvailableModels(Enum):
    EXPERT = "mistral"
    FAST = "orca2"
    CODING = "codellama:7b-code"
    
    # EXPERT = "mistral"
    # FAST = "phi"
    # CODING = "magicoder"