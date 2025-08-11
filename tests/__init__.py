"""
CLI-Agent Comprehensive Test Suite

Hierarchical testing framework that provides multiple levels of testing
from high-level integration tests to detailed unit tests.

Test Levels:
- Level 0: Integration Tests (tools working together)
- Level 1: Tool Tests (each tool in isolation)  
- Level 2: Core Infrastructure Tests
- Level 3: Shared Utility Tests
- Level 4: Unit Tests (detailed function-level testing)
"""

__version__ = "1.0.0"
__all__ = ["run_tests"]