"""
CLI-Agent Test Runner Infrastructure

Hierarchical test framework for running tests at different levels
from integration tests to unit tests.
"""

import asyncio
import sys
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional
from termcolor import colored

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestLevel(Enum):
    """Test hierarchy levels from integration to unit tests"""
    INTEGRATION = 0  # End-to-end tests
    TOOLS = 1        # Tool-specific tests
    CORE = 2         # Core infrastructure tests (default)
    SHARED = 3       # Shared utilities tests
    UNIT = 4         # Detailed unit tests


class TestResult:
    """Result of running a single test"""
    def __init__(self, name: str, passed: bool, duration: float, error: Optional[str] = None):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.error = error


class TestRunner:
    """Main test runner that discovers and executes tests"""
    
    def __init__(self, max_level: TestLevel = TestLevel.CORE, verbose: bool = False, quick: bool = False):
        self.max_level = max_level
        self.verbose = verbose
        self.quick = quick
        self.results: List[TestResult] = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests up to the specified level"""
        test_functions = self._discover_tests()
        
        if self.verbose:
            print(colored(f"🔍 Discovered {len(test_functions)} tests", "blue"))
            print()
        
        total_tests = len(test_functions)
        passed_tests = 0
        
        for i, (test_name, test_func, test_level) in enumerate(test_functions, 1):
            if self.verbose:
                print(colored(f"[{i}/{total_tests}] Running {test_name}...", "blue"))
            else:
                # Progress indicator
                progress = f"[{i}/{total_tests}]"
                print(f"\r{progress} Running tests... ", end="", flush=True)
            
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                
                duration = time.time() - start_time
                
                if result is True:
                    passed_tests += 1
                    self.results.append(TestResult(test_name, True, duration))
                    if self.verbose:
                        print(colored(f"  ✅ {test_name} ({duration:.2f}s)", "green"))
                elif result is False:
                    self.results.append(TestResult(test_name, False, duration, "Test returned False"))
                    if self.verbose:
                        print(colored(f"  ❌ {test_name} ({duration:.2f}s)", "red"))
                else:
                    # Handle tests that return other values (like test count)
                    passed_tests += 1
                    self.results.append(TestResult(test_name, True, duration))
                    if self.verbose:
                        print(colored(f"  ✅ {test_name} ({duration:.2f}s) - {result}", "green"))
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                self.results.append(TestResult(test_name, False, duration, error_msg))
                if self.verbose:
                    print(colored(f"  ❌ {test_name} ({duration:.2f}s) - {error_msg}", "red"))
                    if self.verbose and hasattr(e, '__traceback__'):
                        print(colored("    " + traceback.format_exc().replace('\n', '\n    '), "red"))
        
        if not self.verbose:
            print()  # Clear the progress line
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'results': self.results
        }
    
    def _discover_tests(self) -> List[tuple]:
        """Discover all test functions based on the current level"""
        test_functions = []
        
        # Import and scan test modules based on level
        if self.max_level.value <= TestLevel.INTEGRATION.value:
            # Integration tests (Level 0)
            test_functions.extend(self._scan_module_for_tests("tests.test_cross_tool", TestLevel.INTEGRATION))
            test_functions.extend(self._scan_module_for_tests("tests.test_main_local_mode", TestLevel.INTEGRATION))
        
        if self.max_level.value <= TestLevel.TOOLS.value:
            # Tool tests (Level 1)
            test_functions.extend(self._scan_module_for_tests("tests.test_file_copier", TestLevel.TOOLS))
            test_functions.extend(self._scan_module_for_tests("tests.test_smart_paster", TestLevel.TOOLS))
        
        if self.max_level.value <= TestLevel.CORE.value:
            # Core tests (Level 2) - LLM and model discovery tests
            test_functions.extend(self._scan_module_for_tests("tests.core.test_model_discovery", TestLevel.CORE))
            test_functions.extend(self._scan_module_for_tests("tests.core.test_llm_router_local_mode", TestLevel.CORE))
            test_functions.extend(self._scan_module_for_tests("tests.test_cross_tool", TestLevel.CORE))
        
        if self.max_level.value <= TestLevel.SHARED.value:
            # Shared tests (Level 3)
            test_functions.extend(self._scan_module_for_tests("tests.test_path_resolver", TestLevel.SHARED))
        
        if self.max_level.value <= TestLevel.UNIT.value:
            # Unit tests (Level 4)
            test_functions.extend(self._scan_module_for_tests("tests.test_type_hints", TestLevel.UNIT))
        
        return test_functions
    
    def _scan_module_for_tests(self, module_name: str, level: TestLevel) -> List[tuple]:
        """Scan a module for test functions"""
        try:
            module = __import__(module_name, fromlist=[''])
            test_functions = []
            
            for name in dir(module):
                if name.startswith('test_'):
                    func = getattr(module, name)
                    if callable(func):
                        test_functions.append((f"{module_name}.{name}", func, level))
            
            return test_functions
            
        except ImportError as e:
            if self.verbose:
                print(colored(f"⚠️  Could not import {module_name}: {e}", "yellow"))
            return []
        except Exception as e:
            if self.verbose:
                print(colored(f"❌ Error scanning {module_name}: {e}", "red"))
            return []


if __name__ == "__main__":
    # Simple test of the runner itself
    async def main():
        runner = TestRunner(max_level=TestLevel.CORE, verbose=True)
        summary = await runner.run_all_tests()
        print(f"\nTest Summary: {summary}")
    
    asyncio.run(main())