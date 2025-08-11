# CLI-Agent Testing Framework

## 🧪 Overview

CLI-Agent uses a comprehensive hierarchical testing framework that organizes tests by complexity and scope, from high-level integration tests down to detailed unit tests. The framework integrates both new hierarchical tests and all existing proven `test_*.py` scripts.

## 🚀 Quick Start

```bash
# Run default core tests (recommended)
python run_tests.py

# Run all tests including detailed unit tests
python run_tests.py --level 4 --verbose

# Run specific level tests
python run_tests.py --level 0  # Integration only
python run_tests.py --level 1  # + Tool tests
python run_tests.py --level 2  # + Core tests (default)
python run_tests.py --level 3  # + Shared utilities
python run_tests.py --level 4  # + Unit tests

# Run tests for specific tools
python run_tests.py --tools file_copier podcast_generator

# Quick essential tests
python run_tests.py --quick
```

## 📊 Test Hierarchy

### Level 0: Integration Tests
**Purpose**: Verify tools work together and share infrastructure properly

- **Cross-tool imports**: All tools can access core and shared modules
- **Path resolution**: PathResolver works consistently across all components
- **Shared infrastructure**: Core and shared directories are properly accessible

### Level 1: Tool Tests  
**Purpose**: Test individual tool functionality in isolation

- **File Copier**: Smart pasting, file discovery, GUI functionality
- **Podcast Generator**: Audio processing, content generation
- **Main CLI Agent**: Core agent functionality

### Level 2: Core Infrastructure Tests (Default Level)
**Purpose**: Test essential core systems that everything depends on

- **Core imports**: LlmRouter, Chat, Role, AIStrengths, globals
- **LLM selection**: Multi-provider routing system
- **Configuration**: Globals and settings management
- **Dependencies**: System dependency checking
- **Summary mode**: Debug output filtering

### Level 3: Shared Utilities Tests
**Purpose**: Test shared utilities and helper functions

- **PathResolver**: Directory and import path resolution
- **Shared modules**: Common utilities accessible to all tools
- **Import structure**: Proper module organization

### Level 4: Unit Tests
**Purpose**: Detailed testing of specific components and edge cases

- **Core LLM functionality**: LLM selection and routing systems
- **Provider systems**: Ollama caching, host tracking, metadata
- **Utility components**: Image viewing, architecture generation
- **Code quality**: Type hints verification and code standards
- **Component details**: Specific function and class testing

## 🔧 Adding New Tests

### Adding Hierarchical Tests

1. **Choose the appropriate level** based on what you're testing
2. **Add test method** to the relevant section in `run_tests.py`
3. **Follow the pattern**:

```python
async def _run_your_level_tests(self, specific_tools: Optional[List[str]] = None):
    """Level X: Your test category"""
    self.log(f"\n🔧 Running Your Tests (Level X)", "info")
    
    await self._run_test(
        "your_test_name",
        TestLevel.YOUR_LEVEL,
        self._test_your_functionality
    )

def _test_your_functionality(self) -> Dict[str, Any]:
    """Test your specific functionality"""
    # Your test implementation
    assert condition, "Error message"
    return {"passed": True, "details": "Test description"}
```

### Adding Test Scripts

1. **Create your test script** as `test_yourfeature.py` in the root directory
2. **Follow the established pattern**:

```python
#!/usr/bin/env python3
"""
Test for your feature functionality.
"""

def test_your_feature():
    """Test that your feature works correctly."""
    print("🧪 Testing Your Feature")
    print("=" * 50)
    
    try:
        # Your test implementation
        print("✅ Test passed")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if test_your_feature():
        sys.exit(0)
    else:
        sys.exit(1)
```

3. **Add to the test runner** by updating the appropriate level in `run_tests.py`:

```python
# For Level 2 (Core) tests
core_tests = [
    ("test_dependencies", "Dependency system verification"),
    ("test_summary_mode", "Summary mode core functionality"),
    ("test_yourfeature", "Your feature description"),  # Add this line
]

# For Level 4 (Unit) tests  
unit_tests = [
    ("test_core_functionality", "Core LLM functionality"),
    # ... existing tests ...
    ("test_yourfeature", "Your feature description"),  # Add this line
]
```

## 📈 Test Results Interpretation

### Success Indicators
- ✅ **Green checkmarks**: Test passed
- **High success rate**: >90% indicates healthy system
- **Fast execution**: Most tests should complete in <1 second

### Warning Indicators  
- ⚠️ **Yellow warnings**: Partial failures (some sub-tests failed)
- **Medium success rate**: 70-90% indicates issues need attention
- **Slow execution**: Individual tests taking >10 seconds

### Failure Indicators
- ❌ **Red X marks**: Test completely failed
- **Low success rate**: <70% indicates serious problems
- **Timeouts**: Tests hanging or taking >60 seconds

## 🔍 Current Test Coverage

### ✅ Working Tests (22/26 passing)

**Integration Tests (3/3)**
- Path resolver integration
- Cross-tool imports  
- Shared infrastructure

**Tool Tests (6/6)**
- File copier structure and imports
- Podcast generator functionality
- Main CLI agent components

**Core Tests (4/5)**
- Core imports and LLM router
- Globals configuration
- Summary mode functionality
- ⚠️ Dependencies (2/3 sub-tests pass)

**Shared Tests (2/2)**
- PathResolver functionality
- Shared imports structure

**Unit Tests (7/10)**
- ✅ Core LLM selection and functionality
- ✅ Ollama provider caching system
- ✅ Host tracking consistency
- ✅ Enhanced metadata and vector DB
- ✅ Interactive LLM selection interface
- ✅ Code quality and type verification
- ❌ Image viewing utility (missing API keys)
- ❌ Architecture utility (implementation incomplete)

### 🚧 Tests Needing Attention

1. **Dependency checking**: Partial failure (2/3 tests pass) - some app startup checks failing
2. **Image viewing utility**: Missing OpenAI API key for image analysis
3. **Architecture utility**: Test implementation needs refinement

## 🛠️ Maintenance

### Regular Testing Schedule
- **Before commits**: Run `python run_tests.py --level 2`
- **Before releases**: Run `python run_tests.py --level 4 --verbose`
- **Daily CI**: Run `python run_tests.py --level 3`

### Adding New Components
1. Add integration tests first (Level 0)
2. Add component-specific tests (Level 1-3)
3. Add detailed unit tests (Level 4)
4. Update this documentation

### Performance Monitoring
- Track test execution times
- Monitor success rates over time
- Alert on new failures or significant slowdowns

## 📚 Best Practices

### Test Design
- **Fast feedback**: Most tests should complete in <5 seconds
- **Clear failures**: Descriptive error messages and assertions
- **Isolated tests**: Each test should be independent
- **Comprehensive coverage**: Test both success and failure cases

### Code Organization
- **Hierarchical structure**: Organize by complexity and scope
- **Consistent patterns**: Follow established test patterns
- **Clear naming**: Use descriptive test and function names
- **Documentation**: Comment complex test logic

### Continuous Integration
- **Automated runs**: Integrate with git hooks or CI/CD
- **Parallel execution**: Run independent tests concurrently
- **Environment consistency**: Use consistent Python and dependency versions
- **Result archiving**: Store test results for trend analysis