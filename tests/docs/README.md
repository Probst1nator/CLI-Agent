# Testing Documentation

Comprehensive testing infrastructure for CLI-Agent with hierarchical test levels and automated test runners.

## Architecture Overview

CLI-Agent uses a **hierarchical testing framework** with 5 levels (0-4):
- **Level 0**: Integration and system tests
- **Level 1**: Component integration tests
- **Level 2**: Core functionality tests (quick suite)
- **Level 3**: Unit tests for utilities
- **Level 4**: Fine-grained unit tests

## Directory Structure

```
tests/
├── test.py              # User-facing test entrypoint (recommended)
├── run_tests.py         # Core test runner infrastructure
├── test_gui_components.py  # GUI component tests
├── profiling.py         # Import profiling for optimization
├── core/                # Core infrastructure tests
│   ├── test_model_discovery.py
│   └── test_piper_interface.py
├── shared/              # Shared utilities tests
│   ├── test_path_resolver.py
│   ├── test_project_storage.py
│   ├── test_prompt_regenerator.py
│   └── test_chat_serialization.py
├── tools/               # Tool-specific tests
│   ├── test_file_copier.py
│   └── test_smart_paster.py
├── frontend/            # Frontend tests
├── audio/               # Audio system tests
│   ├── test_audio_transcription.py
│   ├── test_voice_pipeline.py
│   ├── test_german_transcription_pipeline.py
│   └── pipeline/        # Generated test audio files
└── README.md            # This file
```

## Test Levels

### Level 0: Integration Tests

**Scope**: End-to-end system integration
**Examples**:
- Full conversation flows
- Multi-tool workflows
- Cross-component integration

**When to add**:
- Testing complete user journeys
- Validating system-wide behavior
- Integration across multiple components

### Level 1: Component Integration

**Scope**: Component interaction tests
**Examples**:
- Agent + LLM Router integration
- API + Frontend interaction
- Tool + Permission system

**When to add**:
- Testing component boundaries
- Validating interfaces between systems
- Integration within subsystems

### Level 2: Core Functionality (Quick Suite)

**Scope**: Essential functionality validation
**Examples**:
- LLM model discovery
- Chat serialization
- Path resolution
- Project storage

**Quick test mode**:
```bash
python test.py --quick
# or
python run_tests.py --quick
```

**When to add**:
- Core business logic
- Critical path functionality
- Regression prevention

### Level 3: Utility Tests

**Scope**: Utility function validation
**Examples**:
- String manipulation
- File operations
- Data processing

**When to add**:
- Testing helper functions
- Utility validation
- Edge case coverage

### Level 4: Fine-Grained Unit Tests

**Scope**: Individual function tests
**Examples**:
- Single function validation
- Edge cases
- Error handling

**When to add**:
- Complex function logic
- Error condition validation
- Boundary testing

## Test Runners

### test.py (Recommended)

**User-facing entrypoint** with simple interface:

```bash
python test.py              # Run all tests
python test.py --quick      # Run quick suite (Level 2+)
python test.py --full       # Run full suite (all levels)
python test.py --verbose    # Detailed output
```

**Features**:
- Clear, actionable results
- User-friendly output
- Orchestrates underlying infrastructure
- Primary command for developers

### run_tests.py

**Core test runner infrastructure**:

```bash
python run_tests.py --quick
python run_tests.py --full
python run_tests.py --verbose
```

**Features**:
- Test discovery (scans for `test_*` functions)
- Hierarchical execution by level
- Result aggregation
- Low-level mechanics

**Note**: Developers typically use `test.py`, not this directly.

## Running Tests

### Quick Suite (Most Common)

Run core tests only (fastest):
```bash
python test.py --quick
```

**Runs**: Level 2+ tests
**Time**: ~10-30 seconds
**Use**: Regular development validation

### Full Suite

Run all tests including integration:
```bash
python test.py --full
```

**Runs**: All levels (0-4)
**Time**: Several minutes
**Use**: Pre-commit validation, CI/CD

### Specific Test File

Run individual test file:
```bash
python -m pytest tests/core/test_model_discovery.py -v
python -m pytest tests/shared/test_path_resolver.py -v
```

### Specific Test Function

Run single test:
```bash
python -m pytest tests/core/test_model_discovery.py::test_function_name -v
```

### With Output

Verbose output:
```bash
python test.py --verbose
python -m pytest tests/path/test_file.py -v --tb=short
```

## Test Organization

### Core Tests (`tests/core/`)

**LLM and infrastructure**:
- `test_model_discovery.py` - LLM provider discovery
- `test_piper_interface.py` - Piper TTS interface

**Coverage**:
- LLM routing
- Provider integration
- Model enumeration
- Global configuration

### Shared Tests (`tests/shared/`)

**Utilities validation**:
- `test_path_resolver.py` - Path resolution
- `test_project_storage.py` - Project data storage
- `test_prompt_regenerator.py` - Prompt generation
- `test_chat_serialization.py` - Chat save/load

**Coverage**:
- Cross-platform path handling
- Data persistence
- Utility functions

### Tool Tests (`tests/tools/`)

**Tool functionality**:
- `test_file_copier.py` - File copying
- `test_smart_paster.py` - Smart paste

**Coverage**:
- Tool interface compliance
- XML parsing
- Error handling
- Permission integration

See: `tests/tools/INDEX.md`

### GUI Tests

**GUI component validation**:
- `test_gui_components.py` - Widget and layout tests

**Coverage**:
- ChatDisplay, StatusPanel, InputArea
- Layout systems
- Command integration
- prompt_toolkit integration

**Note**: Tests without requiring terminal interface

### Audio Tests (`tests/audio/`)

**Speech and TTS validation**:

**Test files**:
- `test_audio_transcription.py` - Device selection, German language, scientific vocabulary
- `test_voice_pipeline.py` - Synthetic audio generation and transcription (12 test files)
- `test_german_transcription_pipeline.py` - German scientific vocabulary across model sizes

**Test audio**:
- `german_transcription_failure.wav` - Real-world failure case
- `pipeline/` - Generated test files (German/English × voice types)

**Known issues**:
- Faster-Whisper tiny model fails on German scientific terms
- See test files for model size comparisons

## Test Patterns

### Test Function Naming

```python
def test_feature_name():
    """Test that feature works correctly"""
    pass
```

### Test Structure

```python
def test_something():
    # Arrange
    setup_data = prepare_test_data()

    # Act
    result = function_under_test(setup_data)

    # Assert
    assert result == expected_value
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mock():
    with patch('module.external_api') as mock_api:
        mock_api.return_value = "mocked response"
        result = function_that_calls_api()
        assert result == "expected based on mock"
```

### Testing Async Functions

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

## Profiling

### Import Time Profiling

**Identify slow imports**:
```bash
python profiling.py
```

**Use case**:
- Optimize startup time
- Identify heavy dependencies
- Guide lazy loading decisions

**Output**:
- Module import times
- Cumulative times
- Bottleneck identification

## Continuous Integration

### Pre-Commit Validation

Recommended workflow:
```bash
# Before committing
python test.py --quick

# Before PR
python test.py --full
```

### CI/CD Integration

GitHub Actions / GitLab CI:
```yaml
test:
  script:
    - pip install -r requirements.txt
    - python test.py --full
```

## Test Data

### Test Fixtures

Reusable test data:
```python
@pytest.fixture
def sample_chat():
    return Chat()
```

### Test Files

**Location**: `tests/audio/`, `tests/fixtures/`

**Usage**: Real-world test cases
- Audio files for transcription testing
- Sample PDFs for extraction testing
- Example configs for validation

## Coverage

### Running with Coverage

```bash
pytest --cov=. --cov-report=html tests/
```

**Output**: `htmlcov/index.html`

### Coverage Goals

- **Core**: >80%
- **Shared**: >70%
- **Tools**: >60%
- **Integration**: Best effort

## Debugging Tests

### Verbose Output

```bash
python -m pytest tests/path/test_file.py -v --tb=long
```

### Print Debugging

```python
def test_something():
    result = function()
    print(f"Debug: result={result}")  # Shown with -v
    assert result == expected
```

### Breakpoint Debugging

```python
def test_something():
    result = function()
    import pdb; pdb.set_trace()  # Debugger
    assert result == expected
```

## Adding New Tests

### 1. Choose Test Level

Determine appropriate level (0-4) based on scope.

### 2. Create Test File

```bash
# For core tests
tests/core/test_new_feature.py

# For tool tests
tests/tools/test_new_tool.py
```

### 3. Write Test Functions

```python
def test_feature():
    """Test description"""
    # Test implementation
    pass
```

### 4. Run Tests

```bash
python -m pytest tests/path/test_new_feature.py -v
```

### 5. Add to Suite

Test automatically discovered if:
- File named `test_*.py`
- Function named `test_*`
- Located in `tests/` directory

## Best Practices

### 1. Test Independence

Each test should be independent:
- No shared state between tests
- Clean setup and teardown
- Idempotent execution

### 2. Clear Assertions

Use descriptive assertion messages:
```python
assert result == expected, f"Expected {expected}, got {result}"
```

### 3. Test Edge Cases

Cover boundary conditions:
- Empty input
- None values
- Maximum values
- Error conditions

### 4. Keep Tests Fast

- Mock external services
- Minimize I/O
- Use in-memory databases where possible

### 5. Maintainable Tests

- Clear test names
- Descriptive docstrings
- Avoid test duplication
- Refactor when needed

## Related Documentation

- Test architecture: `/tests/README.md`
- Tool tests: `/tests/tools/INDEX.md`
- Core tests: `/tests/core/INDEX.md`
- Shared tests: `/tests/shared/INDEX.md`
- Audio tests: `/tests/audio/README.md`
