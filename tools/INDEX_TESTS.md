# Tests Index

This document provides a comprehensive listing of all test files in the CLI-Agent system, organized by category and purpose.

## 📊 Test Summary

**Total Test Files:** 26
**Test Status:** ✅ 24/24 Core Tests Passing (100%)

## 🗂️ Test Organization

### Core Infrastructure Tests

#### `/tests/core/`
- **`test_model_discovery.py`** - LLM model discovery and provider integration
  - API key detection
  - Discovery caching mechanisms
  - Error handling for provider failures
  - Google, Groq, Ollama model fetching
  - Parallel discovery progress tracking

- **`test_llm_router_local_mode.py`** - Local LLM routing and model management
  - Fallback model list filtering
  - Local completion generation
  - Model capability checking with local flag filtering
  - Ollama model detection

- **`test_piper_interface.py`** - Piper TTS interface functionality
  - Voice selection and audio generation
  - Quality verification testing

- **`test_xml_tool_routing.py`** - XML tool routing and validation (Fixed malformed XML routing)
  - Valid XML block detection and interruption
  - Malformed XML rejection testing
  - Edge cases: nested structures, special characters
  - Performance impact assessment
  - Regression prevention for existing functionality

### Tool-Specific Tests

#### `/tests/tools/`
- **`test_editfile.py`** - File editing functionality
  - Find/replace operations
  - Line-specific editing modes
  - Error handling and edge cases
  - Special character support

- **`test_writefile.py`** - File writing operations
  - File creation and overwriting
  - Directory auto-creation
  - UTF-8 encoding support
  - Error handling for invalid paths

- **`test_readfile.py`** - File reading with advanced features
  - Basic file reading
  - Line range specification (XML format)
  - Large file truncation handling
  - Image file support
  - Error handling for missing files

- **`test_insertline.py`** - Line insertion operations
  - Position-specific insertion (after/before)
  - End-of-file appending (default)
  - Newline handling and formatting
  - Line number validation

- **`test_deletelines.py`** - Line deletion operations
  - Single line deletion
  - Line range deletion
  - Content validation and output formatting
  - Error handling for invalid ranges

- **`test_search.py`** - Recursive file searching
  - Pattern matching and regex support
  - Path specification and defaults
  - Result limits and formatting
  - Error handling for invalid patterns

- **`test_todos.py`** - Task management functionality
  - Markdown task list persistence
  - Task state management
  - File I/O operations with mocking

### Cross-Component Integration Tests

#### `/tests/` (Root Level)
- **`test_cross_tool.py`** - Inter-tool compatibility
  - Core infrastructure import verification
  - Shared utilities import testing
  - Path resolver consistency checks

- **`test_path_resolver.py`** - Path resolution system
  - CLI-Agent root directory detection
  - Core, shared, and tools path resolution
  - Import system setup validation

- **`test_type_hints.py`** - Code quality enforcement
  - Generic Dict usage detection (improved logic)
  - Type hint compliance checking
  - Import statement validation
  - Current status: Core 66.7% clean, Tools 94.1% clean

### Audio & Media Processing Tests

#### `/tests/` (Audio Subsystem)
- **`test_audio_transcription.py`** - Speech-to-text functionality
  - Device selection and calibration
  - German language support testing
  - Scientific vocabulary handling
  - Known issues: Faster-Whisper tiny model limitations

- **`test_voice_pipeline.py`** - Voice processing pipeline
  - Synthetic audio generation with espeak TTS
  - Transcription accuracy testing
  - Multi-language support (German/English)
  - Voice type variations (12 test files)

- **`test_german_transcription_pipeline.py`** - German-specific transcription
  - Scientific vocabulary focus
  - Faster-Whisper model comparison (tiny/base/small/medium)
  - Real German audio test cases

- **`test_piper_transcription_pipeline.py`** - Piper TTS integration testing

### File Management & Utilities Tests

#### `/tests/` (File Tools)
- **`test_file_copier.py`** - File copying and management
  - Intelligent path resolution
  - Duplicate detection algorithms
  - Smart paste functionality

- **`test_smart_paster.py`** - Clipboard content analysis
  - Content type detection
  - Processing suggestions
  - Integration with file management

### Application-Specific Tests

#### `/tests/` (External Integrations)
- **`test_homeassistant.py`** - Home Assistant integration
  - Device control and automation
  - IoT interaction testing

- **`test_homeassistant_simple.py`** - Simplified Home Assistant tests
  - Basic connectivity and commands

- **`test_main_local_mode.py`** - Main application local mode
  - Local LLM functionality
  - Offline operation testing

## 🏃 Running Tests

### Full Test Suite
```bash
python test.py --verbose
```

### Quick Tests (< 30s)
```bash
python test.py --quick
```

### Category-Specific Tests
```bash
# Core functionality only
python test.py --core

# All tests including autonomous
python test.py --full

# Unit tests only
python test.py --unit
```

### Individual Test Files
```bash
python tests/tools/test_editfile.py
python tests/core/test_xml_tool_routing.py
```

## 📈 Test Coverage Areas

### ✅ Well-Covered Areas
- **File Operations**: Comprehensive coverage of all CRUD operations
- **Tool Integration**: Cross-tool communication and shared resources
- **Path Resolution**: Complete coverage of import and path systems
- **LLM Integration**: Model discovery, routing, and local mode operation
- **Audio Processing**: Multi-language transcription and TTS

### 🔄 Areas for Improvement
- **Type Hints**: Core modules need improvement (currently 66.7% compliant)
- **Error Handling**: More edge case testing needed
- **Performance Testing**: Limited performance benchmarks
- **Integration Testing**: More end-to-end workflow testing

### 📋 Test Quality Standards

#### Current Metrics
- **Core Tests**: 100% passing (24/24)
- **Type Hint Compliance**: Tools 94.1%, Core 66.7%
- **Test Coverage**: High for file operations, moderate for complex workflows

#### Quality Goals
- **Type Hints**: Target 95% compliance across all modules
- **Test Coverage**: Maintain 100% core functionality coverage
- **Performance**: All tests complete within reasonable time limits
- **Reliability**: Zero flaky tests, consistent results

## 🔧 Test Infrastructure

### Test Framework
- **Base**: Python unittest and custom test discovery
- **Mocking**: unittest.mock for isolated testing
- **File I/O**: Temporary directories and file fixtures
- **Async**: asyncio support for concurrent operations

### Test Data
- **Audio Files**: Real-world samples in `tests/audio/`
- **Fixtures**: Temporary test files and directories
- **Mock Data**: Simulated API responses and file contents

### Continuous Integration
- Automated test discovery and execution
- Hierarchical test levels (0-4: integration to unit)
- Performance monitoring and time limits
- Detailed reporting with actionable feedback

## 🚀 Future Enhancements

### Planned Additions
1. **Performance Benchmarks**: Response time and memory usage tests
2. **Load Testing**: Concurrent operation stress tests
3. **Security Testing**: Input validation and sanitization tests
4. **Documentation Tests**: API documentation accuracy verification

### Quality Improvements
1. **Type Hint Fixes**: Address remaining Dict usage in core modules
2. **Test Parallelization**: Faster test suite execution
3. **Better Mocking**: More realistic test scenarios
4. **Coverage Reports**: Detailed coverage analysis and gaps identification

---

*Generated: Tests indexed and verified for CLI-Agent v1.0*
*Status: All core tests passing, infrastructure stable*