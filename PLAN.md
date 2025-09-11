# CLI-Agent Refactoring Plan: Phase 2 - Aggressive Function Extraction

## Overview
This document outlines Phase 2 of the CLI-Agent refactoring effort, targeting the massive 3,336-line main.py file for intelligent code extraction. The goal is to reduce main.py by ~1,500+ lines while preserving exact logic and enabling the transition to modular architecture.

## Phase 1 Results (Completed)
✅ **PDF Processing Consolidation**: Eliminated 150+ duplicate lines  
✅ **CLI Argument Parsing**: Extracted to `agent/core/cli_args.py`  
✅ **Configuration Utilities**: Extracted to `agent/core/config_utils.py`  
✅ **Graceful Shutdown**: Extracted to `shared/utils/shutdown_utils.py`  

**Phase 1 Impact**: ~300 lines removed, 4 new modules created

## Phase 2A Results (Completed)
✅ **UI/Display Utilities**: Extracted to `shared/utils/display_utils.py` (~300 lines)
✅ **Command System**: Extracted to `agent/core/command_system.py` (~90 lines)  
✅ **Logging Utilities**: Extracted to `shared/utils/logging_utils.py` (~112 lines)

**Phase 2A Impact**: ~500 lines removed, 3 new modules created
**Main.py Reduction**: Successfully reduced from 3,336 to 3,007 lines (**329 lines removed**)
**Total Progress**: ~800 lines extracted, 7 new modules, main.py imports working correctly

## Phase 2 Analysis: 6 Major Extraction Opportunities

### Current State
- **main.py**: 3,336 lines
- **Functions identified**: 27 functions + 5 classes
- **Extraction potential**: ~1,500+ lines (45% reduction)

## Extraction Targets

### 1. **UI/Display Utilities Module** 
**Target**: `shared/utils/display_utils.py`

**Functions to extract:**
- `colorize_model_name()` (38 lines) - Provider-based model coloring logic
- `colorize_model_list()` (3 lines) - List colorization wrapper  
- `print_startup_summary()` (200+ lines) - Comprehensive agent configuration display
- `show_unified_bindings_display()` (60+ lines) - Key bindings help display

**Complexity**: 🟢 Low - Self-contained display logic  
**Dependencies**: termcolor, colored  
**Impact**: ~300 lines removed

### 2. **Command System Module**
**Target**: `agent/core/command_system.py`

**Functions to extract:**
- `get_command_definitions()` (12 lines) - Dynamic command definition generation
- `get_command_descriptions()` (14 lines) - Command description mapping  
- `_build_command_map()` (11 lines) - Alias-to-canonical mapping
- `CommandCompleterWithDescriptions` class (50+ lines) - Prompt toolkit completer

**Complexity**: 🟡 Medium - Well-defined interfaces, some global state  
**Dependencies**: prompt_toolkit  
**Impact**: ~90 lines removed

### 3. **Tool Discovery & Documentation Module**
**Target**: `agent/core/tool_discovery.py`

**Functions to extract:**
- `_generate_tool_documentation()` (52 lines) - Dynamic tool discovery and doc generation
- `_check_getuserfiles_available()` (21 lines) - GUI availability checker
- Tool scanning and import logic

**Complexity**: 🟡 Medium - Global state manipulation (g.tool_classes)  
**Dependencies**: importlib, Path, logging  
**Impact**: ~80 lines removed

### 4. **Input Handling Module** 
**Target**: `agent/core/input_handlers.py`

**Functions to extract:**
- `handle_multiline_input()` (526+ lines) - Complex multiline input processing
- `get_user_input_with_bindings()` (313+ lines) - Main input loop with key bindings  
- `handle_screenshot_capture()` (59 lines) - Screenshot handling logic
- `handle_permission_flags()` (74 lines) - Permission flag processing
- `interactive_permission_manager()` (100+ lines) - Permission management UI

**Complexity**: 🔴 High - Complex async logic, extensive state management  
**Dependencies**: asyncio, prompt_toolkit, threading  
**Impact**: ~1,070 lines removed

### 5. **Monte Carlo Engine Module**
**Target**: `agent/monte_carlo/monte_carlo_engine.py`

**Functions to extract:**
- `generate_branch()` (100+ lines) - Branch generation logic
- `select_best_branch()` (221+ lines) - Branch selection and evaluation  
- `confirm_code_execution()` (234+ lines) - Execution confirmation logic
- Monte Carlo coordination functions

**Complexity**: 🔴 High - Complex async coordination, LLM interaction  
**Dependencies**: asyncio, Chat, LlmRouter, TextStreamPainter  
**Impact**: ~560 lines removed

### 6. **Logging & Formatting Module**
**Target**: `shared/utils/logging_utils.py`

**Classes to extract:**
- `ConsoleFormatter` class (18 lines) - Console log formatting with elapsed time
- `SimpleFormatter` class (5 lines) - Simple message formatting  
- `FileFormatter` class (15 lines) - File log formatting with ANSI stripping
- `setup_logging()` (74 lines) - Comprehensive logging configuration

**Complexity**: 🟢 Low - Well-isolated infrastructure  
**Dependencies**: logging, re, termcolor  
**Impact**: ~112 lines removed

## Implementation Strategy

### Phase 2A: Safe, High Impact (UI & Infrastructure)
**Priority 1**: Low complexity, high line reduction
1. Extract UI/Display utilities (`shared/utils/display_utils.py`)
2. Extract logging utilities (`shared/utils/logging_utils.py`) 
3. Extract command system (`agent/core/command_system.py`)
4. Update imports and test basic functionality

**Estimated Impact**: ~500 lines removed

### Phase 2B: Core Logic Extraction (Medium Complexity)
**Priority 2**: Medium complexity, significant impact
1. Extract tool discovery (`agent/core/tool_discovery.py`)
2. Careful handling of global state (g.tool_classes)
3. Extensive testing of tool discovery functionality

**Estimated Impact**: ~80 lines removed

### Phase 2C: Complex Async Logic (High Risk, High Reward)
**Priority 3**: Complex logic requiring careful preservation
1. Extract input handlers (`agent/core/input_handlers.py`)
2. Extract Monte Carlo engine (`agent/monte_carlo/monte_carlo_engine.py`)
3. Comprehensive testing of async workflows
4. Preserve exact branching and evaluation logic

**Estimated Impact**: ~1,630 lines removed

## Expected Results

### Before Phase 2
- **main.py**: 3,336 lines
- **Modules**: Basic structure

### After Phase 2 
- **main.py**: ~1,800 lines (46% reduction)
- **New modules**: 6 specialized modules
- **Functions extracted**: 20+ major functions  
- **Classes extracted**: 5+ classes
- **Architecture**: Clear separation of concerns

## Risk Mitigation

### Testing Strategy
1. **Unit tests** for each extracted module
2. **Integration tests** for async workflows  
3. **Regression tests** for Monte Carlo engine
4. **Import verification** across all entry points

### Rollback Plan
1. Git branching for each extraction phase
2. Incremental commits for rollback points
3. Preserve original main.py logic as reference

### Critical Preservation Points
1. **Exact async behavior** in input handlers
2. **Monte Carlo branching logic** unchanged
3. **Global state management** (g.tool_classes, etc.)
4. **Error handling** and exception flows

## Benefits

### Technical Benefits
- **Maintainability**: Single responsibility modules
- **Testability**: Isolated function testing  
- **Reusability**: Shared utilities across entry points
- **Architecture**: Clear modular boundaries
- **Performance**: Potential for lazy loading

### Strategic Benefits  
- **main_refactored.py evolution**: Enables clean architecture
- **Team development**: Parallel work on different modules
- **Code review**: Smaller, focused changes
- **Documentation**: Module-specific documentation

## Success Metrics
- [ ] main.py reduced to <2,000 lines
- [ ] 6 new specialized modules created
- [ ] All existing functionality preserved
- [ ] Import statements updated across codebase
- [ ] Tests passing for extracted functionality
- [ ] main_refactored.py independence from main.py

## Next Steps
1. Execute Phase 2A (safe extractions)
2. Validate functionality and update INDEX.md
3. Execute Phase 2B (tool discovery)  
4. Execute Phase 2C (complex async logic)
5. Final integration testing and documentation

---
*This plan maintains the prime directive: exact logic preservation while changing only code locations.*

---

# 🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION

## Current Blocking Issues (December 2025)

### 1. Visual Response Server Cleanup Issue ❌
**Status**: CRITICAL - Server closes immediately after content generation  
**Priority**: HIGH

**Problem**: Despite attempted fix, visual servers still register cleanup callbacks causing premature shutdown.

**Evidence**:
```
[VisualServer] Started successfully at http://127.0.0.1:7860
Registered cleanup callback: VisualServerSession.close
```

**Root Cause**: 
- `VisualServerSession(register_cleanup=False)` fix not working correctly
- Cleanup callback still being registered in `visualresponse._get_session()`
- `atexit.register(self._run_cleanup)` in globals.py triggering premature closure

**Files Requiring Fix**:
- `/home/prob/repos/CLI-Agent/tools/visualresponse.py` (lines 142, 223-232)
- `/home/prob/repos/CLI-Agent/core/globals.py` (lines 185-186, 610-638)
- `/home/prob/repos/CLI-Agent/main.py` (lines 1874-1894)

### 2. Bash Session EOF Critical Failure ❌
**Status**: CRITICAL - All bash commands failing  
**Priority**: CRITICAL

**Problem**: Every bash command fails with EOF (End of File) errors during session initialization.

**Evidence**:
```
Failed to spawn bash: End Of File (EOF). Exception style platform.
before (last 100 chars): '"\r\nstty -echo; set +H\r\n'
after: <class 'pexpect.exceptions.EOF'>
```

**Root Cause**: 
- `pexpect` cannot establish pseudo-terminal with bash
- Shell initialization fails during `stty -echo; set +H` command
- Possible environment/permissions conflict with terminal handling

**Files Requiring Fix**:
- `/home/prob/repos/CLI-Agent/tools/bash.py` (lines 58-75, BashSession.__init__)
- Potentially `/home/prob/repos/CLI-Agent/agent/core/input_handlers.py` (dependent on bash fix)

## 🔧 Immediate Action Plan

### Phase 1: Debug Visual Response Cleanup
**Files**: `tools/visualresponse.py`, `core/globals.py`

1. **Verify register_cleanup parameter is working**:
   - Add debug logging in `VisualServerSession.__init__()` 
   - Confirm `register_cleanup=False` is being respected
   - Check why cleanup is still being registered

2. **Implement conditional cleanup bypass**:
   - Modify cleanup registration to be truly optional
   - Add flags to prevent automatic cleanup for visual servers

### Phase 2: Resolve Bash EOF Issue  
**Files**: `tools/bash.py`

1. **Diagnose EOF root cause**:
   - Add detailed error logging during bash initialization
   - Test alternative shell parameters and timeout values
   - Check environment variables and terminal settings

2. **Implement bash fallback mechanism**:
   - Add retry logic for session establishment  
   - Test with different pexpect configurations
   - Handle EOF gracefully without complete failure

### Phase 3: Verify Integration
**Files**: All modified files

1. **Test visual response persistence**:
   - Ensure servers stay alive after content generation
   - Verify no cleanup callbacks are registered
   - Test manual cleanup on user exit

2. **Test bash functionality**:
   - Verify basic commands execute without EOF
   - Test session persistence and command chaining
   - Confirm timeout handling works correctly

## 📊 Critical Files Summary

| File | Issue | Priority | Lines |
|------|-------|----------|-------|
| `tools/visualresponse.py` | Cleanup still registered | HIGH | 142, 223-232 |
| `core/globals.py` | Automatic cleanup conflict | HIGH | 185-186, 610-638 |  
| `main.py` | Exit logic needs visual fix | HIGH | 1874-1894 |
| `tools/bash.py` | EOF during initialization | CRITICAL | 58-75 |
| `agent/core/input_handlers.py` | Dependent on bash fix | MEDIUM | TBD |

## 🎯 Success Criteria

### Visual Response Fix:
- [ ] No "Registered cleanup callback" messages in logs
- [ ] Visual servers remain active after content generation
- [ ] Manual cleanup works on explicit user exit (Ctrl+C)
- [ ] Multiple visual responses work in sequence

### Bash Session Fix:
- [ ] `bash._get_session()` succeeds without EOF errors  
- [ ] Basic commands like `ls`, `pwd` execute successfully
- [ ] Session prompt regex matching functions correctly
- [ ] Timeout handling works without crashes

## 💡 Root Cause Analysis

**Visual Server**: The attempted fix using `VisualServerSession(register_cleanup=False)` is not preventing cleanup registration. The system is still calling `g.register_cleanup()` somewhere in the code path.

**Bash EOF**: The pexpect library cannot establish a proper pseudo-terminal session with bash, failing specifically during shell configuration commands. This suggests either environmental incompatibility or permission issues with terminal access.

Both issues are blocking core functionality and require immediate resolution before any further development can proceed.