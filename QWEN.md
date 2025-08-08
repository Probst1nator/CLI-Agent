# CLI-Agent Framework Documentation & Task Tracking

## CURRENT ACTIVE TASKS

### 🔴 HIGH PRIORITY - IN PROGRESS
- [ ] **--fast Argument Model Filtering Bug** (Claude Active)
  - **Issue**: gemini-2.5-flash allowed when --fast used, should only allow "SMALL" strength models
  - **Location**: `main.py` model selection logic around line with `available_models = [m.model_key for m in LlmRouter.get_models(force_local=g.FORCE_LOCAL)]`
  - **Root Cause**: No strength filtering when `g.FORCE_FAST=True`
  - **Action Items**:
    - [ ] Investigate `LlmRouter.get_models()` implementation
    - [ ] Check model metadata for strength classification
    - [ ] Add strength filtering logic
    - [ ] Test with gemini models to ensure lite version selected

### 🟡 HIGH PRIORITY - QUEUED
- [ ] **MCT Infinite Retry Loop Fix** (Gemini Next)
  - **Location**: `main.py` lines ~1260-1303
  - **Symptoms**: Endless "[MCT Branching X/Y] generating response..." messages
  - **Action Items**:
    - [ ] Add retry counter and maximum retry limit
    - [ ] Implement timeout mechanism
    - [ ] Better error handling for failed branches
    - [ ] Remove failed models from retry attempts

### 🟢 MEDIUM PRIORITY - BACKLOG
- [ ] **Memory Management Enhancement**
  - **Risk**: Long conversations accumulate excessive memory
  - [ ] Implement conversation compaction/truncation
  - [ ] Chat history optimization
  - [ ] Response buffer management

- [ ] **Vector Database Performance**
  - [ ] Optimize similarity calculations
  - [ ] Implement caching mechanisms
  - [ ] Profile and improve search speed

## FRAMEWORK ARCHITECTURE OVERVIEW

### Core Components
1. **Entry Point**: `main.py` - Main application orchestration
2. **LLM Router**: `py_classes/cls_llm_router.py` - Multi-provider LLM routing
3. **Chat System**: `py_classes/cls_chat.py` - Conversation management
4. **Utility Manager**: `py_classes/cls_util_manager.py` - Tool management with vector search
5. **Computational Notebook**: `py_classes/cls_computational_notebook.py` - Code execution
6. **Vector Database**: `py_classes/cls_vector_db.py` - Semantic tool matching

### Key Features
- Multi-model reasoning with Monte Carlo Tree Search (MCT)
- Dynamic utility suggestions via vector embeddings
- Code execution with safety guards
- Real-time response streaming
- Provider-agnostic LLM integration (OpenAI, Anthropic, Google, Ollama)

### Configuration System
- **Global Variables**: `py_classes/globals.py`
  - `FORCE_FAST`: Performance mode (fast models only)
  - `FORCE_STRONG`: Power mode (strong models)
  - `SELECTED_LLMS`: Manual model selection
  - `FORCE_LOCAL`: Local-only model preference

## AGENT COORDINATION PROTOCOL

### Current Session Assignment
- **Claude**: Model filtering and selection logic fixes
- **Gemini**: Algorithm optimization (MCT, performance)
- **QWEN**: Central documentation maintenance (this file)

### Handoff Protocol
1. **Task Completion**: Mark items as ✅ completed
2. **Status Updates**: Update progress in real-time
3. **Code Changes**: Include `file:line` references
4. **Testing**: Validate fixes before handoff
5. **Documentation**: Update this central file with findings

### Next Session Priorities
1. Complete --fast model filtering fix
2. Validate with test cases
3. Hand off to Gemini for MCT optimization
4. Begin memory management improvements

## RECENT FINDINGS & INVESTIGATIONS

### --fast Argument Analysis
- **Command Line**: `python main.py --fast` sets `g.FORCE_FAST=True`
- **Expected**: Only models with `strength="SMALL"` should be available
- **Current**: gemini-2.5-flash (non-lite) being selected inappropriately
- **Investigation Status**: Claude examining model selection logic

### Code Locations of Interest
- **Argument Parsing**: `main.py:174` - `--fast` flag definition
- **Model Selection**: Around model iteration loops in main execution
- **Router Implementation**: `py_classes/cls_llm_router.py` - `get_models()` method

---
**Last Updated**: Current Session - Claude investigating --fast bug
**Maintained By**: Multi-agent coordination (Claude/Gemini/QWEN)
**Auto-Update**: ✅ Enabled for real-time task tracking

### 🔵 LOW PRIORITY - CODE QUALITY
- [ ] **main.py Linting Issues** (QWEN Active)
  - **Location**: `main.py`
  - **Symptoms**: Various `flake8` style violations.
  - **Action Items**:
    - [ ] Add newline at end of file (`W292:1:1`)
    - [ ] Fix E302: Ensure 2 blank lines before function/class definitions (many instances, e.g., `main.py:249:1`)
    - [ ] Fix E402: Move module-level imports to the top of the file (e.g., `main.py:4:1`)
    - [ ] Fix E261/E262: Ensure inline comments have at least two spaces before them and start with '# ' (e.g., `main.py:2677:38`)
    - [ ] Fix E501: Refactor long lines to be under 79 characters (many instances, e.g., `main.py:1584:80`)
    - [ ] Fix W293: Remove whitespace from blank lines (e.g., `main.py:1610:1`)
