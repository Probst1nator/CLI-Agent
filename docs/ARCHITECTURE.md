# CLI-Agent Monorepo Decomposition Analysis

## Overview

This document analyzes the current CLI-Agent monorepo structure to understand how to decompose it into 3 separate projects, each corresponding to the tools in `tools/*`. The goal is to create fully independent project roots with minimal interdependencies.

## Current Project Structure

### 1. **Main CLI Agent** (``)
- **Purpose**: Advanced agentic coding assistant with "Monte Carlo Action Engine"
- **Entry Point**: `main.py`
- **Key Features**: 
  - Council of LLMs (Branch Generators, Branch Evaluators, Execution Guards)
  - Interactive CLI with full AI capabilities
  - Tool integration and utility management
  - Vector database for playbooks and tool hints

### 2. **File Copier** (`tools/file_copier/`)
- **Purpose**: Intelligent file management with AI-powered path finding
- **Entry Point**: `tools/file_copier/main.py`
- **Key Features**:
  - Smart paste functionality
  - AI path resolution
  - GUI interface
  - PDF and ODT file support

### 3. **Podcast Generator** (`tools/podcast_generator/`)
- **Purpose**: AI-powered podcast generation from text, PDFs, or clipboard
- **Entry Point**: `tools/podcast_generator/main.py`
- **Key Features**:
  - Multiple TTS engine support (Google GenAI)
  - Audio manipulation and processing
  - Interactive CLI interface
  - PDF text extraction

## Dependency Analysis

### Shared Infrastructure Dependencies

#### Core Package (`core/`)
Used by all 3 projects:
- `core.globals` - Global state management
- `core.chat` - Chat/conversation management
- `core.llm_router` - LLM routing and provider management
- `core.ai_strengths` - AI capability definitions
- `core.providers.*` - AI provider implementations (Anthropic, Google, Groq, etc.)

#### Shared Package (`shared/`)
Used by projects:
- `shared.dia_helper` - Dialog helpers
- `shared.path_resolver` - Path resolution utilities
- `shared.common_utils` - Common utility functions

#### Legacy Dependencies (`py_classes/`, `py_methods/`)
Heavy usage in Main CLI Agent:
- `py_classes.cls_computational_notebook` - Notebook management
- `py_classes.cls_util_manager` - Utility orchestration
- `py_classes.cls_playbook_manager` - Strategic workflows
- `py_classes.cls_util_base` - Base utility class
- `py_classes.utils.*` - Web, Python, RAG utilities
- `py_classes.cls_text_stream_painter` - Text formatting
- `py_methods.utils` - General utilities

### Project-Specific Dependencies

#### Main CLI Agent
- **High Coupling**: Extensively uses `py_classes/` and `core/` infrastructure
- **External Dependencies**: prompt-toolkit, pyperclip, pydub, soundfile, PyMuPDF, numpy
- **Test Infrastructure**: Uses `test.py` from monorepo root

#### File Copier
- **Moderate Coupling**: Uses `core.llm_router`, `core.chat` for AI features
- **Minimal Legacy**: No `py_classes` dependencies
- **External Dependencies**: pyperclip, PyMuPDF, odfpy

#### Podcast Generator
- **Moderate Coupling**: Uses `core` for LLM functionality, `shared.dia_helper`
- **Legacy Usage**: `py_methods.utils` for text extraction
- **External Dependencies**: pydub, google-genai, prompt-toolkit, PyMuPDF, numpy

## Decomposition Strategy

### Option 1: Full Independence (Recommended)
Each project becomes completely self-contained:

```
cli-agent-main/
├── core/ (copy + project-specific modifications)
├── shared/ (minimal subset)
├── py_classes/ (subset needed)
├── py_methods/ (subset needed)
├── src/
├── tests/
├── requirements.txt
└── pyproject.toml

cli-agent-file-copier/
├── core/ (minimal LLM subset)
├── shared/ (path resolution only)
├── src/
├── tests/
├── requirements.txt
└── pyproject.toml

cli-agent-podcast-generator/
├── core/ (LLM + audio providers)
├── shared/ (minimal subset)
├── py_methods/ (utils only)
├── src/
├── tests/
├── requirements.txt
└── pyproject.toml
```

**Pros**: 
- Complete independence
- No version conflicts
- Independent deployment
- Clear ownership

**Cons**: 
- Code duplication
- Maintenance overhead
- Potential drift

### Option 2: Shared Libraries (Alternative)
Create published packages for common functionality:

```
cli-agent-core/ (published package)
├── LLM routing and providers
├── Chat management
├── AI capabilities

cli-agent-shared/ (published package)  
├── Path utilities
├── Common helpers
├── Dialog management

cli-agent-main/
├── requirements.txt (includes cli-agent-core, cli-agent-shared)
├── src/
└── tests/

cli-agent-file-copier/
├── requirements.txt (includes cli-agent-core[minimal])
├── src/
└── tests/

cli-agent-podcast-generator/
├── requirements.txt (includes cli-agent-core, cli-agent-shared)
├── src/
└── tests/
```

**Pros**:
- Reduced duplication
- Centralized maintenance
- Consistent behavior

**Cons**:
- Version coordination complexity
- Deployment dependencies
- Cross-project changes

## Migration Complexity Assessment

### Main CLI Agent: **HIGH COMPLEXITY**
- **Dependencies**: 15+ imports from `py_classes/`, 8+ from `core/`
- **Infrastructure**: Requires most of the monorepo infrastructure
- **Testing**: Uses monorepo test runner
- **Effort**: 3-4 weeks of refactoring

### File Copier: **LOW COMPLEXITY**  
- **Dependencies**: Only 3 imports from `core/`, 1 from `shared/`
- **Infrastructure**: Minimal external requirements
- **Testing**: Self-contained
- **Effort**: 1-2 days of migration

### Podcast Generator: **MEDIUM COMPLEXITY**
- **Dependencies**: 4 imports from `core/`, 2 from `shared/`, 1 from `py_methods/`
- **Infrastructure**: Moderate external requirements
- **Testing**: Self-contained
- **Effort**: 1-2 weeks of refactoring

## Recommendations

### Immediate Actions (Phase 1)
1. **Start with File Copier**: Lowest complexity, can validate decomposition approach
2. **Extract minimal dependencies**: Copy only needed `core/` and `shared/` components
3. **Establish independent testing**: Each project needs its own test infrastructure

### Medium Term (Phase 2) 
1. **Migrate Podcast Generator**: Medium complexity, establishes pattern for AI-enabled tools
2. **Refactor shared AI components**: Create reusable LLM interface patterns

### Long Term (Phase 3)
1. **Migrate Main CLI Agent**: Highest complexity, may require significant architectural changes
2. **Consider publishing shared libraries**: If multiple projects need same functionality

### Technical Considerations
- **Import Path Changes**: All `from core.*` and `from shared.*` imports need updating
- **Configuration Management**: Each project needs independent configuration
- **Testing Infrastructure**: Duplicate or rewrite test runners
- **Dependency Management**: Separate requirements.txt and pyproject.toml files
- **CI/CD**: Independent build and deployment pipelines

The recommended approach is **Option 1 (Full Independence)** starting with the File Copier project to validate the decomposition strategy, followed by incremental migration of the other projects.