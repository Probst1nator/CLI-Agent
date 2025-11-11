# Documentation Index

**CLI-Agent uses hierarchical documentation** where docs live close to the code they describe. This index provides a complete map of all documentation throughout the project.

## Documentation Structure

### 📚 Component Documentation

Each major component has its own `docs/` directory with architecture and implementation details:

#### Agent Subsystem (`/agent/docs/`)
- **README.md** - Agent architecture overview
  - AppContext pattern and dependency injection
  - Command system architecture
  - GUI components (prompt_toolkit)
  - Input management system

**Subcomponents:**
- `/agent/core/docs/` - Core agent infrastructure
- `/agent/commands/docs/` - Command system implementation
- `/agent/gui/docs/` - GUI layouts and widgets

#### Backend API (`/api/docs/`)
- **README.md** - Backend API architecture
  - aiohttp server and WebSocket
  - Session management and audit logging
  - REST API endpoints
- **BACKEND_FRONTEND_ARCHITECTURE.md** - Modern backend/frontend separation
- **CHAT_STATE_SYNC.md** - Chat state synchronization via WebSocket
- **SESSION_SHARING.md** - Multi-session support
- **MULTI_USER_ARCHITECTURE.md** - Multi-user deployment

#### Frontend (`/frontend/docs/`)
- **README.md** - Frontend architecture
  - Browser-based UI (HTML/CSS/JS)
  - State management (ChatStateManager)
  - Audio manager (voice interaction)
  - WebSocket client
- **STATE_MANAGEMENT.md** - (To be created) Frontend state patterns

#### Core Infrastructure (`/core/docs/`)
- **README.md** - Core infrastructure overview
  - LLM router and provider system
  - Chat management
  - Permission system
  - File manager and caching
- **LLM_ROUTER_API.md** - LLM routing and model selection
- **APIS.md** - Provider API documentation
- **PROFILE_VALIDATION.md** - LLM profile validation

**Subcomponents:**
- `/core/providers/docs/` - AI provider implementations
  - **README.md** - Provider system overview
  - **HUGGINGFACE_LOCAL_PROVIDER.md** - Local HuggingFace models
- `/core/permissions/docs/` - Permission system
  - **PERMISSION_SYSTEM.md** - Permission architecture
  - **PERMISSION_SYSTEM_GUIDE.md** - Usage guide

#### Shared Utilities (`/shared/docs/`)
- **README.md** - Shared utilities overview
  - Audio processing and routing
  - Web utilities and search
  - Logging system
  - Path resolution
- **LOGGING.md** - Comprehensive logging system documentation
- **PERSISTENT_FILES.md** - Persistent storage patterns
- **FILE_FORMAT_SUPPORT.md** - Supported file formats (PDF, Office docs, etc.)
- **SEARCHWEB_VECTOR_DB.md** - Search result caching with vector DB

#### Tools (`/tools/docs/`)
- **README.md** - Tool system architecture
  - Auto-discovery pattern
  - Tool interface specification
  - Per-tool logging
- **SEARCHWEB_VECTOR_DB.md** - SearchWeb tool caching (duplicate of shared/docs/)

#### Docker & Deployment (`/docker_files/docs/`)
- **README.md** - Deployment architecture overview
  - Isolated mode
  - Multi-instance setup
  - Development vs production
- **DEPLOYMENT.md** - Deployment guide
- **ISOLATED_MODE.md** - Isolated container documentation
- **DOCKER_MULTI_CONTAINER.md** - Multi-container setup
- **NETWORK_ACCESS.md** - Container network access and security
- **VISUALRESPONSE_ARCHITECTURE.md** - Legacy architecture (historical)
- **README_CONTAINER_MANAGER_SECURITY.md** - Container security
- **README_EGRESS_FILTERING.md** - Egress filtering
- **README_SECCOMP.md** - Seccomp profiles

#### Infrastructure (`/infrastructure/docs/`)
- **README.md** - Infrastructure overview
  - Rate limiting
  - Vector database
  - Remote hosting
- **INFRASTRUCTURE_STATUS.md** - Current infrastructure status

#### Tests (`/tests/docs/`)
- **README.md** - Testing architecture and strategy
  - Hierarchical test levels (0-4)
  - Test runners (test.py, run_tests.py)
  - Testing patterns and best practices

### 📂 Root Documentation (`/docs/`)

Cross-cutting concerns and project-wide documentation:

#### Development (`/docs/development/`)
- **LINTER_SETUP.md** - Code quality tools (Black, Ruff)
- **DEBUGGING_ENHANCEMENTS.md** - Debugging tools and techniques

#### Deployment (`/docs/deployment/`)
- **PUBLIC_SYNC.md** - Public repository sync workflow
- **RELEASE_WORKFLOW.md** - Release process
- **SELF_HOSTED_RUNNER_SETUP.md** - GitHub Actions self-hosted runners

#### Archived & Experimental
- `/docs/archived/` - Historical documentation
  - `permission-migration/` - Old permission system docs
- `/docs/experimental/` - Experimental features and proposals
- `/docs/typescript/` - TypeScript migration documentation

## Quick Reference

### Finding Documentation

**Working on a specific component?** → Check `<component>/docs/README.md`

| Working on... | Check... |
|---------------|----------|
| LLM router, providers, permissions | `core/docs/README.md` |
| Backend API, WebSocket, sessions | `api/docs/README.md` |
| Frontend UI, state management | `frontend/docs/README.md` |
| Commands, GUI, app context | `agent/docs/README.md` |
| Tools implementation | `tools/docs/README.md` |
| Logging, audio, utilities | `shared/docs/README.md` |
| Docker deployment | `docker_files/docs/README.md` |
| Infrastructure services | `infrastructure/docs/README.md` |
| Testing | `tests/docs/README.md` |

**Need cross-cutting info?** → Check `/docs/development/` or `/docs/deployment/`

### Documentation Principles

1. **Locality**: Docs live next to the code they describe
2. **Specificity**: More detail at deeper levels
3. **Navigation**: Each INDEX.md links to local docs
4. **No legacy**: Clean structure, no backward compatibility layers

See `/CLAUDE.md` → "Documentation Strategy" and "No Legacy Mindset" for complete guide.

## Master File List

### Root Level
- `/README.md` - Project overview and quick start
- `/CLAUDE.md` - AI assistant instructions (includes Documentation Strategy)
- `/ARCHITECTURE.md` - High-level system architecture
- `/INDEX.md` - Complete file index (component structure)

### Component Documentation Hubs
- `/agent/docs/README.md` - Agent subsystem
- `/api/docs/README.md` - Backend API
- `/frontend/docs/README.md` - Frontend
- `/core/docs/README.md` - Core infrastructure
- `/shared/docs/README.md` - Shared utilities
- `/tools/docs/README.md` - Tools system
- `/docker_files/docs/README.md` - Deployment
- `/infrastructure/docs/README.md` - Infrastructure
- `/tests/docs/README.md` - Testing

### Specialized Guides
- `/tools/CLAUDE.md` - Tool implementation guide
- `/frontend/CLAUDE.md` - Frontend development guide

---

*This hierarchical documentation structure reduces complexity and makes it easy to find and maintain relevant documentation throughout the CLI-Agent project.*
