# TypeScript Migration Plan

## Problem Statement

The frontend currently uses JSDoc for type definitions, which has a critical limitation: **types are file-local**. Each file that uses a type must duplicate the type definition or accept no type checking. This leads to:

1. **Type duplication** - `MessageMetadata`, `ToolInfo`, etc. are defined in multiple files
2. **Maintenance burden** - Updating a type requires changing multiple files
3. **Type drift** - Definitions can become inconsistent across files
4. **No compile-time validation** - Type errors only caught at runtime or in IDE

## Proposed Solution: Migrate to TypeScript

### Benefits

**Developer Experience:**
- ✅ Single source of truth for types (DRY principle)
- ✅ Real type checking at compile time catches errors before runtime
- ✅ Import types across files: `import { MessageMetadata } from './types'`
- ✅ Better IDE support (autocomplete, go-to-definition, refactoring)
- ✅ Type inference reduces boilerplate

**Code Quality:**
- ✅ Prevents type-related bugs
- ✅ Self-documenting code (types as documentation)
- ✅ Easier refactoring with confidence
- ✅ Better collaboration (explicit contracts between modules)

### Trade-offs

**Cons:**
- ❌ Requires build step (breaks current "no build step" architecture)
- ❌ Additional tooling complexity (TypeScript compiler, bundler)
- ❌ Deployment changes (build artifacts instead of raw files)
- ❌ Learning curve for contributors unfamiliar with TypeScript
- ❌ Slightly slower development iteration (compile before test)

**Mitigation:**
- Modern build tools (Vite, esbuild) are extremely fast (<1s builds)
- TypeScript is widely adopted - most contributors already know it
- Watch mode enables instant recompilation on save
- Source maps maintain debuggability

## Migration Strategy

### Phase 1: Setup TypeScript Tooling (1-2 hours)

**Goal:** Get TypeScript compiler working without breaking existing code.

**Steps:**
1. Install TypeScript and build tools:
   ```bash
   npm install --save-dev typescript vite @types/node
   ```

2. Create `tsconfig.json`:
   ```json
   {
     "compilerOptions": {
       "target": "ES2020",
       "module": "ES2020",
       "lib": ["ES2020", "DOM"],
       "outDir": "./dist",
       "rootDir": "./frontend",
       "strict": true,
       "esModuleInterop": true,
       "skipLibCheck": true,
       "moduleResolution": "node",
       "resolveJsonModule": true,
       "allowJs": true,
       "checkJs": false
     },
     "include": ["frontend/**/*.ts", "frontend/**/*.js"],
     "exclude": ["node_modules", "dist"]
   }
   ```

3. Create `vite.config.ts`:
   ```typescript
   import { defineConfig } from 'vite'

   export default defineConfig({
     root: 'frontend',
     build: {
       outDir: '../dist',
       emptyOutDir: true,
       rollupOptions: {
         input: 'frontend/index.html'
       }
     },
     server: {
       port: 8765,
       proxy: {
         '/api': 'http://localhost:7860',
         '/ws': {
           target: 'ws://localhost:7860',
           ws: true
         }
       }
     }
   })
   ```

4. Update `package.json`:
   ```json
   {
     "scripts": {
       "dev": "vite",
       "build": "tsc && vite build",
       "preview": "vite preview"
     }
   }
   ```

**Validation:**
- Run `npm run dev` - should serve frontend without errors
- Verify hot reload works

### Phase 2: Create Centralized Type Definitions (2-3 hours)

**Goal:** Define all types in one place.

**Steps:**
1. Create `frontend/types/index.ts`:
   ```typescript
   // Core message types
   export interface Message {
     role: 'user' | 'assistant' | 'system'
     content: string
     images?: string[]
     metadata?: MessageMetadata
     type?: 'html' | 'markdown' | 'image_upload_ack'
     timestamp?: number
   }

   export interface MessageMetadata {
     is_bootstrap?: boolean
     timestamp?: string  // ISO 8601
     model?: string
     tool_info?: ToolInfo[]
     message_id?: number
     response_model?: string  // Legacy field
   }

   export interface ToolInfo {
     name: string
     status?: 'success' | 'error' | 'pending'
     execution_time?: number
     duration_ms?: number  // Legacy field
     model?: string
     args?: string
     context_returned?: string
     context_length?: number
     error?: string
   }

   export interface ChatMetadata {
     created_at?: string  // ISO 8601
     updated_at?: string  // ISO 8601
   }

   export interface ChatData {
     id?: string
     messages: Message[]
     metadata?: ChatMetadata
   }

   // API types
   export interface APIResponse<T = any> {
     status: 'success' | 'error'
     data?: T
     error?: string
     code?: string
   }

   // WebSocket types
   export interface WebSocketEvent<T = any> {
     type: string
     data?: T
     timestamp?: number
   }

   // LLM types
   export interface LLMModel {
     id: string
     name: string
     provider: string
     context_window: number
     capabilities: string[]
     status: string
   }

   // Tool types
   export interface ToolDefinition {
     name: string
     description: string
     enabled: boolean
     metadata?: Record<string, any>
   }
   ```

2. Create `frontend/types/backend.ts` for backend-specific types:
   ```typescript
   export interface ContainerInfo {
     uuid: string
     status: 'running' | 'stopped' | 'error'
     backend_url: string
   }

   export interface PermissionRequest {
     id: string
     tool_name: string
     args: string
     timestamp: number
   }
   ```

**Validation:**
- TypeScript compiles without errors
- Types are exported correctly

### Phase 3: Incremental Migration (File by File) (10-15 hours)

**Goal:** Convert JavaScript files to TypeScript incrementally.

**Priority Order:**
1. **Types and utilities first** (no dependencies):
   - `types.js` → Delete (replaced by `types/index.ts`)
   - `logger.js` → `logger.ts`
   - `js-utils.js` → `js-utils.ts`

2. **Core modules** (depend on types):
   - `websocket.js` → `websocket.ts`
   - `backend-connector.js` → `backend-connector.ts`
   - `chat-state.js` → `chat-state.ts`

3. **Renderers and UI**:
   - `renderers.js` → `renderers.ts`
   - `audio-manager.js` → `audio-manager.ts`
   - `llm-selector.js` → `llm-selector.ts`

4. **Main app last**:
   - `main.js` → `main.ts`

**Per-file migration steps:**
1. Rename `.js` → `.ts`
2. Add type imports: `import { Message, ToolInfo } from './types'`
3. Add type annotations to function signatures
4. Fix type errors reported by TypeScript
5. Remove local JSDoc `@typedef` declarations
6. Test in browser

**Example migration (chat-state.js → chat-state.ts):**

**Before (JavaScript with JSDoc):**
```javascript
/**
 * @typedef {Object} Message
 * @property {string} role
 * ...
 */

class ChatStateManager {
    constructor(wsManager, renderer) {
        this.messages = []
        this.metadata = {}
    }

    addMessage(message) {
        this.messages.push(message)
    }
}
```

**After (TypeScript):**
```typescript
import { Message, ChatMetadata } from './types'

class ChatStateManager {
    messages: Message[] = []
    metadata: ChatMetadata = {}

    constructor(
        private wsManager: WebSocketManager,
        private renderer: ContentRenderer
    ) {}

    addMessage(message: Message): void {
        this.messages.push(message)
    }
}
```

**Validation per file:**
- TypeScript compiles without errors
- Runtime behavior unchanged
- No console errors in browser

### Phase 4: Backend Integration (2-3 hours)

**Goal:** Update Python backend to serve TypeScript-compiled frontend.

**Steps:**
1. Update `api/server.py` to serve from `dist/` instead of `frontend/`:
   ```python
   # Old
   app.router.add_static('/frontend', 'frontend')

   # New
   app.router.add_static('/', 'dist')
   ```

2. Update build script in `main.py`:
   ```python
   async def start_visualresponse_server():
       # Build frontend if in production mode
       if not os.path.exists('dist') or os.getenv('ENV') == 'production':
           logging.info("Building frontend...")
           subprocess.run(['npm', 'run', 'build'], check=True)

       # Start server
       ...
   ```

3. Add development mode detection:
   ```python
   if os.getenv('ENV') == 'development':
       # Use Vite dev server (npm run dev)
       logging.info("Development mode: use npm run dev separately")
   else:
       # Serve built files
       ...
   ```

**Validation:**
- `npm run build` produces `dist/` with compiled files
- Backend serves built files correctly
- Production deployment works

### Phase 5: Update Documentation (1 hour)

**Goal:** Update all documentation to reflect TypeScript migration.

**Files to update:**
1. `frontend/CLAUDE.md`:
   - Remove "No Build Step" design principle
   - Add build instructions
   - Update module loading order section
   - Add TypeScript conventions

2. `README.md`:
   - Add frontend build instructions
   - Update development setup

3. `docs/VISUALRESPONSE_ARCHITECTURE.md`:
   - Update frontend architecture section
   - Add build pipeline documentation

4. `.gitignore`:
   ```
   # Frontend build artifacts
   /dist/
   /node_modules/
   /.vite/
   ```

### Phase 6: CI/CD Updates (1-2 hours)

**Goal:** Integrate TypeScript build into CI/CD pipeline.

**Steps:**
1. Update GitHub Actions (if exists):
   ```yaml
   - name: Install frontend dependencies
     run: npm install

   - name: Build frontend
     run: npm run build

   - name: Type check
     run: npx tsc --noEmit
   ```

2. Update Docker build:
   ```dockerfile
   # Install Node.js
   RUN apt-get install -y nodejs npm

   # Build frontend
   COPY package*.json ./
   RUN npm install
   COPY frontend/ ./frontend/
   RUN npm run build
   ```

3. Add pre-commit hook (optional):
   ```bash
   # .git/hooks/pre-commit
   #!/bin/bash
   npm run build
   if [ $? -ne 0 ]; then
     echo "TypeScript build failed"
     exit 1
   fi
   ```

## Rollback Plan

If migration causes critical issues:

1. **Immediate rollback:**
   ```bash
   git revert <migration-commit>
   ```

2. **Serve old files:**
   - Keep original `.js` files in `frontend/legacy/`
   - Switch backend to serve from `frontend/legacy/`

3. **Fix forward:**
   - TypeScript errors are usually straightforward to fix
   - Use `// @ts-ignore` for quick workarounds
   - Gradual migration allows partial rollback (some files in TS, some in JS)

## Timeline Estimate

| Phase | Duration | Can Start After |
|-------|----------|-----------------|
| Phase 1: Setup | 1-2 hours | Immediately |
| Phase 2: Types | 2-3 hours | Phase 1 |
| Phase 3: Migration | 10-15 hours | Phase 2 |
| Phase 4: Backend | 2-3 hours | Phase 3 |
| Phase 5: Docs | 1 hour | Phase 4 |
| Phase 6: CI/CD | 1-2 hours | Phase 4 |
| **Total** | **17-26 hours** | |

**Parallel work:** Phases 5 and 6 can happen alongside Phase 3.

## Success Criteria

Migration is complete when:
- ✅ All `.js` files converted to `.ts`
- ✅ TypeScript compiles with zero errors
- ✅ All runtime tests pass
- ✅ No console errors in browser
- ✅ Build time < 5 seconds for development
- ✅ Documentation updated
- ✅ CI/CD pipeline includes TypeScript build
- ✅ Team can develop without JSDoc duplication

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Build step slows development | Medium | High | Use Vite watch mode (instant rebuild) |
| Breaking changes in production | High | Low | Incremental migration, thorough testing |
| Team unfamiliarity with TypeScript | Low | Low | TypeScript is widely adopted, good docs |
| Increased deployment complexity | Medium | Medium | Document build process, automate in CI/CD |
| Type errors block development | Medium | Medium | Use `// @ts-ignore` sparingly, fix forward |

## Alternatives Considered

### Alternative 1: Keep JSDoc, Accept Duplication
- **Pros:** No build step, immediate deployment
- **Cons:** Type duplication, maintenance burden, no compile-time validation
- **Decision:** Rejected - scaling pain outweighs simplicity benefit

### Alternative 2: Use TypeScript in JSDoc Mode
- **Pros:** Type checking without conversion
- **Cons:** Still file-local types, no import/export, awkward syntax
- **Decision:** Rejected - doesn't solve root problem

### Alternative 3: Generate Types from Backend
- **Pros:** Single source of truth (Python TypedDicts)
- **Cons:** Complex tooling, frontend/backend coupling, doesn't solve frontend-only types
- **Decision:** Future enhancement after TS migration

## Open Questions

1. **Should we use strict mode from day one?**
   - Recommendation: Start with `strict: false`, enable incrementally

2. **How to handle third-party libraries without types?**
   - Use `@types/*` packages where available
   - Write minimal `.d.ts` files for others

3. **What about backend Python types?**
   - Keep Python TypedDicts as separate source of truth
   - Consider generating TypeScript from Python in future

## Next Steps

1. Get team buy-in on migration
2. Create feature branch: `feature/typescript-migration`
3. Start with Phase 1 (setup) as proof of concept
4. Review and iterate on approach
5. Proceed with incremental migration
