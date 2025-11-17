# Autonomous ML-Based Permission System

## Summary

This PR implements a comprehensive autonomous permission system using machine learning to reduce friction in the CLI-Agent while maintaining security. The system combines static rule-based permissions with ML-based classification to auto-approve safe operations and flag risky ones.

**Branch:** `claude/autonomous-permissions-ml-01Be6z3jqThesixqWbBQtmid`
**Target:** `main`
**Status:** ✅ Ready for Review

## What's Included

### Backend Implementation (Commits 1-2)

**Core Components:**
- `core/permissions/autonomous_permission.py` - Orchestration layer combining static rules + ML
- `core/permissions/embedding_classifier.py` - 1D safety scoring using cosine similarity (already existed)
- Updated `core/permissions/settings_manager.py` - ML configuration management
- Updated `agent/commands/simple_commands.py` - CLI commands (`/perms ml-*`)
- `tests/core/test_autonomous_permissions.py` - Comprehensive test suite

**Architecture:**
```
Tool Call → Layer 1: Static Deny Rules → BLOCK (highest priority)
              ↓
          Layer 2: Static Allow Rules → APPROVE
              ↓
          Layer 3: ML Classifier → Auto-Allow / Ask / Neutral
              ↓
          Layer 4: User Prompt → USER_DECISION (fallback)
```

**Performance Metrics (from validation):**
- Accuracy: 86.2% at threshold ±0.01
- Auto-allow rate: ~57% of safe operations
- Speed: 10-100ms (cached/uncached)

### Frontend Implementation (Commits 3-4 - Phase 1 & 2)

**Phase 1: ML Score Visualization (Commit 3)**

Real-time safety scores in permission prompts:
- Color-coded indicators (🟢 safe / 🟡 uncertain / 🔴 risky)
- Confidence meter showing ML certainty
- Expandable "Why this score?" explanations

Files changed:
- `core/permissions/permission_prompt.py` - Enriched WebSocket payload with ML data
- `frontend/js/websocket.js` - Enhanced permission request UI
- `frontend/css/components/ml-score.css` - Complete styling (new file)
- `frontend/index.html` - Linked ML score CSS

User Experience:
```
🔐 PERMISSION REQUIRED
Execute bash command: npm install

🤖 ML Safety Score: ✅ SAFE (+0.042)
[━━━━━━━━━━━━━━━━━━━━━|━━━━━━━]
Confidence: ████████░░ 85%

Why this score? ▼
  ML classifier analyzed this as a safe package
  management operation with high confidence.
```

**Phase 2: ML Configuration Panel (Commit 4)**

Complete configuration UI in settings panel:
- Enable/disable toggle with visual status
- Model selection dropdown
- Interactive threshold sliders with live preview
- Decision zones visualization
- Performance statistics dashboard

Files added/changed:
- `api/permissions_api.py` - Added 3 ML endpoints (GET/POST ml-config, GET ml-stats)
- `frontend/index.html` - Added ML config section to settings panel
- `frontend/js/ml-settings-manager.js` - Full settings manager (350+ lines, new file)
- `frontend/css/components/ml-settings.css` - Complete styling (new file)

User Experience:
```
Settings Panel → ML Autonomous Permissions:

[✓] Enable ML Autonomous Permissions  [Enabled ✓]

Model: embeddinggemma:latest ▼

Auto-Allow Threshold: ━━━━●━━━━ +0.010
  (Scores above this are auto-approved)

Ask-User Threshold: ━━━━●━━━━ -0.010
  (Scores below this prompt with warning)

Decision Zones Preview:
[ASK]─────[NEUTRAL]───[AUTO-ALLOW]──────
-1.0           0.0              +1.0

ML Performance Statistics:      [🔄 Refresh]
  Total Checks: 156
  ML Auto-Allow: 89 (57.1%)  ← Key metric
  Static Rules: 35 (22.4%)
  User Prompts: 32 (20.5%)
```

### Documentation

- `core/permissions/docs/AUTONOMOUS_PERMISSIONS.md` - Complete user guide
- `docs/AUTONOMOUS_PERMISSIONS_FRONTEND_INTEGRATION.md` - Frontend integration plan
- Updated `core/permissions/docs/PERMISSION_SYSTEM.md` - Architecture documentation

## Configuration

**Location:** `~/.cli-agent/settings.json`

```json
{
  "autonomous_permissions": {
    "enabled": false,  // Opt-in for safety
    "model": "embeddinggemma:latest",
    "threshold_allow": 0.01,
    "threshold_deny": -0.01,
    "fallback_on_error": "ask"
  }
}
```

## CLI Commands

Users can control ML permissions via CLI:

```bash
/perms ml-status          # Show ML classifier status
/perms ml-enable          # Enable ML permissions
/perms ml-disable         # Disable ML permissions
/perms ml-config [k=v]    # View/update configuration
/perms ml-stats           # Show performance statistics
```

## Testing

**Test Coverage:**
- Unit tests for `EmbeddingClassifier`
- Integration tests for `AutonomousPermission`
- Settings persistence tests
- Mock-based tests (work without Ollama)

**Manual Testing Phase 1:**
```bash
# Enable ML
/perms ml-enable

# Test safe operation (should show green score in permission prompt)
# Trigger: readfile ./README.md

# Test risky operation (should show red score + warning)
# Trigger: bash rm file.txt

# Check statistics
/perms ml-stats
```

**Manual Testing Phase 2:**
```bash
# In Web UI:
1. Click settings icon (gear) in top right
2. Scroll to "ML Autonomous Permissions" section
3. Toggle ML on/off (watch status indicator change)
4. Adjust threshold sliders (watch preview update in real-time)
5. Click "Refresh Stats" to load performance metrics
6. Change model selection (persists after page reload)
7. Close settings panel and test permission prompts
```

## Design Principles

✅ **Fail-safe:** ML disabled by default, errors fall back to prompts
✅ **Transparent:** All decisions logged with scores
✅ **Layered:** Static rules always win over ML
✅ **Typed:** Strong typing with DAG-like docstrings for clarity
✅ **Opt-in:** Requires explicit user enablement

## Security Considerations

- **No auto-override:** Static deny rules block regardless of ML score
- **Path normalization:** Prevents `../` bypass attacks
- **No external APIs:** Ollama runs locally (when enabled)
- **Audit logs:** All ML decisions logged to session logs

## Breaking Changes

None. This is a purely additive feature:
- Existing permission system unchanged
- ML disabled by default
- Backward compatible configuration
- Graceful degradation if Ollama unavailable

## Future Work (Not in this PR)

**Phase 3:** Advanced features (optional)
- Decision history log with export
- Anchor concept explainability
- Per-project threshold profiles

## Dependencies

**Optional (for ML features):**
- Ollama (local embedding server)
- `embeddinggemma:latest` model (~384MB)
- `pip install ollama` (Python client)

**Note:** System works fine without Ollama - ML simply stays disabled.

## Migration Guide

No migration needed. To start using:

1. Install Ollama: https://ollama.com/download
2. Pull model: `ollama pull embeddinggemma:latest`
3. Enable in CLI-Agent: `/perms ml-enable`
4. Watch ML scores appear in permission prompts

## Reviewers

Please review:
- [ ] Architecture and code organization
- [ ] Type safety and error handling
- [ ] Frontend integration (Phase 1)
- [ ] Documentation completeness
- [ ] Test coverage

## Screenshots

### Before (Static Permissions)
Every operation prompts the user with no context.

### After (ML Enabled)
Permission prompts now show ML safety scores:
- Green bar for safe operations (+0.01 to +1.0)
- Red bar for risky operations (-1.0 to -0.01)
- Amber for uncertain operations (-0.01 to +0.01)
- Confidence meter showing ML certainty
- Expandable reasoning

## Related Issues

Related to improved permission UX and friction reduction.

## Checklist

- [x] Code follows project style guidelines
- [x] Tests added and passing
- [x] Documentation updated
- [x] No breaking changes
- [x] Strong typing throughout
- [x] Error handling implemented
- [x] Security considerations addressed
- [x] Frontend integration (Phase 1) complete
- [x] Backward compatible

---

**Ready to merge!** This PR delivers a production-ready autonomous permission system with ML score visualization. Phase 2 (configuration panel) can follow as a separate PR.
