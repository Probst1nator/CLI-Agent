# Autonomous Permission System

Machine learning-based permission classifier that reduces friction by auto-approving safe operations while flagging risky ones.

## Quick Start

```bash
# Enable ML permissions
/perms ml-enable

# Check status
/perms ml-status

# View statistics
/perms ml-stats

# Disable if needed
/perms ml-disable
```

## Overview

The autonomous permission system combines **static rules** (user-configured) with **ML classification** (embedding-based) to make intelligent permission decisions without constant user prompting.

### Architecture

```
Tool Call → Static Deny? → BLOCK
              ↓ NO
          Static Allow? → APPROVE
              ↓ NO
          ML Classifier → Auto-Allow (safe) / Ask (unsafe) / Neutral
              ↓
          User Prompt (fallback)
```

**Decision Flow:**

1. **Layer 1: Static Deny Rules** (highest priority)
   - Blocks immediately if matches deny pattern
   - Example: `bash(sudo:*)` → instant block

2. **Layer 2: Static Allow Rules**
   - Approves immediately if matches allow pattern
   - Example: `readfile(./src/**)` → instant approval

3. **Layer 3: ML Classifier** (if enabled)
   - Analyzes tool call using embedding similarity
   - Auto-allows if safety score > threshold
   - Asks with warning if score < -threshold
   - Neutral → continues to user prompt

4. **Layer 4: User Prompt** (default fallback)
   - Shows traditional permission prompt
   - Includes ML score if available

## How It Works

### Embedding-Based Classification

The ML classifier uses **1D safety scoring** via cosine similarity:

1. **Anchor Concepts:**
   - 18 safe anchors (e.g., `<bash>ls</bash>`, `<readfile>...</readfile>`)
   - 29 unsafe anchors (e.g., `<bash>rm -rf /</bash>`, `<writefile>...</writefile>`)

2. **Similarity Computation:**
   - Tool call → embedding vector (via Ollama)
   - Compute average similarity to safe group
   - Compute average similarity to unsafe group
   - Safety score = (safe_sim - unsafe_sim) / (safe_sim + unsafe_sim)

3. **Decision Thresholds:**
   - Score > +0.01 → **Auto-allow** (safe)
   - Score < -0.01 → **Ask user** (unsafe)
   - -0.01 ≤ score ≤ +0.01 → **Neutral** (uncertain)

### Performance Metrics

From validation testing:

- **Accuracy:** 86.2% at threshold ±0.01
- **Auto-allow rate:** ~57% of safe operations
- **False positive rate:** ~5% (safe → ask)
- **Speed:**
  - First call: 50-100ms (embedding computation)
  - Cached calls: 10-20ms

## Configuration

### Settings File

Location: `~/.cli-agent/settings.json`

```json
{
  "permissions": {
    "allow": ["bash(npm:*)", "readfile(./src/**)"],
    "deny": ["readfile(.env)", "bash(sudo:*)"],
    "ask": ["bash(git push:*)"]
  },

  "autonomous_permissions": {
    "enabled": false,
    "model": "embeddinggemma:latest",
    "threshold_allow": 0.01,
    "threshold_deny": -0.01,
    "fallback_on_error": "ask",
    "cache_embeddings": true,
    "log_classifications": true
  }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable/disable ML classification |
| `model` | str | `"embeddinggemma:latest"` | Ollama embedding model |
| `threshold_allow` | float | `0.01` | Auto-allow above this score |
| `threshold_deny` | float | `-0.01` | Ask user below this score |
| `fallback_on_error` | str | `"ask"` | Action if ML fails |
| `cache_embeddings` | bool | `true` | Cache embedding vectors |
| `log_classifications` | bool | `true` | Log ML decisions to debug log |

### Threshold Tuning

**Permissive (auto-allow more):**
```json
"threshold_allow": 0.005,    // Lower = more auto-allows
"threshold_deny": -0.02      // More negative = fewer warnings
```

**Strict (prompt more often):**
```json
"threshold_allow": 0.02,     // Higher = fewer auto-allows
"threshold_deny": -0.005     // Less negative = more warnings
```

**Balanced (default):**
```json
"threshold_allow": 0.01,
"threshold_deny": -0.01
```

## CLI Commands

### `/perms ml-status`

Show ML classifier status and configuration.

```
🤖 ML Permission Classifier Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ENABLED ✓
Model: embeddinggemma:latest
Auto-allow threshold: +0.01
Ask-user threshold: -0.01
Fallback on error: ask
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### `/perms ml-enable`

Enable ML autonomous permissions.

```bash
/perms ml-enable
```

**First-time setup:**
1. Downloads Ollama model if needed (~384MB)
2. Pre-computes anchor embeddings (~1-2 seconds)
3. Initializes embedding cache
4. Confirms ready

### `/perms ml-disable`

Disable ML (falls back to static rules + prompts).

```bash
/perms ml-disable
```

### `/perms ml-config [key=value]`

View or update ML configuration.

**View config:**
```bash
/perms ml-config
```

**Update threshold:**
```bash
/perms ml-config threshold_allow=0.02
```

**Update model:**
```bash
/perms ml-config model=embeddinggemma:2b
```

### `/perms ml-stats`

Show detailed ML permission statistics.

```
📊 ML Permission Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Checks: 156

Decision Breakdown:
  Static Deny:       12 (  7.7%)
  Static Allow:      23 ( 14.7%)
  ML Auto-Allow:     89 ( 57.1%)
  ML Ask:            18 ( 11.5%)
  ML Neutral:        14 (  9.0%)
  User Prompts:       0 (  0.0%)

Embedding Cache:
  Cached embeddings: 45
  Anchor concepts:   47
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## User Experience

### Before (Static Rules Only)

Every new operation prompts the user:

```
🔐 PERMISSION REQUIRED
Execute bash command: npm install

Reason: No matching permission rule
[y/n/a/d/?]: _
```

### After (ML Enabled)

Safe operations auto-approved:

```
✅ Auto-approved by ML classifier (score=+0.042)
Executing: npm install
```

Risky operations shown with ML warning:

```
🔐 PERMISSION REQUIRED
Execute bash command: rm file.txt

Reason: ML classifier: potentially unsafe (score=-0.053)
Safety: ⚠️ POTENTIALLY UNSAFE (high confidence)

[y/n/a/d/?]: _
```

## Developer Integration

### Using in Code

```python
from core.permissions.autonomous_permission import AutonomousPermission
from core.permissions.settings_manager import SettingsManager

# Initialize
mgr = SettingsManager()
ml_config = mgr.get_ml_config()

autonomous = AutonomousPermission(
    settings_manager=mgr,
    ml_enabled=ml_config["enabled"],
    ml_model=ml_config["model"],
    threshold_allow=ml_config["threshold_allow"],
    threshold_deny=ml_config["threshold_deny"]
)

# Check permission
decision, reason, confidence, ml_score = await autonomous.check_permission(
    tool_name="bash",
    command="npm install"
)

if decision == "allow":
    # Execute tool
    pass
elif decision == "deny":
    # Block tool
    pass
else:  # "ask"
    # Prompt user
    pass
```

### Type Signatures

```python
# Decision flow types
PermissionDecision = Literal["allow", "deny", "ask"]
PermissionReason = str
ConfidenceScore = float  # 0.0-1.0
SafetyScore = float      # -1.0 to +1.0

# Main API
async def check_permission(
    tool_name: str,
    file_path: Optional[str] = None,
    command: Optional[str] = None,
    xml_content: Optional[str] = None
) -> tuple[PermissionDecision, PermissionReason, ConfidenceScore, Optional[SafetyScore]]
```

### DAG Flow Documentation

```python
"""
Decision Flow (DAG):
    Tool Call (tool_name, file_path, command, xml_content)
        ↓
    Layer 1: Static Deny Rules
        ├─ "deny" → BLOCK (highest priority)
        ↓
    Layer 2: Static Allow Rules
        ├─ "allow" → APPROVE
        ↓
    Layer 3: ML Classifier (if enabled)
        ├─ "auto-allow" (score > threshold) → APPROVE
        ├─ "ask" (score < -threshold) → USER_PROMPT(warning)
        ├─ "neutral" → USER_PROMPT
        └─ error/disabled → USER_PROMPT
        ↓
    Layer 4: User Prompt (default fallback)
        → USER_DECISION
"""
```

## Troubleshooting

### ML Not Initializing

**Symptom:** `/perms ml-enable` fails

**Causes:**
1. Ollama not installed: `pip install ollama`
2. Ollama server not running: `ollama serve`
3. Model not available: `ollama pull embeddinggemma:latest`

**Check logs:**
```bash
grep "ML classifier" ~/.cli-agent/logs/session.log
```

### High Memory Usage

**Symptom:** Large memory footprint with ML enabled

**Solutions:**
1. Use smaller embedding model: `embeddinggemma:2b` instead of `latest`
2. Disable embedding cache: `"cache_embeddings": false` (trades speed for memory)
3. Restart periodically to clear cache

### Incorrect Classifications

**Symptom:** Safe operations being flagged as unsafe (or vice versa)

**Solutions:**
1. Tune thresholds via `/perms ml-config`
2. Add static rules for specific patterns
3. Check if operation is genuinely ambiguous (e.g., `npm install` can be safe or risky)

**Report feedback:**
- File issues with misclassified examples
- Include: tool call XML, expected decision, actual decision, ML score

### Performance Issues

**Symptom:** Slow permission checks

**Causes:**
1. Ollama server overloaded
2. Network latency to Ollama host
3. Embedding cache disabled

**Optimizations:**
1. Use local Ollama instance
2. Enable embedding cache (default)
3. Pre-warm cache by running common operations

## Security Considerations

### Fail-Safe Design

- **ML disabled by default:** Opt-in to reduce risk
- **Static deny rules always win:** No ML override for security rules
- **Fallback to prompts:** If ML fails, ask user (never auto-allow)
- **No model training:** Read-only anchor concepts prevent poisoning

### Attack Vectors

**Bypass attempts:**
- Obfuscated commands: `b``a``s``h rm -rf /` → ML sees raw XML, not obfuscated shell
- Path traversal: `../../.env` → Static rules normalize paths first

**Mitigation:**
- Layer precedence ensures static deny rules block regardless of ML
- Path normalization in static layer prevents `../` bypass
- XML parsing uses safe patterns

### Privacy

- **No data collection:** Embeddings computed locally
- **No external APIs:** Ollama runs on local machine
- **Audit logs:** All ML decisions logged to session logs (opt-in)

## Future Enhancements

Planned for future releases:

1. **User feedback loop:** Learn from "always allow/deny" choices
2. **Project-specific anchors:** Custom safe/unsafe examples per codebase
3. **Confidence visualization:** Show score distribution in GUI
4. **Multi-model support:** Try different embedding models (OpenAI, Cohere)
5. **Anomaly detection:** Flag unusual tool call patterns
6. **Explainability:** Show which anchor concepts influenced decision

## References

- **Original research:** See `/experiments/permission_classifier.ipynb`
- **Validation metrics:** 86.2% accuracy, 57% auto-allow rate at ±0.01 threshold
- **Embedding model:** embeddinggemma (Google, 384-dimensional embeddings)
- **Architecture:** Anchor-based 1D safety scoring with cosine similarity

---

**Last Updated:** 2025-11-17
**Version:** 1.0.0
**Status:** Production-ready
