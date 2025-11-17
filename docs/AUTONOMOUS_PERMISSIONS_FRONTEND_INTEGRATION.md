# Autonomous Permissions Frontend Integration Plan

Comprehensive guide for integrating ML-based permission interpretability and advanced controls into the web UI.

## Overview

The backend autonomous permission system is complete. This document outlines the **phased frontend integration** to provide users with:

1. **Interpretability:** Visual ML scores, confidence indicators, reasoning
2. **Advanced Control:** Threshold tuning, enable/disable ML, statistics dashboard
3. **Enhanced UX:** Inline explanations, color-coded safety indicators, decision history

## Current State

### Backend (✅ Complete)
- `core/permissions/autonomous_permission.py` - Orchestration layer
- `core/permissions/embedding_classifier.py` - ML classifier
- `core/permissions/settings_manager.py` - Configuration management
- `agent/commands/simple_commands.py` - CLI commands (`/perms ml-*`)
- `api/permissions_api.py` - REST API for permissions

### Frontend (🚧 Needs Enhancement)
- `frontend/js/websocket.js:296-453` - Permission request UI (basic)
- `frontend/js/settings-manager.js` - Settings panel for static rules
- `api/permissions_api.py` - Needs ML endpoints

---

## Phase 1: ML Score Visualization (2-3 hours)

**Goal:** Show ML safety scores and confidence in permission prompts

### Backend Changes

#### 1.1 Update Permission Prompt Payload

**File:** `core/permissions/permission_prompt.py:156-174`

**Current payload:**
```python
permission_payload = {
    "id": f"permission-{int(time.time() * 1000)}",
    "action": "create",
    "type": "permission_request",
    "permission_data": {
        "tool_name": tool_name,
        "action_description": action_desc,
        "file_path": file_path or "",
        "command": command or "",
        "args_display": args_display,
        "reason": reason,
        "options": ["allow", "deny", "always_allow", "always_deny"],
        "rule_options": [...]
    }
}
```

**Add ML fields:**
```python
permission_payload = {
    # ... existing fields ...
    "permission_data": {
        # ... existing fields ...
        "ml_enabled": ml_config.get("enabled", False),
        "ml_score": ml_score if ml_score is not None else None,
        "ml_confidence": confidence_score,
        "ml_decision": ml_decision,  # "auto-allow" | "ask" | "neutral"
        "ml_reason": ml_reason  # Human-readable explanation
    }
}
```

**Implementation:**
```python
# In permission_prompt.py, around line 156
from core.permissions.autonomous_permission import AutonomousPermission

# Before sending permission_payload, check ML classifier
ml_score = None
ml_confidence = 0.0
ml_decision = None
ml_reason = None

# Get ML config
from core.permissions.settings_manager import SettingsManager
mgr = SettingsManager()
ml_config = mgr.get_ml_config()

if ml_config.get("enabled", False):
    try:
        # Create autonomous permission instance
        autonomous = AutonomousPermission(
            settings_manager=mgr,
            ml_enabled=True,
            ml_model=ml_config.get("model"),
            threshold_allow=ml_config.get("threshold_allow"),
            threshold_deny=ml_config.get("threshold_deny")
        )

        # Get ML classification
        decision, reason, confidence, score = await autonomous.check_permission(
            tool_name=tool_name,
            file_path=file_path,
            command=command,
            xml_content=xml_content
        )

        ml_score = score
        ml_confidence = confidence
        ml_decision = decision
        ml_reason = reason

    except Exception as e:
        logger.warning(f"ML classification failed: {e}")

# Add to payload
permission_payload["permission_data"]["ml_enabled"] = ml_config.get("enabled", False)
permission_payload["permission_data"]["ml_score"] = ml_score
permission_payload["permission_data"]["ml_confidence"] = ml_confidence
permission_payload["permission_data"]["ml_decision"] = ml_decision
permission_payload["permission_data"]["ml_reason"] = ml_reason
```

### Frontend Changes

#### 1.2 Update Permission Request UI

**File:** `frontend/js/websocket.js:296-453`

**Add ML score indicator after action description:**

```javascript
handlePermissionRequest(data) {
    // ... existing code ...

    const permissionData = data.permission_data;

    // ... existing contentHTML setup ...

    contentHTML += `
        <div class="permission-info">
            <h3>${toolEmoji} ${permissionData.tool_name} permission required</h3>
            <p><strong>Action:</strong> ${permissionData.action_description}</p>
    `;

    // ===== NEW: ML Score Indicator =====
    if (permissionData.ml_enabled && permissionData.ml_score !== null) {
        const mlScore = permissionData.ml_score;
        const mlConfidence = permissionData.ml_confidence || 0;

        // Determine color and icon based on score
        let scoreColor, scoreIcon, scoreText, confidencePercent;
        if (mlScore > 0.01) {
            scoreColor = '#10b981';  // green
            scoreIcon = '✅';
            scoreText = 'SAFE';
            confidencePercent = Math.min(mlConfidence * 100, 100);
        } else if (mlScore < -0.01) {
            scoreColor = '#ef4444';  // red
            scoreIcon = '⚠️';
            scoreText = 'RISKY';
            confidencePercent = Math.min(mlConfidence * 100, 100);
        } else {
            scoreColor = '#f59e0b';  // amber
            scoreIcon = '❓';
            scoreText = 'UNCERTAIN';
            confidencePercent = 50;
        }

        contentHTML += `
            <div class="ml-score-container">
                <div class="ml-score-header">
                    <span class="ml-score-label">🤖 ML Safety Score:</span>
                    <span class="ml-score-value" style="color: ${scoreColor}">
                        ${scoreIcon} ${scoreText} (${mlScore.toFixed(3)})
                    </span>
                </div>
                <div class="ml-score-bar-container">
                    <div class="ml-score-bar">
                        <div class="ml-score-fill" style="width: ${50 + (mlScore * 50)}%; background-color: ${scoreColor}"></div>
                        <div class="ml-score-markers">
                            <span class="ml-score-marker" style="left: 0%">-1</span>
                            <span class="ml-score-marker" style="left: 25%">-0.5</span>
                            <span class="ml-score-marker" style="left: 50%">0</span>
                            <span class="ml-score-marker" style="left: 75%">+0.5</span>
                            <span class="ml-score-marker" style="left: 100%">+1</span>
                        </div>
                    </div>
                </div>
                <div class="ml-confidence-container">
                    <span class="ml-confidence-label">Confidence:</span>
                    <div class="ml-confidence-bar">
                        <div class="ml-confidence-fill" style="width: ${confidencePercent}%; background-color: ${scoreColor}"></div>
                    </div>
                    <span class="ml-confidence-value">${confidencePercent.toFixed(0)}%</span>
                </div>
                <div class="ml-reason">
                    <details>
                        <summary>Why this score?</summary>
                        <p>${this.escapeHtml(permissionData.ml_reason || 'No reason provided')}</p>
                    </details>
                </div>
            </div>
        `;
    } else if (permissionData.ml_enabled) {
        contentHTML += `
            <div class="ml-score-container ml-disabled">
                <p>⚙️ ML classifier enabled but no score available</p>
            </div>
        `;
    }
    // ===== END ML Score Indicator =====

    // ... rest of existing code ...
}
```

#### 1.3 Add CSS Styles

**File:** `frontend/css/main.css` (or new `frontend/css/components/ml-score.css`)

```css
/* ML Score Indicator */
.ml-score-container {
    margin: 12px 0;
    padding: 12px;
    background: rgba(99, 102, 241, 0.05);
    border-left: 3px solid #6366f1;
    border-radius: 4px;
}

.ml-score-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.ml-score-label {
    font-weight: 500;
    color: #6b7280;
}

.ml-score-value {
    font-weight: 600;
    font-size: 1.05em;
}

/* Score Bar */
.ml-score-bar-container {
    position: relative;
    margin: 8px 0;
}

.ml-score-bar {
    position: relative;
    height: 24px;
    background: linear-gradient(to right,
        #ef4444 0%,
        #f59e0b 50%,
        #10b981 100%);
    border-radius: 12px;
    overflow: visible;
}

.ml-score-fill {
    position: absolute;
    top: -4px;
    left: 0;
    height: 32px;
    border: 2px solid white;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}

.ml-score-markers {
    position: absolute;
    top: 28px;
    width: 100%;
    display: flex;
    justify-content: space-between;
}

.ml-score-marker {
    font-size: 0.75em;
    color: #9ca3af;
}

/* Confidence Bar */
.ml-confidence-container {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
}

.ml-confidence-label {
    font-size: 0.9em;
    color: #6b7280;
    min-width: 80px;
}

.ml-confidence-bar {
    flex: 1;
    height: 8px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
}

.ml-confidence-fill {
    height: 100%;
    transition: width 0.3s ease;
}

.ml-confidence-value {
    font-size: 0.9em;
    font-weight: 600;
    min-width: 40px;
    text-align: right;
}

/* ML Reason Details */
.ml-reason {
    margin-top: 8px;
}

.ml-reason details {
    cursor: pointer;
}

.ml-reason summary {
    font-size: 0.9em;
    color: #6366f1;
    font-weight: 500;
}

.ml-reason summary:hover {
    text-decoration: underline;
}

.ml-reason p {
    margin: 8px 0 0 0;
    padding: 8px;
    background: rgba(99, 102, 241, 0.05);
    border-radius: 4px;
    font-size: 0.85em;
    color: #4b5563;
}

/* Disabled state */
.ml-score-container.ml-disabled {
    opacity: 0.6;
    border-left-color: #9ca3af;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .ml-score-container {
        background: rgba(99, 102, 241, 0.1);
    }

    .ml-score-label,
    .ml-confidence-label {
        color: #9ca3af;
    }

    .ml-score-marker {
        color: #6b7280;
    }

    .ml-reason p {
        background: rgba(99, 102, 241, 0.1);
        color: #d1d5db;
    }
}
```

---

## Phase 2: ML Configuration Panel (3-4 hours)

**Goal:** Add ML settings to the settings panel with interactive threshold tuning

### Backend Changes

#### 2.1 Add ML Endpoints to Permissions API

**File:** `api/permissions_api.py` (add these functions)

```python
async def get_ml_config(request: Any) -> Any:
    """GET /api/permissions/ml-config

    Get ML configuration.

    Returns:
        {
            "status": "success",
            "ml_config": {
                "enabled": false,
                "model": "embeddinggemma:latest",
                "threshold_allow": 0.01,
                "threshold_deny": -0.01,
                ...
            }
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.permissions.settings_manager import SettingsManager

        mgr = SettingsManager()
        ml_config = mgr.get_ml_config()

        return web.json_response({
            "status": "success",
            "ml_config": ml_config
        })

    except Exception as e:
        logger.error(f"Failed to get ML config: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "INTERNAL_ERROR"
        }, status=500)


async def update_ml_config(request: Any) -> Any:
    """POST /api/permissions/ml-config

    Update ML configuration.

    Request body:
        {
            "enabled": true,
            "threshold_allow": 0.02,
            "threshold_deny": -0.02,
            ...
        }

    Returns:
        {
            "status": "success",
            "message": "ML config updated",
            "ml_config": {...}
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.permissions.settings_manager import SettingsManager

        data = await request.json()

        mgr = SettingsManager()
        mgr.update_ml_config(**data)

        return web.json_response({
            "status": "success",
            "message": "ML configuration updated",
            "ml_config": mgr.get_ml_config()
        })

    except Exception as e:
        logger.error(f"Failed to update ML config: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "INTERNAL_ERROR"
        }, status=500)


async def get_ml_stats(request: Any) -> Any:
    """GET /api/permissions/ml-stats

    Get ML permission statistics.

    Returns:
        {
            "status": "success",
            "stats": {
                "total_checks": 156,
                "static_deny": 12,
                "ml_auto_allow": 89,
                ...
            }
        }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.permissions.settings_manager import SettingsManager
        from core.permissions.autonomous_permission import AutonomousPermission

        mgr = SettingsManager()
        ml_config = mgr.get_ml_config()

        if not ml_config.get("enabled", False):
            return web.json_response({
                "status": "success",
                "stats": None,
                "message": "ML not enabled"
            })

        # Create autonomous instance to get stats
        # NOTE: In production, this should be a singleton/global instance
        autonomous = AutonomousPermission(
            settings_manager=mgr,
            ml_enabled=True,
            ml_model=ml_config.get("model"),
            threshold_allow=ml_config.get("threshold_allow"),
            threshold_deny=ml_config.get("threshold_deny")
        )

        stats = autonomous.get_stats()

        return web.json_response({
            "status": "success",
            "stats": stats
        })

    except Exception as e:
        logger.error(f"Failed to get ML stats: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "error": str(e),
            "code": "INTERNAL_ERROR"
        }, status=500)


# Update setup_routes() to register new endpoints
def setup_routes(app: Any):
    """Register all permissions API routes"""
    # ... existing routes ...
    app.router.add_get("/api/permissions/ml-config", get_ml_config)
    app.router.add_post("/api/permissions/ml-config", update_ml_config)
    app.router.add_get("/api/permissions/ml-stats", get_ml_stats)
    logger.info("Permissions API routes registered (including ML endpoints)")
```

### Frontend Changes

#### 2.2 Add ML Section to Settings Panel

**File:** `frontend/index.html` (find settings panel, add ML section)

```html
<!-- Existing settings panel structure -->
<div id="settings-panel" class="settings-panel hidden">
    <!-- ... existing headers ... -->

    <!-- ===== NEW: ML Configuration Section ===== -->
    <div class="settings-section ml-config-section">
        <h3>🤖 ML Autonomous Permissions</h3>
        <p class="section-description">
            Machine learning-based classifier that auto-approves safe operations
            and flags risky ones. Reduce friction while maintaining security.
        </p>

        <!-- Enable/Disable Toggle -->
        <div class="ml-toggle-container">
            <label class="toggle-switch">
                <input type="checkbox" id="ml-enabled-toggle">
                <span class="toggle-slider"></span>
            </label>
            <span class="toggle-label">
                Enable ML Autonomous Permissions
            </span>
            <span id="ml-status-indicator" class="ml-status">Disabled</span>
        </div>

        <!-- ML Configuration (shown when enabled) -->
        <div id="ml-config-details" class="ml-config-details" style="display: none;">
            <!-- Model Selection -->
            <div class="config-row">
                <label for="ml-model-select">Embedding Model:</label>
                <select id="ml-model-select">
                    <option value="embeddinggemma:latest">embeddinggemma:latest (recommended)</option>
                    <option value="embeddinggemma:2b">embeddinggemma:2b (faster, less accurate)</option>
                    <option value="mxbai-embed-large">mxbai-embed-large (alternative)</option>
                </select>
            </div>

            <!-- Threshold Sliders -->
            <div class="config-row threshold-config">
                <label for="threshold-allow-slider">Auto-Allow Threshold:</label>
                <div class="slider-container">
                    <input type="range" id="threshold-allow-slider"
                           min="0.001" max="0.05" step="0.001" value="0.01">
                    <span id="threshold-allow-value" class="threshold-value">+0.010</span>
                </div>
                <p class="config-hint">
                    Scores above this are auto-approved.
                    Lower = more auto-approvals (less strict)
                </p>
            </div>

            <div class="config-row threshold-config">
                <label for="threshold-deny-slider">Ask-User Threshold:</label>
                <div class="slider-container">
                    <input type="range" id="threshold-deny-slider"
                           min="-0.05" max="-0.001" step="0.001" value="-0.01">
                    <span id="threshold-deny-value" class="threshold-value">-0.010</span>
                </div>
                <p class="config-hint">
                    Scores below this prompt with warning.
                    More negative = fewer warnings (less strict)
                </p>
            </div>

            <!-- Live Preview -->
            <div class="threshold-preview">
                <h4>Decision Zones Preview</h4>
                <div class="threshold-preview-bar">
                    <div class="preview-zone ask" id="preview-ask-zone">
                        <span class="zone-label">ASK</span>
                    </div>
                    <div class="preview-zone neutral" id="preview-neutral-zone">
                        <span class="zone-label">NEUTRAL</span>
                    </div>
                    <div class="preview-zone auto-allow" id="preview-allow-zone">
                        <span class="zone-label">AUTO-ALLOW</span>
                    </div>
                </div>
                <div class="preview-labels">
                    <span>-1.0 (unsafe)</span>
                    <span>0.0</span>
                    <span>+1.0 (safe)</span>
                </div>
            </div>

            <!-- Statistics -->
            <div class="ml-stats-container">
                <h4>ML Performance Statistics</h4>
                <button id="refresh-ml-stats" class="btn-secondary">
                    🔄 Refresh Stats
                </button>
                <div id="ml-stats-display" class="ml-stats-display">
                    <p class="loading">Loading statistics...</p>
                </div>
            </div>
        </div>
    </div>

    <!-- ... existing allow/deny/ask sections ... -->
</div>
```

#### 2.3 Add JavaScript ML Settings Manager

**File:** `frontend/js/ml-settings-manager.js` (new file)

```javascript
/**
 * ML Settings Manager
 * Handles ML configuration UI and API interactions
 */

class MLSettingsManager {
    constructor() {
        this.mlEnabled = false;
        this.mlConfig = null;

        // DOM elements
        this.toggleCheckbox = null;
        this.statusIndicator = null;
        this.configDetails = null;
        this.modelSelect = null;
        this.thresholdAllowSlider = null;
        this.thresholdDenySlider = null;
        this.thresholdAllowValue = null;
        this.thresholdDenyValue = null;
        this.refreshStatsBtn = null;
        this.statsDisplay = null;

        // Preview zones
        this.previewAskZone = null;
        this.previewNeutralZone = null;
        this.previewAllowZone = null;

        // Debounce timer for threshold changes
        this.updateDebounceTimer = null;
    }

    async init() {
        // Get DOM elements
        this.toggleCheckbox = document.getElementById('ml-enabled-toggle');
        this.statusIndicator = document.getElementById('ml-status-indicator');
        this.configDetails = document.getElementById('ml-config-details');
        this.modelSelect = document.getElementById('ml-model-select');
        this.thresholdAllowSlider = document.getElementById('threshold-allow-slider');
        this.thresholdDenySlider = document.getElementById('threshold-deny-slider');
        this.thresholdAllowValue = document.getElementById('threshold-allow-value');
        this.thresholdDenyValue = document.getElementById('threshold-deny-value');
        this.refreshStatsBtn = document.getElementById('refresh-ml-stats');
        this.statsDisplay = document.getElementById('ml-stats-display');

        // Preview zones
        this.previewAskZone = document.getElementById('preview-ask-zone');
        this.previewNeutralZone = document.getElementById('preview-neutral-zone');
        this.previewAllowZone = document.getElementById('preview-allow-zone');

        if (!this.toggleCheckbox) {
            console.warn('[MLSettingsManager] ML toggle not found');
            return;
        }

        // Bind events
        this.toggleCheckbox.addEventListener('change', () => this.onToggleML());
        this.modelSelect.addEventListener('change', () => this.onModelChange());
        this.thresholdAllowSlider.addEventListener('input', () => this.onThresholdChange());
        this.thresholdDenySlider.addEventListener('input', () => this.onThresholdChange());
        this.refreshStatsBtn.addEventListener('click', () => this.refreshStats());

        // Load current config
        await this.loadMLConfig();

        console.log('[MLSettingsManager] Initialized');
    }

    async loadMLConfig() {
        try {
            const response = await fetch('/api/permissions/ml-config');
            const data = await response.json();

            if (data.status === 'success') {
                this.mlConfig = data.ml_config;
                this.updateUI();
            } else {
                console.error('[MLSettingsManager] Failed to load ML config:', data.error);
            }
        } catch (error) {
            console.error('[MLSettingsManager] Error loading ML config:', error);
        }
    }

    updateUI() {
        if (!this.mlConfig) return;

        // Update toggle
        this.mlEnabled = this.mlConfig.enabled;
        this.toggleCheckbox.checked = this.mlEnabled;

        // Update status indicator
        this.statusIndicator.textContent = this.mlEnabled ? 'Enabled ✓' : 'Disabled';
        this.statusIndicator.className = this.mlEnabled ? 'ml-status enabled' : 'ml-status disabled';

        // Show/hide config details
        this.configDetails.style.display = this.mlEnabled ? 'block' : 'none';

        if (this.mlEnabled) {
            // Update model select
            this.modelSelect.value = this.mlConfig.model || 'embeddinggemma:latest';

            // Update threshold sliders
            this.thresholdAllowSlider.value = this.mlConfig.threshold_allow || 0.01;
            this.thresholdDenySlider.value = this.mlConfig.threshold_deny || -0.01;

            // Update value displays
            this.thresholdAllowValue.textContent = (+this.thresholdAllowSlider.value).toFixed(3);
            this.thresholdDenyValue.textContent = (+this.thresholdDenySlider.value).toFixed(3);

            // Update preview
            this.updatePreview();

            // Load stats
            this.refreshStats();
        }
    }

    async onToggleML() {
        const newEnabled = this.toggleCheckbox.checked;

        try {
            const response = await fetch('/api/permissions/ml-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: newEnabled })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.mlConfig = data.ml_config;
                this.updateUI();

                // Show notification
                this.showNotification(
                    newEnabled ?
                    '✅ ML permissions enabled' :
                    '⚠️ ML permissions disabled',
                    'success'
                );
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            console.error('[MLSettingsManager] Error toggling ML:', error);
            this.showNotification('❌ Failed to update ML settings', 'error');

            // Revert toggle
            this.toggleCheckbox.checked = !newEnabled;
        }
    }

    async onModelChange() {
        const newModel = this.modelSelect.value;

        try {
            await this.updateMLConfig({ model: newModel });
            this.showNotification('✅ Model updated. Restart to apply changes.', 'info');
        } catch (error) {
            console.error('[MLSettingsManager] Error updating model:', error);
            this.showNotification('❌ Failed to update model', 'error');
        }
    }

    onThresholdChange() {
        // Update value displays
        this.thresholdAllowValue.textContent = (+this.thresholdAllowSlider.value).toFixed(3);
        this.thresholdDenyValue.textContent = (+this.thresholdDenySlider.value).toFixed(3);

        // Update preview immediately
        this.updatePreview();

        // Debounce API call
        clearTimeout(this.updateDebounceTimer);
        this.updateDebounceTimer = setTimeout(() => {
            this.saveThresholds();
        }, 500);
    }

    async saveThresholds() {
        const thresholdAllow = parseFloat(this.thresholdAllowSlider.value);
        const thresholdDeny = parseFloat(this.thresholdDenySlider.value);

        try {
            await this.updateMLConfig({
                threshold_allow: thresholdAllow,
                threshold_deny: thresholdDeny
            });

            this.showNotification('✅ Thresholds saved', 'success');
        } catch (error) {
            console.error('[MLSettingsManager] Error saving thresholds:', error);
            this.showNotification('❌ Failed to save thresholds', 'error');
        }
    }

    async updateMLConfig(updates) {
        const response = await fetch('/api/permissions/ml-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates)
        });

        const data = await response.json();

        if (data.status === 'success') {
            this.mlConfig = data.ml_config;
            return data.ml_config;
        } else {
            throw new Error(data.error);
        }
    }

    updatePreview() {
        const thresholdAllow = parseFloat(this.thresholdAllowSlider.value);
        const thresholdDeny = parseFloat(this.thresholdDenySlider.value);

        // Calculate zone widths (scale -1 to +1 → 0% to 100%)
        const denyEnd = ((thresholdDeny + 1) / 2) * 100;
        const allowStart = ((thresholdAllow + 1) / 2) * 100;

        this.previewAskZone.style.width = `${denyEnd}%`;
        this.previewNeutralZone.style.width = `${allowStart - denyEnd}%`;
        this.previewAllowZone.style.width = `${100 - allowStart}%`;
    }

    async refreshStats() {
        if (!this.mlEnabled) {
            this.statsDisplay.innerHTML = '<p class="info">ML not enabled</p>';
            return;
        }

        this.statsDisplay.innerHTML = '<p class="loading">Loading statistics...</p>';

        try {
            const response = await fetch('/api/permissions/ml-stats');
            const data = await response.json();

            if (data.status === 'success' && data.stats) {
                this.displayStats(data.stats);
            } else {
                this.statsDisplay.innerHTML = '<p class="info">No statistics available yet</p>';
            }
        } catch (error) {
            console.error('[MLSettingsManager] Error loading stats:', error);
            this.statsDisplay.innerHTML = '<p class="error">Failed to load statistics</p>';
        }
    }

    displayStats(stats) {
        const total = stats.total_checks || 0;

        if (total === 0) {
            this.statsDisplay.innerHTML = '<p class="info">No permissions checked yet</p>';
            return;
        }

        const percentage = (count) => ((count / total) * 100).toFixed(1);

        const html = `
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-label">Total Checks</span>
                    <span class="stat-value">${total}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Static Deny</span>
                    <span class="stat-value">${stats.static_deny} (${percentage(stats.static_deny)}%)</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Static Allow</span>
                    <span class="stat-value">${stats.static_allow} (${percentage(stats.static_allow)}%)</span>
                </div>
                <div class="stat-item highlight">
                    <span class="stat-label">ML Auto-Allow</span>
                    <span class="stat-value">${stats.ml_auto_allow} (${percentage(stats.ml_auto_allow)}%)</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">ML Ask</span>
                    <span class="stat-value">${stats.ml_ask} (${percentage(stats.ml_ask)}%)</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">User Prompts</span>
                    <span class="stat-value">${stats.user_prompts} (${percentage(stats.user_prompts)}%)</span>
                </div>
            </div>

            ${stats.cache_stats ? `
                <div class="cache-stats">
                    <h5>Embedding Cache</h5>
                    <p>Cached: ${stats.cache_stats.cache_size} | Anchors: ${stats.cache_stats.anchor_count}</p>
                </div>
            ` : ''}
        `;

        this.statsDisplay.innerHTML = html;
    }

    showNotification(message, type = 'info') {
        // Use existing notification system if available
        if (window.showNotification) {
            window.showNotification(message, type);
        } else {
            console.log(`[MLSettingsManager] ${message}`);
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const mlSettingsManager = new MLSettingsManager();
    mlSettingsManager.init();

    // Expose globally for debugging
    window.mlSettingsManager = mlSettingsManager;
});
```

---

## Phase 3: Advanced Features (Optional, 4-5 hours)

### 3.1 Decision History Log

Track all ML decisions with filtering and export.

**UI Location:** New tab in settings panel

**Features:**
- Chronological list of ML decisions
- Filter by: auto-allowed, asked, denied
- Show: timestamp, tool, score, decision
- Export to CSV for analysis

### 3.2 Anchor Concept Explainability

Show which anchor concepts influenced the decision.

**Implementation:**
- Return top-N similar safe/unsafe anchors from classifier
- Display in permission prompt details panel
- Visual similarity bars

### 3.3 Per-Project Thresholds

Allow different threshold profiles per project.

**Implementation:**
- Store thresholds in project metadata
- Auto-detect project and load appropriate thresholds
- UI to manage project-specific configs

---

## Implementation Priority

### High Priority (Phase 1)
✅ ML score visualization in permission prompts
- **Effort:** 2-3 hours
- **Impact:** Immediate user value
- **Dependencies:** None

### Medium Priority (Phase 2)
⏳ ML configuration panel
- **Effort:** 3-4 hours
- **Impact:** User control and tuning
- **Dependencies:** Phase 1

### Low Priority (Phase 3)
⏸️ Advanced features
- **Effort:** 4-5 hours
- **Impact:** Power users
- **Dependencies:** Phase 2

---

## Testing Checklist

### Phase 1
- [ ] ML score appears in permission prompts
- [ ] Score bar correctly positions based on value
- [ ] Confidence indicator matches backend calculation
- [ ] Color coding (green/amber/red) matches safety level
- [ ] "Why this score?" details expand/collapse

### Phase 2
- [ ] ML toggle enables/disables classifier
- [ ] Threshold sliders update preview in real-time
- [ ] Configuration persists across page reloads
- [ ] Statistics refresh shows accurate data
- [ ] Model selection updates backend config

### Phase 3
- [ ] Decision history logs all ML decisions
- [ ] Export CSV contains correct data
- [ ] Anchor explainability shows relevant concepts
- [ ] Per-project thresholds load correctly

---

## Next Steps

1. **Implement Phase 1** (ML score visualization)
   - Update `permission_prompt.py` to include ML fields
   - Update `websocket.js` permission handler
   - Add CSS for ML score UI

2. **Test with real tool calls**
   - Enable ML: `/perms ml-enable`
   - Trigger various tool calls
   - Verify scores appear correctly

3. **Implement Phase 2** (configuration panel)
   - Add ML endpoints to `permissions_api.py`
   - Create `ml-settings-manager.js`
   - Update `index.html` settings panel

4. **Gather user feedback**
   - Monitor threshold usage patterns
   - Collect misclassification examples
   - Iterate on threshold defaults

---

**Last Updated:** 2025-11-17
**Status:** Ready for implementation
**Contact:** See `core/permissions/docs/AUTONOMOUS_PERMISSIONS.md`
