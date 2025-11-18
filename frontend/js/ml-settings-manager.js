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
            const response = await fetch('api/permissions/ml-config');
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
            const response = await fetch('api/permissions/ml-config', {
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
        const response = await fetch('api/permissions/ml-config', {
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
            const response = await fetch('api/permissions/ml-stats');
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
        // Try to use existing notification system if available
        if (window.showNotification) {
            window.showNotification(message, type);
        } else {
            // Fallback: console log
            console.log(`[MLSettingsManager] ${message}`);

            // Simple visual feedback
            const indicator = this.statusIndicator;
            const originalText = indicator.textContent;
            indicator.textContent = message;
            setTimeout(() => {
                indicator.textContent = originalText;
            }, 3000);
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
