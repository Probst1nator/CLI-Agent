/**
 * Container Controls - Manage container lifecycle from UI
 * Provides wipe/reload and status display functionality
 */

class ContainerControls {
    constructor(identityManager, backendConnector) {
        this.identityManager = identityManager;
        this.backendConnector = backendConnector;
        this.containerInfo = null;
    }

    /**
     * Wipe container and reload
     * Destroys current container and creates a new one
     */
    async wipeAndReload() {
        if (this.backendConnector.isLocal) {
            alert('Container wipe is only available in containerized mode');
            return;
        }

        const confirmed = confirm(
            '⚠️ Wipe Container & Reload?\n\n' +
            'This will:\n' +
            '• Delete all chat history\n' +
            '• Remove all stored data\n' +
            '• Create a fresh container\n\n' +
            'This action cannot be undone.'
        );

        if (!confirmed) {
            return;
        }

        const uuid = this.identityManager.getUUID();

        try {
            // Show loading indicator
            this.showLoadingIndicator('Wiping container...');

            const response = await fetch(`/api/manager/container/${uuid}/wipe`, {
                method: 'POST',
                headers: {
                    'X-User-UUID': uuid
                }
            });

            if (!response.ok) {
                throw new Error(`Wipe failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();

            if (data.status === 'success') {
                console.log('Container wiped successfully:', data);

                // Clear local storage (optional - keeps UUID but clears other data)
                const uuid_backup = this.identityManager.getUUID();
                localStorage.clear();
                localStorage.setItem('cli-agent-user-uuid', uuid_backup);

                // Reload page to connect to new container
                window.location.reload();
            } else {
                throw new Error('Wipe response missing success status');
            }
        } catch (error) {
            console.error('Container wipe failed:', error);
            this.hideLoadingIndicator();
            alert(`Failed to wipe container: ${error.message}`);
        }
    }

    /**
     * Get container status info
     */
    async getContainerStatus() {
        if (this.backendConnector.isLocal) {
            return {
                mode: 'local',
                status: 'running'
            };
        }

        try {
            const response = await fetch('/api/manager/containers', {
                headers: {
                    'X-User-UUID': this.identityManager.getUUID()
                }
            });

            if (!response.ok) {
                throw new Error(`Status fetch failed: ${response.status}`);
            }

            const data = await response.json();

            // Find our container in the list
            const uuid = this.identityManager.getUUID();
            const container = data.containers.find(c => c.uuid.startsWith(uuid.substring(0, 8)));

            if (container) {
                this.containerInfo = container;
                return {
                    mode: 'containerized',
                    status: 'running',
                    ...container
                };
            }

            return {
                mode: 'containerized',
                status: 'unknown'
            };
        } catch (error) {
            console.error('Failed to get container status:', error);
            return {
                mode: 'containerized',
                status: 'error',
                error: error.message
            };
        }
    }

    /**
     * Render container controls UI
     */
    renderControls() {
        const isLocal = this.backendConnector.isLocal;

        if (isLocal) {
            return `
                <div class="container-controls local-mode">
                    <div class="mode-badge">
                        <span class="badge badge-info">🖥️ Local Mode</span>
                    </div>
                </div>
            `;
        }

        return `
            <div class="container-controls">
                <div class="mode-badge">
                    <span class="badge badge-success">☁️ Containerized Mode</span>
                </div>
                <button class="btn btn-warning btn-sm" onclick="containerControls.wipeAndReload()">
                    🔄 Wipe & Reload Container
                </button>
                <button class="btn btn-secondary btn-sm" onclick="containerControls.showStatusModal()">
                    📊 Container Status
                </button>
            </div>
        `;
    }

    /**
     * Show container status modal
     */
    async showStatusModal() {
        const status = await this.getContainerStatus();

        let content = '<div class="container-status-modal">';

        if (status.mode === 'local') {
            content += '<h3>Local Mode</h3>';
            content += '<p>Running in local development mode.</p>';
        } else if (status.status === 'running') {
            content += '<h3>Container Status</h3>';
            content += '<table class="status-table">';
            content += `<tr><th>UUID</th><td>${status.uuid || 'N/A'}</td></tr>`;
            content += `<tr><th>Container</th><td>${status.container_name || 'N/A'}</td></tr>`;
            content += `<tr><th>Age</th><td>${status.age_days || 0} days</td></tr>`;
            content += `<tr><th>Idle</th><td>${status.idle_days || 0} days</td></tr>`;
            content += `<tr><th>Retention</th><td>${status.retention_days || 7} days</td></tr>`;
            const daysRemaining = (status.retention_days || 7) - (status.idle_days || 0);
            content += `<tr><th>Auto-cleanup in</th><td>${Math.max(0, daysRemaining)} days</td></tr>`;
            content += '</table>';
        } else {
            content += '<h3>Status Unknown</h3>';
            content += `<p>Could not retrieve container status.</p>`;
        }

        content += '</div>';

        // Show modal (assuming a modal system exists)
        if (typeof showModal === 'function') {
            showModal('Container Status', content);
        } else {
            alert(content.replace(/<[^>]*>/g, '\n'));
        }
    }

    /**
     * Show loading indicator
     */
    showLoadingIndicator(message) {
        const indicator = document.createElement('div');
        indicator.id = 'loading-indicator';
        indicator.className = 'loading-indicator';
        indicator.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-message">${message}</div>
        `;
        document.body.appendChild(indicator);
    }

    /**
     * Hide loading indicator
     */
    hideLoadingIndicator() {
        const indicator = document.getElementById('loading-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
}

// Export for use in other modules
window.ContainerControls = ContainerControls;
