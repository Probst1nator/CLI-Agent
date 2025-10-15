/**
 * Backend Connector - Discovers and connects to backend container
 * Handles UUID-based container discovery via Container Manager
 */

class BackendConnector {
    constructor(identityManager) {
        this.identityManager = identityManager;
        this.backendURL = null;
        this.containerName = null;
        this.isDiscovered = false;
        this.isLocal = this.detectLocalMode();
        this.discoveryPromise = null;
        this.readyCallbacks = [];
    }

    /**
     * Detect if running in local mode (single instance, no Container Manager)
     */
    detectLocalMode() {
        // In local mode, backend serves frontend directly
        // We can detect this by checking if we're on localhost
        return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    }

    /**
     * Wait for backend to be ready
     */
    async waitForReady() {
        if (this.isDiscovered) {
            return this.backendURL;
        }
        
        if (!this.discoveryPromise) {
            this.discoveryPromise = this.discoverBackend();
        }
        
        return this.discoveryPromise;
    }

    /**
     * Discover backend container via Container Manager
     * In local mode, uses current origin
     */
    async discoverBackend() {
        if (this.isLocal) {
            // Local mode: backend is same as frontend origin
            this.backendURL = window.location.origin;
            this.isDiscovered = true;
            console.log('Local mode: backend URL =', this.backendURL);
            this._notifyReady();
            return this.backendURL;
        }

        // Containerized mode: discover via Container Manager
        const uuid = this.identityManager.getUUID();

        try {
            const response = await fetch('/api/manager/discover', {
                headers: {
                    'X-User-UUID': uuid
                }
            });

            if (!response.ok) {
                throw new Error(`Discovery failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();

            if (data.status === 'success') {
                // Backend URL is proxied through Container Manager
                this.backendURL = `/api/backend/${uuid}`;
                this.containerName = data.container_name;
                this.isDiscovered = true;

                console.log('Backend discovered:', {
                    backendURL: this.backendURL,
                    containerName: this.containerName,
                    createdAt: data.created_at
                });

                this._notifyReady();
                return this.backendURL;
            } else {
                throw new Error('Discovery response missing success status');
            }
        } catch (error) {
            console.error('Backend discovery failed:', error);
            throw error;
        }
    }

    /**
     * Get WebSocket URL for backend
     */
    async getWebSocketURL() {
        if (!this.isDiscovered) {
            await this.discoverBackend();
        }

        if (this.isLocal) {
            // Local mode: direct WebSocket connection
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${protocol}//${window.location.host}/ws`;
        } else {
            // Containerized mode: WebSocket proxied through Container Manager
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const uuid = this.identityManager.getUUID();
            return `${protocol}//${window.location.host}/api/backend/${uuid}/ws`;
        }
    }

    /**
     * Build full URL for backend path
     */
    async buildURL(path) {
        if (!this.isDiscovered) {
            await this.discoverBackend();
        }

        return `${this.backendURL}${path}`;
    }

    /**
     * Fetch from backend with automatic discovery
     */
    async fetch(path, options = {}) {
        if (!this.isDiscovered) {
            await this.discoverBackend();
        }

        const url = `${this.backendURL}${path}`;

        // Add UUID header for activity tracking
        options.headers = options.headers || {};
        if (!this.isLocal) {
            options.headers['X-User-UUID'] = this.identityManager.getUUID();
        }

        return fetch(url, options);
    }

    /**
     * Update activity timestamp (keep container alive)
     */
    async updateActivity() {
        if (this.isLocal) {
            return; // No activity tracking in local mode
        }

        try {
            await fetch('/api/manager/activity', {
                method: 'POST',
                headers: {
                    'X-User-UUID': this.identityManager.getUUID()
                }
            });
        } catch (error) {
            console.warn('Failed to update activity:', error);
        }
    }

    /**
     * Start periodic activity updates (keep-alive)
     */
    startActivityUpdates(intervalMs = 60000) {
        if (this.isLocal) {
            return; // No keep-alive needed in local mode
        }

        this.activityInterval = setInterval(() => {
            this.updateActivity();
        }, intervalMs);

        console.log(`Started activity updates (interval: ${intervalMs}ms)`);
    }

    /**
     * Stop periodic activity updates
     */
    stopActivityUpdates() {
        if (this.activityInterval) {
            clearInterval(this.activityInterval);
            this.activityInterval = null;
            console.log('Stopped activity updates');
        }
    }

    /**
     * Notify all waiting callbacks that backend is ready
     */
    _notifyReady() {
        this.readyCallbacks.forEach(callback => callback(this.backendURL));
        this.readyCallbacks = [];
    }

    /**
     * Register callback for when backend is ready
     */
    onReady(callback) {
        if (this.isDiscovered) {
            callback(this.backendURL);
        } else {
            this.readyCallbacks.push(callback);
        }
    }
}

// Export for use in other modules
window.BackendConnector = BackendConnector;
