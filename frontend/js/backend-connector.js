/**
 * Backend Connector - Discovers and connects to backend container
 * Handles UUID-based container discovery via Container Manager
 * @class
 */
class BackendConnector {
    /**
     * Initialize backend connector with identity manager
     * @param {IdentityManager} identityManager - Identity manager instance
     */
    constructor(identityManager) {
        /** @type {IdentityManager} */
        this.identityManager = identityManager;
        /** @type {string|null} Backend URL once discovered */
        this.backendURL = null;
        /** @type {string|null} Container name (containerized mode only) */
        this.containerName = null;
        /** @type {boolean} Whether backend has been discovered */
        this.isDiscovered = false;
        /** @type {boolean} Whether running in local mode */
        this.isLocal = this.detectLocalMode();
        /** @type {Promise<string>|null} Pending discovery promise */
        this.discoveryPromise = null;
        /** @type {Function[]} Callbacks waiting for backend ready */
        this.readyCallbacks = [];
        /** @type {number|null} Activity update interval timer */
        this.activityInterval = null;
    }

    /**
     * Detect if running in local mode (single instance, no Container Manager)
     * @returns {boolean} True if local mode, false if containerized
     */
    detectLocalMode() {
        return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    }

    /**
     * Wait for backend to be ready and discovered
     * @returns {Promise<string>} Backend URL
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
     * Discover backend container via Container Manager (or use local URL)
     * @returns {Promise<string>} Discovered backend URL
     * @throws {Error} If discovery fails
     */
    async discoverBackend() {
        if (this.isLocal) {
            this.backendURL = window.location.origin;
            this.isDiscovered = true;
            console.log('Local mode: backend URL =', this.backendURL);
            this._notifyReady();
            return this.backendURL;
        }

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
     * Get WebSocket URL for backend connection
     * @returns {Promise<string>} WebSocket URL
     */
    async getWebSocketURL() {
        if (!this.isDiscovered) {
            await this.discoverBackend();
        }

        if (this.isLocal) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${protocol}//${window.location.host}/ws`;
        } else {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const uuid = this.identityManager.getUUID();
            return `${protocol}//${window.location.host}/api/backend/${uuid}/ws`;
        }
    }

    /**
     * Build full URL for backend API path
     * @param {string} path - API path (e.g., "/api/chat/current")
     * @returns {Promise<string>} Full backend URL
     */
    async buildURL(path) {
        if (!this.isDiscovered) {
            await this.discoverBackend();
        }

        return `${this.backendURL}${path}`;
    }

    /**
     * Fetch from backend with automatic discovery and UUID headers
     * @param {string} path - API path to fetch
     * @param {RequestInit} [options={}] - Fetch options
     * @returns {Promise<Response>} Fetch response
     */
    async fetch(path, options = {}) {
        if (!this.isDiscovered) {
            await this.discoverBackend();
        }

        const url = `${this.backendURL}${path}`;

        options.headers = options.headers || {};
        if (!this.isLocal) {
            options.headers['X-User-UUID'] = this.identityManager.getUUID();
        }

        return fetch(url, options);
    }

    /**
     * Update activity timestamp to keep container alive
     * @returns {Promise<void>}
     */
    async updateActivity() {
        if (this.isLocal) {
            return;
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
     * Start periodic activity updates to prevent container timeout
     * @param {number} [intervalMs=60000] - Update interval in milliseconds
     */
    startActivityUpdates(intervalMs = 60000) {
        if (this.isLocal) {
            return;
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
     * @private
     */
    _notifyReady() {
        this.readyCallbacks.forEach(callback => callback(this.backendURL));
        this.readyCallbacks = [];
    }

    /**
     * Register callback for when backend is ready
     * @param {Function} callback - Callback receiving backend URL
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
