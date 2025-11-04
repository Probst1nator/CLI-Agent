/**
 * Auth Manager - JWT authentication for SaaS mode
 * Handles JWT token exchange and authentication headers
 * @class
 */
class AuthManager {
    /**
     * Initialize auth manager with identity manager
     * @param {IdentityManager} identityManager - Identity manager instance
     */
    constructor(identityManager) {
        this.identityManager = identityManager;
        this.token = localStorage.getItem('cli-agent-jwt-token');
        this.tokenExpiry = localStorage.getItem('cli-agent-jwt-expiry');
        console.log('Auth Manager initialized');
    }

    /**
     * Check if token is expired
     * @returns {boolean} True if token is expired or missing
     */
    isTokenExpired() {
        if (!this.token || !this.tokenExpiry) {
            return true;
        }
        const expiryTime = parseInt(this.tokenExpiry);
        const now = Date.now() / 1000; // Convert to seconds
        return now >= expiryTime;
    }

    /**
     * Ensure we have a valid JWT token
     * @returns {Promise<string|null>} JWT token or null on failure
     */
    async ensureAuthenticated() {
        // Check if we have a valid token
        if (this.token && !this.isTokenExpired()) {
            return this.token;
        }

        // Token expired or missing - get new one
        return await this.refreshToken();
    }

    /**
     * Refresh JWT token by exchanging UUID
     * @returns {Promise<string|null>} New JWT token or null on failure
     */
    async refreshToken() {
        // Get UUID from identity manager
        const uuid = this.identityManager.getUUID();

        try {
            const response = await fetch('/api/manager/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({uuid})
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.token = data.token;
                const expiryTime = Math.floor(Date.now() / 1000) + data.expires_in;
                this.tokenExpiry = expiryTime.toString();

                // Store in localStorage
                localStorage.setItem('cli-agent-jwt-token', this.token);
                localStorage.setItem('cli-agent-jwt-expiry', this.tokenExpiry);

                console.log('JWT token obtained, expires in', data.expires_in, 'seconds');
                return this.token;
            } else {
                console.error('Authentication failed:', data.error);
                this.clearAuth();
                return null;
            }
        } catch (error) {
            console.error('Authentication error:', error);
            this.clearAuth();
            return null;
        }
    }

    /**
     * Get authentication headers for API requests
     * @returns {Object} Headers object with Authorization and X-User-UUID
     */
    getAuthHeaders() {
        return {
            'Authorization': `Bearer ${this.token}`,
            'X-User-UUID': this.identityManager.getUUID()
        };
    }

    /**
     * Clear authentication data
     */
    clearAuth() {
        this.token = null;
        this.tokenExpiry = null;
        localStorage.removeItem('cli-agent-jwt-token');
        localStorage.removeItem('cli-agent-jwt-expiry');
        console.log('Authentication cleared');
    }

    /**
     * Check if running in containerized mode (needs JWT)
     * @returns {boolean} True if containerized mode
     */
    isContainerizedMode() {
        return this.identityManager.isContainerized;
    }
}

// Export for use in other modules
window.AuthManager = AuthManager;
