/**
 * Identity Manager - UUID-based user identification
 * Generates and stores UUID in localStorage for containerized mode
 */

class IdentityManager {
    constructor() {
        this.uuid = this.getOrCreateUUID();
        this.isContainerized = this.detectContainerizedMode();
        console.log(`Identity Manager initialized: UUID=${this.uuid.substring(0, 8)}..., Containerized=${this.isContainerized}`);
    }

    /**
     * Get or create user UUID
     * Uses crypto.randomUUID() with fallback for older browsers
     */
    getOrCreateUUID() {
        let uuid = localStorage.getItem('cli-agent-user-uuid');
        if (!uuid) {
            // Generate new UUID with fallback for older browsers
            if (typeof crypto.randomUUID === 'function') {
                uuid = crypto.randomUUID();
            } else {
                // Fallback UUID v4 generator
                uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                    const r = Math.random() * 16 | 0;
                    const v = c === 'x' ? r : (r & 0x3 | 0x8);
                    return v.toString(16);
                });
            }
            localStorage.setItem('cli-agent-user-uuid', uuid);
            console.log('Generated new user UUID:', uuid);
        }
        return uuid;
    }

    /**
     * Get current UUID
     */
    getUUID() {
        return this.uuid;
    }

    /**
     * Detect if running in containerized mode (frontend separated from backend)
     * In containerized mode, we need to discover backend via Container Manager
     */
    detectContainerizedMode() {
        // Check if we're running behind Container Manager
        // This can be detected by checking if /api/manager/discover endpoint exists
        return window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
    }

    /**
     * Reset UUID (wipe identity)
     * Used when user explicitly wants to start fresh
     */
    resetUUID() {
        const oldUUID = this.uuid;
        localStorage.removeItem('cli-agent-user-uuid');
        this.uuid = this.getOrCreateUUID();
        console.log(`UUID reset: ${oldUUID.substring(0, 8)}... → ${this.uuid.substring(0, 8)}...`);
        return this.uuid;
    }

    /**
     * Export UUID for backup
     */
    exportUUID() {
        return {
            uuid: this.uuid,
            created: localStorage.getItem('cli-agent-user-uuid-created') || new Date().toISOString()
        };
    }

    /**
     * Import UUID from backup
     */
    importUUID(uuid) {
        if (!uuid || typeof uuid !== 'string') {
            throw new Error('Invalid UUID format');
        }
        localStorage.setItem('cli-agent-user-uuid', uuid);
        localStorage.setItem('cli-agent-user-uuid-created', new Date().toISOString());
        this.uuid = uuid;
        console.log('UUID imported:', uuid.substring(0, 8) + '...');
    }
}

// Export for use in other modules
window.IdentityManager = IdentityManager;
