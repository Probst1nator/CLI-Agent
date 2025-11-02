/**
 * Visual Response Interface Logger
 *
 * Provides comprehensive logging capabilities for debugging and monitoring
 * the visual response system. Includes:
 * - Multiple log levels (DEBUG, INFO, WARN, ERROR)
 * - Session storage persistence
 * - Browser console integration
 * - Performance timing
 * - Error tracking
 * - Export capabilities for debugging
 *
 * @author CLI-Agent Development Team
 * @version 1.0.0
 */

class Logger {
    /**
     * Initialize the logger with configuration options
     * @param {Object} config - Logger configuration
     * @param {string} config.level - Minimum log level (DEBUG, INFO, WARN, ERROR)
     * @param {boolean} config.persist - Whether to persist logs in sessionStorage
     * @param {boolean} config.console - Whether to output to browser console
     * @param {number} config.maxEntries - Maximum number of log entries to keep
     */
    constructor(config = {}) {
        this.config = {
            level: config.level || 'INFO',
            persist: config.persist !== false, // Default true
            console: config.console !== false, // Default true
            maxEntries: config.maxEntries || 1000,
            ...config
        };

        this.levels = {
            DEBUG: 0,
            INFO: 1,
            WARN: 2,
            ERROR: 3
        };

        this.sessionKey = 'visual-response-logs';
        this.startTime = Date.now();
        this.sessionId = this.generateSessionId();

        // Performance tracking
        this.performanceMarkers = new Map();

        // Initialize logs array
        this.logs = this.loadPersistedLogs();

        this.info('Logger initialized', {
            sessionId: this.sessionId,
            config: this.config,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Generate a unique session ID for this logging session
     * @returns {string} Session identifier
     */
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Check if a log level should be processed
     * @param {string} level - Log level to check
     * @returns {boolean} Whether this level should be logged
     */
    shouldLog(level) {
        return this.levels[level] >= this.levels[this.config.level];
    }

    /**
     * Core logging method
     * @param {string} level - Log level
     * @param {string} message - Log message
     * @param {Object} data - Additional data to log
     * @param {Error} error - Optional error object
     */
    log(level, message, data = {}, error = null) {
        if (!this.shouldLog(level)) return;

        const timestamp = Date.now();
        const logEntry = {
            id: this.generateLogId(),
            sessionId: this.sessionId,
            timestamp: timestamp,
            relativeTime: timestamp - this.startTime,
            level: level,
            message: message,
            data: data,
            error: error ? {
                name: error.name,
                message: error.message,
                stack: error.stack
            } : null,
            url: window.location.href,
            userAgent: navigator.userAgent.substring(0, 100) // Truncate for storage
        };

        // Add to logs array
        this.logs.push(logEntry);

        // Trim logs if necessary
        if (this.logs.length > this.config.maxEntries) {
            this.logs = this.logs.slice(-this.config.maxEntries);
        }

        // Persist if enabled
        if (this.config.persist) {
            this.persistLogs();
        }

        // Output to console if enabled
        if (this.config.console) {
            this.outputToConsole(logEntry);
        }

        // Ship critical logs to backend if enabled
        if (this.config.shipToBackend && (level === 'ERROR' || level === 'WARN')) {
            this.shipLogToBackend(logEntry);
        }

        return logEntry;
    }

    /**
     * Generate unique log entry ID
     * @returns {string} Log entry identifier
     */
    generateLogId() {
        return Date.now() + '_' + Math.random().toString(36).substr(2, 6);
    }

    /**
     * Output log entry to browser console with appropriate styling
     * @param {Object} logEntry - Log entry to output
     */
    outputToConsole(logEntry) {
        const { level, message, data, error, relativeTime } = logEntry;
        const timeStr = `[${(relativeTime / 1000).toFixed(3)}s]`;
        const prefix = `[VR-${level}] ${timeStr}`;

        // Console styling
        const styles = {
            DEBUG: 'color: #6c757d',
            INFO: 'color: #0d6efd',
            WARN: 'color: #fd7e14',
            ERROR: 'color: #dc3545; font-weight: bold'
        };

        const style = styles[level] || '';

        if (error) {
            console.group(`%c${prefix} ${message}`, style);
            console.error('Error:', error);
            if (Object.keys(data).length > 0) {
                console.log('Data:', data);
            }
            console.groupEnd();
        } else if (Object.keys(data).length > 0) {
            console.group(`%c${prefix} ${message}`, style);
            console.log('Data:', data);
            console.groupEnd();
        } else {
            console.log(`%c${prefix} ${message}`, style);
        }
    }

    /**
     * Ship log entry to backend for persistence
     * @param {Object} logEntry - Log entry to ship
     */
    async shipLogToBackend(logEntry) {
        // Don't block on network errors - fire and forget
        const payload = {
            level: logEntry.level,
            message: logEntry.message,
            stack: logEntry.error?.stack,
            context: logEntry.data,
            url: logEntry.url,
            userAgent: logEntry.userAgent,
            timestamp: new Date(logEntry.timestamp).toISOString()
        };

        try {
            // Use BackendConnector if available for multi-user support
            const url = window.backendConnector
                ? await window.backendConnector.buildURL('/api/logs/client')
                : '/api/logs/client';

            fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': window.sessionId || 'unknown',
                    'X-Request-ID': logEntry.data?.requestId || 'unknown'
                },
                body: JSON.stringify(payload)
            }).catch(err => {
                // Silently fail to avoid error loops
                // Only log to console if in debug mode
                if (this.config.level === 'DEBUG') {
                    console.error('Failed to ship log to backend:', err);
                }
            });
        } catch (err) {
            // Silently fail if URL building fails
            if (this.config.level === 'DEBUG') {
                console.error('Failed to build backend URL:', err);
            }
        }
    }

    /**
     * Load persisted logs from sessionStorage
     * @returns {Array} Array of persisted log entries
     */
    loadPersistedLogs() {
        if (!this.config.persist) return [];

        try {
            const stored = sessionStorage.getItem(this.sessionKey);
            return stored ? JSON.parse(stored) : [];
        } catch (error) {
            console.warn('Failed to load persisted logs:', error);
            return [];
        }
    }

    /**
     * Persist logs to sessionStorage
     */
    persistLogs() {
        if (!this.config.persist) return;

        try {
            // Only persist recent logs to avoid storage limits
            const recentLogs = this.logs.slice(-500);
            sessionStorage.setItem(this.sessionKey, JSON.stringify(recentLogs));
        } catch (error) {
            console.warn('Failed to persist logs:', error);
        }
    }

    /**
     * Start a performance timing marker
     * @param {string} name - Name of the operation to time
     */
    startTiming(name) {
        this.performanceMarkers.set(name, {
            start: performance.now(),
            startTime: Date.now()
        });
        this.debug(`Started timing: ${name}`);
    }

    /**
     * End a performance timing marker and log the duration
     * @param {string} name - Name of the operation that was timed
     * @param {Object} additionalData - Additional data to include in timing log
     */
    endTiming(name, additionalData = {}) {
        const marker = this.performanceMarkers.get(name);
        if (!marker) {
            this.warn(`No timing marker found for: ${name}`);
            return;
        }

        const duration = performance.now() - marker.start;
        this.performanceMarkers.delete(name);

        this.info(`Timing completed: ${name}`, {
            duration: `${duration.toFixed(2)}ms`,
            startTime: new Date(marker.startTime).toISOString(),
            ...additionalData
        });

        return duration;
    }

    // Convenience methods for different log levels

    /**
     * Log debug message
     * @param {string} message - Debug message
     * @param {Object} data - Additional debug data
     */
    debug(message, data = {}) {
        return this.log('DEBUG', message, data);
    }

    /**
     * Log info message
     * @param {string} message - Info message
     * @param {Object} data - Additional info data
     */
    info(message, data = {}) {
        return this.log('INFO', message, data);
    }

    /**
     * Log warning message
     * @param {string} message - Warning message
     * @param {Object} data - Additional warning data
     */
    warn(message, data = {}) {
        return this.log('WARN', message, data);
    }

    /**
     * Log error message
     * @param {string} message - Error message
     * @param {Object} data - Additional error data
     * @param {Error} error - Error object
     */
    error(message, data = {}, error = null) {
        return this.log('ERROR', message, data, error);
    }

    /**
     * Export logs as JSON for debugging
     * @param {Object} options - Export options
     * @returns {string} JSON string of logs
     */
    exportLogs(options = {}) {
        const {
            level = null,
            since = null,
            includeMetadata = true
        } = options;

        let filteredLogs = [...this.logs];

        // Filter by level if specified
        if (level) {
            const minLevel = this.levels[level];
            filteredLogs = filteredLogs.filter(log =>
                this.levels[log.level] >= minLevel
            );
        }

        // Filter by time if specified
        if (since) {
            const sinceTime = typeof since === 'string' ? new Date(since).getTime() : since;
            filteredLogs = filteredLogs.filter(log => log.timestamp >= sinceTime);
        }

        const exportData = {
            logs: filteredLogs,
            ...(includeMetadata && {
                metadata: {
                    sessionId: this.sessionId,
                    exportTime: new Date().toISOString(),
                    totalLogs: this.logs.length,
                    filteredLogs: filteredLogs.length,
                    config: this.config,
                    url: window.location.href,
                    userAgent: navigator.userAgent
                }
            })
        };

        return JSON.stringify(exportData, null, 2);
    }

    /**
     * Download logs as a file
     * @param {string} filename - Name of the download file
     * @param {Object} options - Export options
     */
    downloadLogs(filename = null, options = {}) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const defaultFilename = `visual-response-logs-${timestamp}.json`;

        const logsJson = this.exportLogs(options);
        const blob = new Blob([logsJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename || defaultFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.info('Logs downloaded', { filename: filename || defaultFilename });
    }

    /**
     * Clear all logs
     */
    clearLogs() {
        this.logs = [];
        if (this.config.persist) {
            sessionStorage.removeItem(this.sessionKey);
        }
        this.info('Logs cleared');
    }

    /**
     * Get statistics about current logs
     * @returns {Object} Log statistics
     */
    getStats() {
        const stats = {
            total: this.logs.length,
            byLevel: {},
            sessionDuration: Date.now() - this.startTime,
            oldestLog: this.logs.length > 0 ? this.logs[0].timestamp : null,
            newestLog: this.logs.length > 0 ? this.logs[this.logs.length - 1].timestamp : null
        };

        // Count by level
        Object.keys(this.levels).forEach(level => {
            stats.byLevel[level] = this.logs.filter(log => log.level === level).length;
        });

        return stats;
    }
}

// Detect debug mode from URL parameter or localStorage
function getLogLevel() {
    // Check URL parameter: ?debug=true or ?loglevel=DEBUG
    const urlParams = new URLSearchParams(window.location.search);
    const debugParam = urlParams.get('debug');
    const logLevelParam = urlParams.get('loglevel');

    // Check localStorage for persistent setting
    const storedLevel = localStorage.getItem('vr-log-level');

    // Priority: URL param > localStorage > default
    if (logLevelParam) {
        return logLevelParam.toUpperCase();
    }
    if (debugParam === 'true') {
        return 'DEBUG';
    }
    if (storedLevel) {
        return storedLevel;
    }

    // Default to WARN to reduce console spam
    return 'WARN';
}

// Create global logger instance
window.visualLogger = new Logger({
    level: getLogLevel(),
    persist: true,
    console: true,
    shipToBackend: true  // Ship ERROR and WARN logs to backend
});

// Add method to change log level at runtime
window.setLogLevel = function(level) {
    const validLevels = ['DEBUG', 'INFO', 'WARN', 'ERROR'];
    const upperLevel = level.toUpperCase();

    if (!validLevels.includes(upperLevel)) {
        console.error(`Invalid log level: ${level}. Valid levels: ${validLevels.join(', ')}`);
        return false;
    }

    window.visualLogger.config.level = upperLevel;
    localStorage.setItem('vr-log-level', upperLevel);
    console.log(`Log level changed to: ${upperLevel}`);
    return true;
};

// Add convenience methods to global scope for easier debugging
window.logDebug = (msg, data) => window.visualLogger.debug(msg, data);
window.logInfo = (msg, data) => window.visualLogger.info(msg, data);
window.logWarn = (msg, data) => window.visualLogger.warn(msg, data);
window.logError = (msg, data, error) => window.visualLogger.error(msg, data, error);
window.exportLogs = (options) => window.visualLogger.exportLogs(options);
window.downloadLogs = (filename, options) => window.visualLogger.downloadLogs(filename, options);

// Log successful logger initialization
window.visualLogger.info('Visual Response Logger loaded and ready');