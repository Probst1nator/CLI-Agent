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
 * @version 2.0.0 (TypeScript)
 */

import type { LogEntry, LoggerConfig } from '../types/index'

// Extend window interface for global logger access
declare global {
  interface Window {
    visualLogger: Logger
    sessionId?: string
    // Note: backendConnector type is defined in backend-connector.ts
    setLogLevel: (level: string) => boolean
    logDebug: (msg: string, data?: Record<string, any>) => LogEntry | undefined
    logInfo: (msg: string, data?: Record<string, any>) => LogEntry | undefined
    logWarn: (msg: string, data?: Record<string, any>) => LogEntry | undefined
    logError: (msg: string, data?: Record<string, any>, error?: Error | null) => LogEntry | undefined
    exportLogs: (options?: ExportOptions) => string
    downloadLogs: (filename?: string | null, options?: ExportOptions) => void
  }
}

type LogLevel = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'

interface PerformanceMarker {
  start: number
  startTime: number
}

interface ExportOptions {
  level?: LogLevel | null
  since?: string | number | null
  includeMetadata?: boolean
}

interface ExportData {
  logs: LogEntry[]
  metadata?: {
    sessionId: string
    exportTime: string
    totalLogs: number
    filteredLogs: number
    config: LoggerConfig
    url: string
    userAgent: string
  }
}

class Logger {
  public config: LoggerConfig
  private levels: Record<LogLevel, number>
  private sessionKey: string
  private startTime: number
  private sessionId: string
  private performanceMarkers: Map<string, PerformanceMarker>
  private logs: LogEntry[]

  /**
   * Initialize the logger with configuration options
   */
  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = {
      level: config.level || 'INFO',
      sendToBackend: config.sendToBackend ?? true,
      maxEntries: config.maxEntries || 1000,
      ...config
    }

    this.levels = {
      DEBUG: 0,
      INFO: 1,
      WARN: 2,
      ERROR: 3
    }

    this.sessionKey = 'visual-response-logs'
    this.startTime = Date.now()
    this.sessionId = this.generateSessionId()

    // Performance tracking
    this.performanceMarkers = new Map()

    // Initialize logs array
    this.logs = this.loadPersistedLogs()

    this.info('Logger initialized', {
      sessionId: this.sessionId,
      config: this.config,
      timestamp: new Date().toISOString()
    })
  }

  /**
   * Generate a unique session ID for this logging session
   */
  private generateSessionId(): string {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
  }

  /**
   * Check if a log level should be processed
   */
  private shouldLog(level: LogLevel): boolean {
    return this.levels[level] >= this.levels[this.config.level]
  }

  /**
   * Core logging method
   */
  public log(level: LogLevel, message: string, data: Record<string, any> = {}, error: Error | null = null): LogEntry | undefined {
    if (!this.shouldLog(level)) return

    const timestamp = Date.now()
    const logEntry: LogEntry = {
      id: this.generateLogId(),
      sessionId: this.sessionId,
      timestamp: timestamp,
      relativeTime: timestamp - this.startTime,
      level: level,
      message: message,
      data: data,
      url: window.location.href
    }

    // Add to logs array
    this.logs.push(logEntry)

    // Trim logs if necessary
    if (this.logs.length > this.config.maxEntries) {
      this.logs = this.logs.slice(-this.config.maxEntries)
    }

    // Persist logs
    this.persistLogs()

    // Output to console
    this.outputToConsole(logEntry)

    // Ship critical logs to backend if enabled
    if (this.config.sendToBackend && (level === 'ERROR' || level === 'WARN')) {
      this.shipLogToBackend(logEntry)
    }

    return logEntry
  }

  /**
   * Generate unique log entry ID
   */
  private generateLogId(): string {
    return Date.now() + '_' + Math.random().toString(36).substr(2, 6)
  }

  /**
   * Output log entry to browser console with appropriate styling
   */
  private outputToConsole(logEntry: LogEntry): void {
    const { level, message, data, relativeTime } = logEntry
    const timeStr = `[${(relativeTime / 1000).toFixed(3)}s]`
    const prefix = `[VR-${level}] ${timeStr}`

    // Console styling
    const styles: Record<LogLevel, string> = {
      DEBUG: 'color: #6c757d',
      INFO: 'color: #0d6efd',
      WARN: 'color: #fd7e14',
      ERROR: 'color: #dc3545; font-weight: bold'
    }

    const style = styles[level] || ''

    if (Object.keys(data).length > 0) {
      console.group(`%c${prefix} ${message}`, style)
      console.log('Data:', data)
      console.groupEnd()
    } else {
      console.log(`%c${prefix} ${message}`, style)
    }
  }

  /**
   * Ship log entry to backend for persistence
   */
  private async shipLogToBackend(logEntry: LogEntry): Promise<void> {
    // Don't block on network errors - fire and forget
    const payload = {
      level: logEntry.level,
      message: logEntry.message,
      stack: undefined,
      context: logEntry.data,
      url: logEntry.url,
      userAgent: navigator.userAgent.substring(0, 100),
      timestamp: new Date(logEntry.timestamp).toISOString()
    }

    try {
      // Use BackendConnector if available for multi-user support
      const url = window.backendConnector
        ? await window.backendConnector.buildURL('/api/logs/client')
        : '/api/logs/client'

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
        if (this.config.level === 'DEBUG') {
          console.error('Failed to ship log to backend:', err)
        }
      })
    } catch (err) {
      // Silently fail if URL building fails
      if (this.config.level === 'DEBUG') {
        console.error('Failed to build backend URL:', err)
      }
    }
  }

  /**
   * Load persisted logs from sessionStorage
   */
  private loadPersistedLogs(): LogEntry[] {
    try {
      const stored = sessionStorage.getItem(this.sessionKey)
      return stored ? JSON.parse(stored) : []
    } catch (error) {
      console.warn('Failed to load persisted logs:', error)
      return []
    }
  }

  /**
   * Persist logs to sessionStorage
   */
  private persistLogs(): void {
    try {
      // Only persist recent logs to avoid storage limits
      const recentLogs = this.logs.slice(-500)
      sessionStorage.setItem(this.sessionKey, JSON.stringify(recentLogs))
    } catch (error) {
      console.warn('Failed to persist logs:', error)
    }
  }

  /**
   * Start a performance timing marker
   */
  public startTiming(name: string): void {
    this.performanceMarkers.set(name, {
      start: performance.now(),
      startTime: Date.now()
    })
    this.debug(`Started timing: ${name}`)
  }

  /**
   * End a performance timing marker and log the duration
   */
  public endTiming(name: string, additionalData: Record<string, any> = {}): number | undefined {
    const marker = this.performanceMarkers.get(name)
    if (!marker) {
      this.warn(`No timing marker found for: ${name}`)
      return
    }

    const duration = performance.now() - marker.start
    this.performanceMarkers.delete(name)

    this.info(`Timing completed: ${name}`, {
      duration: `${duration.toFixed(2)}ms`,
      startTime: new Date(marker.startTime).toISOString(),
      ...additionalData
    })

    return duration
  }

  // Convenience methods for different log levels

  /**
   * Log debug message
   */
  public debug(message: string, data: Record<string, any> = {}): LogEntry | undefined {
    return this.log('DEBUG', message, data)
  }

  /**
   * Log info message
   */
  public info(message: string, data: Record<string, any> = {}): LogEntry | undefined {
    return this.log('INFO', message, data)
  }

  /**
   * Log warning message
   */
  public warn(message: string, data: Record<string, any> = {}): LogEntry | undefined {
    return this.log('WARN', message, data)
  }

  /**
   * Log error message
   */
  public error(message: string, data: Record<string, any> = {}, error: Error | null = null): LogEntry | undefined {
    return this.log('ERROR', message, data, error)
  }

  /**
   * Export logs as JSON for debugging
   */
  public exportLogs(options: ExportOptions = {}): string {
    const {
      level = null,
      since = null,
      includeMetadata = true
    } = options

    let filteredLogs = [...this.logs]

    // Filter by level if specified
    if (level) {
      const minLevel = this.levels[level]
      filteredLogs = filteredLogs.filter(log =>
        this.levels[log.level] >= minLevel
      )
    }

    // Filter by time if specified
    if (since) {
      const sinceTime = typeof since === 'string' ? new Date(since).getTime() : since
      filteredLogs = filteredLogs.filter(log => log.timestamp >= sinceTime)
    }

    const exportData: ExportData = {
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
    }

    return JSON.stringify(exportData, null, 2)
  }

  /**
   * Download logs as a file
   */
  public downloadLogs(filename: string | null = null, options: ExportOptions = {}): void {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    const defaultFilename = `visual-response-logs-${timestamp}.json`

    const logsJson = this.exportLogs(options)
    const blob = new Blob([logsJson], { type: 'application/json' })
    const url = URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = url
    a.download = filename || defaultFilename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    this.info('Logs downloaded', { filename: filename || defaultFilename })
  }

  /**
   * Clear all logs
   */
  public clearLogs(): void {
    this.logs = []
    sessionStorage.removeItem(this.sessionKey)
    this.info('Logs cleared')
  }

  /**
   * Get statistics about current logs
   */
  public getStats(): {
    total: number
    byLevel: Record<LogLevel, number>
    sessionDuration: number
    oldestLog: number | null
    newestLog: number | null
  } {
    const stats = {
      total: this.logs.length,
      byLevel: {} as Record<LogLevel, number>,
      sessionDuration: Date.now() - this.startTime,
      oldestLog: this.logs.length > 0 ? this.logs[0].timestamp : null,
      newestLog: this.logs.length > 0 ? this.logs[this.logs.length - 1].timestamp : null
    }

    // Count by level
    Object.keys(this.levels).forEach(level => {
      stats.byLevel[level as LogLevel] = this.logs.filter(log => log.level === level).length
    })

    return stats
  }
}

// Detect debug mode from URL parameter or localStorage
function getLogLevel(): LogLevel {
  // Check URL parameter: ?debug=true or ?loglevel=DEBUG
  const urlParams = new URLSearchParams(window.location.search)
  const debugParam = urlParams.get('debug')
  const logLevelParam = urlParams.get('loglevel')

  // Check localStorage for persistent setting
  const storedLevel = localStorage.getItem('vr-log-level')

  // Priority: URL param > localStorage > default
  if (logLevelParam) {
    const upperLevel = logLevelParam.toUpperCase()
    if (['DEBUG', 'INFO', 'WARN', 'ERROR'].includes(upperLevel)) {
      return upperLevel as LogLevel
    }
  }
  if (debugParam === 'true') {
    return 'DEBUG'
  }
  if (storedLevel && ['DEBUG', 'INFO', 'WARN', 'ERROR'].includes(storedLevel)) {
    return storedLevel as LogLevel
  }

  // Default to WARN to reduce console spam
  return 'WARN'
}

// Create global logger instance
window.visualLogger = new Logger({
  level: getLogLevel(),
  sendToBackend: true  // Ship ERROR and WARN logs to backend
})

// Add method to change log level at runtime
window.setLogLevel = function(level: string): boolean {
  const validLevels: LogLevel[] = ['DEBUG', 'INFO', 'WARN', 'ERROR']
  const upperLevel = level.toUpperCase()

  if (!validLevels.includes(upperLevel as LogLevel)) {
    console.error(`Invalid log level: ${level}. Valid levels: ${validLevels.join(', ')}`)
    return false
  }

  window.visualLogger.config.level = upperLevel as LogLevel
  localStorage.setItem('vr-log-level', upperLevel)
  console.log(`Log level changed to: ${upperLevel}`)
  return true
}

// Add convenience methods to global scope for easier debugging
window.logDebug = (msg: string, data?: Record<string, any>) => window.visualLogger.debug(msg, data || {})
window.logInfo = (msg: string, data?: Record<string, any>) => window.visualLogger.info(msg, data || {})
window.logWarn = (msg: string, data?: Record<string, any>) => window.visualLogger.warn(msg, data || {})
window.logError = (msg: string, data?: Record<string, any>, error?: Error | null) => window.visualLogger.error(msg, data || {}, error || null)
window.exportLogs = (options?: ExportOptions) => window.visualLogger.exportLogs(options || {})
window.downloadLogs = (filename?: string | null, options?: ExportOptions) => window.visualLogger.downloadLogs(filename, options || {})

// Log successful logger initialization
window.visualLogger.info('Visual Response Logger loaded and ready')

export default Logger
export { Logger, type LogLevel, type ExportOptions }
