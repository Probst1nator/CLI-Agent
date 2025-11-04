/**
 * Backend Connector - Discovers and connects to backend container
 * Handles UUID-based container discovery via Container Manager
 */

declare global {
  interface Window {
    BackendConnector: typeof BackendConnector
  }
}

interface IdentityManager {
  getUUID: () => string
}

class BackendConnector {
  private identityManager: IdentityManager
  private backendURL: string | null = null
  private containerName: string | null = null
  private isDiscovered: boolean = false
  private isLocal: boolean
  private discoveryPromise: Promise<string> | null = null
  private readyCallbacks: Array<(url: string) => void> = []
  private activityInterval: number | null = null

  constructor(identityManager: IdentityManager) {
    this.identityManager = identityManager
    this.isLocal = this.detectLocalMode()
  }

  /**
   * Detect if running in local mode (single instance, no Container Manager)
   */
  private detectLocalMode(): boolean {
    return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  }

  /**
   * Wait for backend to be ready and discovered
   */
  public async waitForReady(): Promise<string> {
    if (this.isDiscovered) {
      return this.backendURL!
    }

    if (!this.discoveryPromise) {
      this.discoveryPromise = this.discoverBackend()
    }

    return this.discoveryPromise
  }

  /**
   * Discover backend container via Container Manager (or use local URL)
   */
  public async discoverBackend(): Promise<string> {
    if (this.isLocal) {
      this.backendURL = window.location.origin
      this.isDiscovered = true
      console.log('Local mode: backend URL =', this.backendURL)
      this._notifyReady()
      return this.backendURL
    }

    const uuid = this.identityManager.getUUID()

    try {
      const response = await fetch('/api/manager/discover', {
        headers: {
          'X-User-UUID': uuid
        }
      })

      if (!response.ok) {
        throw new Error(`Discovery failed: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()

      if (data.status === 'success') {
        this.backendURL = `/api/backend/${uuid}`
        this.containerName = data.container_name
        this.isDiscovered = true

        console.log('Backend discovered:', {
          backendURL: this.backendURL,
          containerName: this.containerName,
          createdAt: data.created_at
        })

        this._notifyReady()
        return this.backendURL
      } else {
        throw new Error('Discovery response missing success status')
      }
    } catch (error) {
      console.error('Backend discovery failed:', error)
      throw error
    }
  }

  /**
   * Get WebSocket URL for backend connection
   */
  public async getWebSocketURL(): Promise<string> {
    if (!this.isDiscovered) {
      await this.discoverBackend()
    }

    if (this.isLocal) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      return `${protocol}//${window.location.host}/ws`
    } else {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const uuid = this.identityManager.getUUID()
      return `${protocol}//${window.location.host}/api/backend/${uuid}/ws`
    }
  }

  /**
   * Build full URL for backend API path
   */
  public async buildURL(path: string): Promise<string> {
    if (!this.isDiscovered) {
      await this.discoverBackend()
    }

    return `${this.backendURL}${path}`
  }

  /**
   * Fetch from backend with automatic discovery and UUID headers
   */
  public async fetch(path: string, options: RequestInit = {}): Promise<Response> {
    if (!this.isDiscovered) {
      await this.discoverBackend()
    }

    const url = `${this.backendURL}${path}`

    options.headers = options.headers || {}
    if (!this.isLocal) {
      (options.headers as Record<string, string>)['X-User-UUID'] = this.identityManager.getUUID()
    }

    return fetch(url, options)
  }

  /**
   * Update activity timestamp to keep container alive
   */
  public async updateActivity(): Promise<void> {
    if (this.isLocal) {
      return
    }

    try {
      await fetch('/api/manager/activity', {
        method: 'POST',
        headers: {
          'X-User-UUID': this.identityManager.getUUID()
        }
      })
    } catch (error) {
      console.warn('Failed to update activity:', error)
    }
  }

  /**
   * Start periodic activity updates to prevent container timeout
   */
  public startActivityUpdates(intervalMs: number = 60000): void {
    if (this.isLocal) {
      return
    }

    this.activityInterval = window.setInterval(() => {
      this.updateActivity()
    }, intervalMs)

    console.log(`Started activity updates (interval: ${intervalMs}ms)`)
  }

  /**
   * Stop periodic activity updates
   */
  public stopActivityUpdates(): void {
    if (this.activityInterval) {
      clearInterval(this.activityInterval)
      this.activityInterval = null
      console.log('Stopped activity updates')
    }
  }

  /**
   * Notify all waiting callbacks that backend is ready
   */
  private _notifyReady(): void {
    this.readyCallbacks.forEach(callback => callback(this.backendURL!))
    this.readyCallbacks = []
  }

  /**
   * Register callback for when backend is ready
   */
  public onReady(callback: (url: string) => void): void {
    if (this.isDiscovered) {
      callback(this.backendURL!)
    } else {
      this.readyCallbacks.push(callback)
    }
  }
}

// Export for use in other modules
window.BackendConnector = BackendConnector

export default BackendConnector
export { BackendConnector }
