/**
 * Identity Manager - UUID-based user identification
 * Generates and stores UUID in localStorage for containerized mode
 */

declare global {
  interface Window {
    IdentityManager: typeof IdentityManager
  }
}

interface UUIDBackup {
  uuid: string
  created: string
}

class IdentityManager {
  private uuid: string
  private isContainerized: boolean

  constructor() {
    this.uuid = this.getOrCreateUUID()
    this.isContainerized = this.detectContainerizedMode()
    console.log(`Identity Manager initialized: UUID=${this.uuid.substring(0, 8)}..., Containerized=${this.isContainerized}`)
  }

  /**
   * Get or create user UUID from localStorage
   */
  private getOrCreateUUID(): string {
    let uuid = localStorage.getItem('cli-agent-user-uuid')
    if (!uuid) {
      // Generate new UUID with fallback for older browsers
      if (typeof crypto.randomUUID === 'function') {
        uuid = crypto.randomUUID()
      } else {
        // Fallback UUID v4 generator
        uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
          const r = Math.random() * 16 | 0
          const v = c === 'x' ? r : (r & 0x3 | 0x8)
          return v.toString(16)
        })
      }
      localStorage.setItem('cli-agent-user-uuid', uuid)
      console.log('Generated new user UUID:', uuid)
    }
    return uuid
  }

  /**
   * Get current UUID
   */
  public getUUID(): string {
    return this.uuid
  }

  /**
   * Detect if running in containerized mode (frontend separated from backend)
   */
  private detectContainerizedMode(): boolean {
    // Check if we're running behind Container Manager
    return window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1'
  }

  /**
   * Reset UUID (wipe identity) and generate new one
   */
  public resetUUID(): string {
    const oldUUID = this.uuid
    localStorage.removeItem('cli-agent-user-uuid')
    this.uuid = this.getOrCreateUUID()
    console.log(`UUID reset: ${oldUUID.substring(0, 8)}... → ${this.uuid.substring(0, 8)}...`)
    return this.uuid
  }

  /**
   * Export UUID data for backup
   */
  public exportUUID(): UUIDBackup {
    return {
      uuid: this.uuid,
      created: localStorage.getItem('cli-agent-user-uuid-created') || new Date().toISOString()
    }
  }

  /**
   * Import UUID from backup
   */
  public importUUID(uuid: string): void {
    if (!uuid || typeof uuid !== 'string') {
      throw new Error('Invalid UUID format')
    }
    localStorage.setItem('cli-agent-user-uuid', uuid)
    localStorage.setItem('cli-agent-user-uuid-created', new Date().toISOString())
    this.uuid = uuid
    console.log('UUID imported:', uuid.substring(0, 8) + '...')
  }
}

// Export for use in other modules
window.IdentityManager = IdentityManager

export default IdentityManager
export { IdentityManager }
