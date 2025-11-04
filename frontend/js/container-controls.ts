/**
 * Container Controls - Manage container lifecycle from UI
 * Provides wipe/reload and status display functionality
 */

declare global {
  interface Window {
    ContainerControls?: typeof ContainerControls
    containerControls?: ContainerControls
  }
}

interface ContainerStatus {
  mode: 'local' | 'containerized'
  status: string
  uuid?: string
  created_at?: string
  last_activity?: string
  error?: string
  [key: string]: any
}

interface BackendConnector {
  isLocal: boolean
}

interface IdentityManager {
  getUUID: () => string
}

class ContainerControls {
  private identityManager: IdentityManager
  private backendConnector: BackendConnector
  private containerInfo: any = null

  constructor(identityManager: IdentityManager, backendConnector: BackendConnector) {
    this.identityManager = identityManager
    this.backendConnector = backendConnector
  }

  /**
   * Wipe container and reload
   */
  public async wipeAndReload(): Promise<void> {
    if (this.backendConnector.isLocal) {
      alert('Container wipe is only available in containerized mode')
      return
    }

    const confirmed = confirm(
      '⚠️ Wipe Container & Reload?\n\n' +
      'This will:\n' +
      '• Delete all chat history\n' +
      '• Remove all stored data\n' +
      '• Create a fresh container\n\n' +
      'This action cannot be undone.'
    )

    if (!confirmed) {
      return
    }

    const uuid = this.identityManager.getUUID()

    try {
      this.showLoadingIndicator('Wiping container...')

      const response = await fetch(`/api/manager/container/${uuid}/wipe`, {
        method: 'POST',
        headers: {
          'X-User-UUID': uuid
        }
      })

      if (!response.ok) {
        throw new Error(`Wipe failed: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()

      if (data.status === 'success') {
        console.log('Container wiped successfully:', data)

        // Clear local storage (optional - keeps UUID but clears other data)
        const uuid_backup = this.identityManager.getUUID()
        localStorage.clear()
        localStorage.setItem('cli-agent-user-uuid', uuid_backup)

        // Reload page to connect to new container
        window.location.reload()
      } else {
        throw new Error('Wipe response missing success status')
      }
    } catch (error) {
      console.error('Container wipe failed:', error)
      this.hideLoadingIndicator()
      alert(`Failed to wipe container: ${(error as Error).message}`)
    }
  }

  /**
   * Get container status info
   */
  public async getContainerStatus(): Promise<ContainerStatus> {
    if (this.backendConnector.isLocal) {
      return {
        mode: 'local',
        status: 'running'
      }
    }

    try {
      const response = await fetch('/api/manager/containers', {
        headers: {
          'X-User-UUID': this.identityManager.getUUID()
        }
      })

      if (!response.ok) {
        throw new Error(`Status fetch failed: ${response.status}`)
      }

      const data = await response.json()

      // Find our container in the list
      const uuid = this.identityManager.getUUID()
      const container = data.containers.find((c: any) => c.uuid.startsWith(uuid.substring(0, 8)))

      if (container) {
        this.containerInfo = container
        return {
          mode: 'containerized',
          status: 'running',
          ...container
        }
      }

      return {
        mode: 'containerized',
        status: 'unknown'
      }
    } catch (error) {
      console.error('Failed to get container status:', error)
      return {
        mode: 'containerized',
        status: 'error',
        error: (error as Error).message
      }
    }
  }

  /**
   * Render container controls UI
   */
  public renderControls(): string {
    const isLocal = this.backendConnector.isLocal

    if (isLocal) {
      return `
        <div class="container-controls local-mode">
          <div class="mode-badge">
            <span class="badge badge-info">🖥️ Local Mode</span>
          </div>
        </div>
      `
    }

    return `
      <div class="container-controls">
        <div class="mode-badge">
          <span class="badge badge-success">☁️ Containerized Mode</span>
        </div>
        <button class="btn btn-warning btn-sm" onclick="window.containerControls.wipeAndReload()">
          🔄 Wipe & Reload Container
        </button>
        <button class="btn btn-secondary btn-sm" onclick="window.containerControls.showStatusModal()">
          ℹ️ Status
        </button>
      </div>
    `
  }

  /**
   * Show status modal
   */
  public async showStatusModal(): Promise<void> {
    const status = await this.getContainerStatus()

    let statusHTML = '<h3>Container Status</h3>'
    statusHTML += `<p><strong>Mode:</strong> ${status.mode}</p>`
    statusHTML += `<p><strong>Status:</strong> ${status.status}</p>`

    if (status.uuid) {
      statusHTML += `<p><strong>UUID:</strong> ${status.uuid}</p>`
    }

    if (status.created_at) {
      statusHTML += `<p><strong>Created:</strong> ${new Date(status.created_at).toLocaleString()}</p>`
    }

    if (status.last_activity) {
      statusHTML += `<p><strong>Last Activity:</strong> ${new Date(status.last_activity).toLocaleString()}</p>`
    }

    if (status.error) {
      statusHTML += `<p class="error"><strong>Error:</strong> ${status.error}</p>`
    }

    alert(statusHTML.replace(/<[^>]*>/g, '\n'))
  }

  /**
   * Show loading indicator
   */
  private showLoadingIndicator(message: string): void {
    const indicator = document.createElement('div')
    indicator.id = 'container-loading'
    indicator.innerHTML = `<div class="loading-spinner"></div><p>${message}</p>`
    document.body.appendChild(indicator)
  }

  /**
   * Hide loading indicator
   */
  private hideLoadingIndicator(): void {
    const indicator = document.getElementById('container-loading')
    if (indicator) {
      indicator.remove()
    }
  }
}

export default ContainerControls
export { ContainerControls }
