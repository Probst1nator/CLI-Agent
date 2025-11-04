/**
 * Tool Registry - Singleton for tool metadata management
 * Fetches and caches available tools from the server
 * Provides single source of truth for tool names and metadata
 */

import type { ToolDefinition } from '../types/index'

declare global {
  interface Window {
    ToolRegistry: ToolRegistry
    ToolManager?: {
      refresh: () => void
    }
  }
}

class ToolRegistry {
  private static _instance: ToolRegistry | null = null
  private tools: ToolDefinition[] = []
  private toolNames: string[] = []
  private toolMap: Map<string, ToolDefinition> = new Map()
  private isReady: boolean = false
  private isLoading: boolean = false
  private loadPromise: Promise<ToolDefinition[]> | null = null
  private _failedInitially: boolean = false
  private fallbackTools: ToolDefinition[] = []

  constructor() {
    if (ToolRegistry._instance) {
      return ToolRegistry._instance
    }

    ToolRegistry._instance = this
  }

  /**
   * Initialize tool registry from server
   */
  public async initialize(force: boolean = false): Promise<ToolDefinition[]> {
    if (this.isReady && !force) {
      return this.tools
    }

    if (this.isLoading && !force) {
      return this.loadPromise!
    }

    this.isLoading = true
    this.loadPromise = this.fetchTools()

    try {
      await this.loadPromise
    } finally {
      this.isLoading = false
    }

    return this.tools
  }

  /**
   * Fetch tools from backend API
   */
  private async fetchTools(): Promise<ToolDefinition[]> {
    try {
      const url = window.backendConnector ? await window.backendConnector.buildURL('/api/tools') : '/api/tools';
      (window as any).logDebug?.('[ToolRegistry] Fetching tools from:', url)
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()

      if (data.tools && Array.isArray(data.tools)) {
        this.tools = data.tools.map((tool: any) => ({
          name: tool.name || tool.delim,
          description: tool.description || tool.info?.description || '',
          enabled: tool.enabled !== false,
          metadata: tool.metadata || {}
        }))

        this.toolNames = this.tools.map(t => t.name)

        this.tools.forEach(tool => {
          this.toolMap.set(tool.name, tool)
        })

        this.isReady = true
        const wasFailedInitially = this._failedInitially
        this._failedInitially = false

        ;(window as any).visualLogger?.info(`ToolRegistry: Loaded ${this.tools.length} tools from server`)
        ;(window as any).logDebug?.('[ToolRegistry] Successfully loaded', this.tools.length, 'tools')

        if (wasFailedInitially && window.ToolManager) {
          ;(window as any).logDebug?.('[ToolRegistry] Retried after initial failure - refreshing ToolManager')
          window.ToolManager.refresh()
        }

        return this.tools
      } else {
        throw new Error('Invalid tools data format received from server')
      }
    } catch (error) {
      console.error('[ToolRegistry] Fetch failed:', error)
      ;(window as any).visualLogger?.error('ToolRegistry: Error fetching tools from server:', error)
      this.useFallbackTools()
      return []
    }
  }

  /**
   * Use fallback tools when API fetch fails (empty - relies on dynamic discovery)
   */
  private useFallbackTools(): void {
    this.tools = []
    this.toolNames = []
    this.toolMap.clear()
    this.isReady = true
    this._failedInitially = true
    ;(window as any).visualLogger?.warn('ToolRegistry: API fetch failed, will use dynamic tool discovery from XML content')
  }

  /**
   * Get array of tool names
   */
  public getToolNames(): string[] {
    if (!this.isReady) {
      ;(window as any).visualLogger?.warn('ToolRegistry: getToolNames called before ready, returning empty array (will use dynamic discovery)')
      return []
    }
    return this.toolNames
  }

  /**
   * Get array of tool definitions
   */
  public getTools(): ToolDefinition[] {
    if (!this.isReady) {
      ;(window as any).visualLogger?.warn('ToolRegistry: getTools called before ready, returning empty array (will use dynamic discovery)')
      return []
    }
    return this.tools
  }

  /**
   * Get tool definition by name
   */
  public getTool(name: string): ToolDefinition | undefined {
    return this.toolMap.get(name)
  }

  /**
   * Check if any tools are loaded
   */
  public hasTools(): boolean {
    return this.tools.length > 0
  }

  /**
   * Wait for tools to be loaded
   */
  public async waitForReady(): Promise<void> {
    if (this.isReady) {
      return
    }

    if (this.isLoading) {
      await this.loadPromise
      return
    }

    await this.initialize()
  }

  /**
   * Retry loading tools if initial fetch failed
   */
  public async retryIfFailed(): Promise<void> {
    if (this._failedInitially) {
      ;(window as any).logDebug?.('[ToolRegistry] Retrying after initial failure...')
      await this.initialize(true)
    }
  }
}

// Create singleton instance
window.ToolRegistry = new ToolRegistry()

export default ToolRegistry
export { ToolRegistry }
