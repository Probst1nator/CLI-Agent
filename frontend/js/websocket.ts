/**
 * WebSocket Communication Module
 *
 * Manages real-time bidirectional communication between the browser client
 * and the CLI-Agent visual server.
 *
 * @author CLI-Agent Development Team
 * @version 2.0.0 (TypeScript)
 */

import type { WSMessage, EventHandler, EventHandlersRegistry } from '../types/websocket'

// Type definitions for external dependencies
interface ContentRenderer {
  renderContent: (container: HTMLElement, data: any, messageId: string, timestampDiv: HTMLElement) => Promise<void>
}

interface MainApp {
  loadMessageHistory?: () => void
  clearMessageHistory?: () => void
  hideLoadingIndicator?: () => void
  hideInterruptButton?: () => void
  showInterruptButton?: () => void
  displayUserMessage?: (message: string) => void
  displayAssistantMessage?: (message: string) => void
  setScreenshotReady?: () => void
  resetScreenshotState?: () => void
  saveMessageToHistory?: (role: string, data: any) => void
  isRestoring?: boolean
  messageHistory?: any[]
  currentOptimisticMessageId?: string | null
}

interface BackendConnector {
  getWebSocketURL: () => Promise<string>
  buildURL: (path: string) => Promise<string>
  fetch: (path: string, options?: RequestInit) => Promise<Response>
}

interface ChatStateManager {
  addMessage: (message: any) => void
}

interface ConsoleLogManager {
  addLogLine: (line: string, level: string) => void
}

declare global {
  interface Window {
    WebSocketManager: typeof WebSocketManager
    sessionId?: string
    chatState?: ChatStateManager
    audioManager?: {
      playWebSocketAudio: (data: { audio_data: string; text?: string; audio_format?: string; duration?: number; language?: string }) => Promise<void>
    }
    consoleLogManager?: ConsoleLogManager
    pendingPermissionId?: string | null
    sendPermissionResponse?: (id: string, result: string) => void
    app?: MainApp
  }
}

class WebSocketManager {
  private renderer: ContentRenderer
  private statusElement: HTMLElement
  private contentWrapper: HTMLElement
  private ws: WebSocket | null = null
  private reconnectDelay: number = 3000
  private reconnectAttempts: number = 0
  private maxReconnectAttempts: number = 10
  private connectionId: string | null = null
  private logger: any
  private app: MainApp | null = null
  private eventHandlers: EventHandlersRegistry = {}
  private _lastConsoleLog: string | null = null
  private _lastConsoleTime: number | null = null

  constructor(renderer: ContentRenderer, statusElement: HTMLElement, contentWrapper: HTMLElement) {
    this.renderer = renderer
    this.statusElement = statusElement
    this.contentWrapper = contentWrapper
    this.logger = (window as any).visualLogger

    this.logger.info('WebSocketManager initialized', {
      reconnectDelay: this.reconnectDelay,
      maxReconnectAttempts: this.maxReconnectAttempts
    })

    this.connect()
  }

  /**
   * Register an event handler for a specific event type
   */
  public on(event: string, callback: EventHandler): void {
    if (!this.eventHandlers[event]) {
      this.eventHandlers[event] = []
    }
    this.eventHandlers[event].push(callback)
  }

  /**
   * Emit an event to all registered handlers
   */
  public emit(event: string, data: any): void {
    if (this.eventHandlers[event]) {
      for (const callback of this.eventHandlers[event]) {
        try {
          callback(data)
        } catch (error) {
          this.logger.error(`Error in ${event} handler:`, {}, error)
        }
      }
    }
  }

  /**
   * Set reference to main app for loading indicator control
   */
  public setApp(app: MainApp): void {
    this.app = app
  }

  /**
   * Establishes WebSocket connection with automatic reconnection
   */
  public async connect(): Promise<void> {
    // Build WebSocket URL from current location
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsPort = window.location.port || (wsProtocol === 'wss:' ? '443' : '80')
    const wsUrl = `${wsProtocol}//${window.location.hostname}:${wsPort}/ws`

    this.logger.startTiming('websocket_connection')
    this.logger.info('Attempting WebSocket connection', { fullUrl: wsUrl, attempt: this.reconnectAttempts + 1 })

    try {
      this.ws = new WebSocket(wsUrl)
      this.connectionId = 'conn_' + Date.now()
    } catch (error) {
      this.logger.error('Failed to create WebSocket', { wsUrl }, error)
      this.handleConnectionError(error as Event)
      return
    }

    // WebSocket connection opened successfully
    this.ws.onopen = () => {
      this.logger.endTiming('websocket_connection', { connectionId: this.connectionId, reconnectAttempts: this.reconnectAttempts })
      this.reconnectAttempts = 0
      this.statusElement.textContent = '🟢 Connected to Agent'
      this.statusElement.style.color = 'var(--status-connected)'
      this.logger.info('WebSocket connected successfully', { connectionId: this.connectionId })
    }

    // Handle incoming WebSocket messages
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        this.handleMessage(data)
      } catch (error) {
        this.logger.error('Failed to parse WebSocket message', {}, error)
      }
    }

    // Handle WebSocket connection closure with exponential backoff reconnection
    this.ws.onclose = (event) => {
      this.logger.warn('WebSocket connection closed', { code: event.code, reason: event.reason })
      this.statusElement.textContent = '🔴 Disconnected. Retrying...'
      this.statusElement.style.color = 'var(--status-disconnected)'

      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++
        // Exponential backoff: 3s, 6s, 12s, 24s, capped at 30s
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000)
        setTimeout(() => this.connect(), delay)
      } else {
        this.logger.error('Max reconnection attempts reached.')
        this.statusElement.textContent = '🔴 Connection failed'
      }
    }

    // Handle WebSocket errors
    this.ws.onerror = (error) => this.handleConnectionError(error)
  }

  /**
   * Handles WebSocket connection errors
   */
  private handleConnectionError(error: Event): void {
    this.logger.error('WebSocket connection error', {}, error)
    if (this.ws) {
      this.ws.close()
    }
  }

  /**
   * Handles session initialization from server
   */
  private handleSessionInit(data: WSMessage): void {
    this.logger.info('Session initialized', { session_id: data.session_id })

    window.sessionId = data.session_id

    this.statusElement.textContent = '🟢 Connected to Agent'
    this.statusElement.style.color = 'var(--status-connected)'

    this.logger.info('Session ID stored for HTTP requests', { session_id: data.session_id })

    this.emit('session_init', data)
  }

  /**
   * Handles connection configuration from server
   */
  private handleConnectionConfig(data: WSMessage): void {
    this.logger.info('Received connection configuration', { continue_mode: data.continue_mode, clear_history: data.clear_history })

    // Clear history if requested (fresh chat)
    if (data.clear_history) {
      this.logger.info('Clearing message history (fresh chat detected)')
      try {
        localStorage.removeItem('agent-message-history')
        if (this.app?.messageHistory) {
          this.app.messageHistory = []
        }
      } catch (e) {
        this.logger.warn('Failed to clear message history:', e)
      }
    }

    // Update status element to show session type
    if (data.continue_mode) {
      this.statusElement.textContent = '🟢 Reconnected to Agent'
      this.logger.info('Continue mode enabled - will load message history')

      // Load message history since continue mode is enabled
      if (this.app?.loadMessageHistory) {
        setTimeout(() => this.app!.loadMessageHistory!(), 500)
      }
    } else {
      this.statusElement.textContent = '🟢 Connected to Agent'
      this.logger.info('Fresh start mode - skipping message history loading')

      // Clear any existing history since we're starting fresh
      if (this.app?.clearMessageHistory) {
        this.app.clearMessageHistory()
      }

      // Clear the display
      if (this.contentWrapper) {
        this.contentWrapper.innerHTML = ''
      }
    }
  }

  /**
   * Handles permission request from agent
   */
  private handlePermissionRequest(data: WSMessage): void {
    if (!data.permission_data) return

    this.logger.info('Received permission request', { tool_name: data.permission_data.tool_name })

    const permissionData = data.permission_data
    const messageId = 'permission-' + data.id

    // Create permission request UI
    const messageBlock = document.createElement('div')
    messageBlock.className = 'message-block permission-request'
    messageBlock.id = messageId

    const timestamp = new Date().toLocaleTimeString()

    // Create header
    const headerDiv = document.createElement('div')
    headerDiv.className = 'permission-header'
    headerDiv.innerHTML = `
      <span class="permission-timestamp">${timestamp} - Permission Request</span>
      <span class="permission-tool">🔒 ${permissionData.tool_name}</span>
    `

    // Create permission content
    const contentDiv = document.createElement('div')
    contentDiv.className = 'permission-content'

    // Tool information
    const toolEmojis: Record<string, string> = {
      "editfile": "✏️", "readfile": "📖", "writefile": "📝",
      "insertline": "➕", "deletelines": "🗑️", "bash": "🔧",
      "python": "🐍", "searchweb": "🌐", "showuser": "👁️",
      "memory": "🧠", "grepsearch": "🔍"
    }
    const toolEmoji = toolEmojis[permissionData.tool_name] || "⚙️"

    let contentHTML = `
      <div class="permission-info">
        <h3>${toolEmoji} ${permissionData.tool_name} permission required</h3>
        <p><strong>Action:</strong> ${permissionData.action_description}</p>
    `

    if (permissionData.file_path) {
      contentHTML += `<p><strong>File:</strong> ${permissionData.file_path}</p>`
    }

    if (permissionData.args_display) {
      contentHTML += `<p><strong>Command:</strong> ${permissionData.args_display}</p>`
    }

    contentHTML += `</div>`

    // Create action buttons
    contentHTML += `
      <div class="permission-actions">
        <button class="permission-btn allow" onclick="window.sendPermissionResponse('${data.id}', 'allow')">
          ✅ Allow
        </button>
        <button class="permission-btn deny" onclick="window.sendPermissionResponse('${data.id}', 'deny')">
          ❌ Deny
        </button>
        <button class="permission-btn always-allow" onclick="window.sendPermissionResponse('${data.id}', 'always_allow')">
          ✅ Always Allow
        </button>
        <button class="permission-btn always-deny" onclick="window.sendPermissionResponse('${data.id}', 'always_deny')">
          ❌ Always Deny
        </button>
      </div>
    `

    contentDiv.innerHTML = contentHTML

    messageBlock.appendChild(headerDiv)
    messageBlock.appendChild(contentDiv)

    this.contentWrapper.appendChild(messageBlock)
    this.scrollToBottom()

    // Store the permission ID for response handling
    window.pendingPermissionId = data.id
  }

  /**
   * Handles user message from command line
   */
  private handleUserMessage(data: WSMessage): void {
    this.logger.info('Received user message from command line', { message: data.message })

    // Use the app's displayUserMessage function if available
    if (this.app?.displayUserMessage) {
      this.app.displayUserMessage(data.message!)
    } else {
      // Fallback: create user message directly
      this.createUserMessageFallback(data.message!)
    }
  }

  /**
   * Handles assistant message from agent
   */
  private handleAssistantMessage(data: WSMessage): void {
    this.logger.debug('Received assistant message', { message: data.message })

    // Use the app's displayAssistantMessage function if available
    if (this.app?.displayAssistantMessage) {
      this.app.displayAssistantMessage(data.message!)
    } else {
      // Fallback: create assistant message directly
      this.createAssistantMessageFallback(data.message!)
    }
  }

  /**
   * Handles console log message from backend
   */
  private handleConsoleLog(data: WSMessage): void {
    this.logger.debug('Received console log message', { line: data.line, level: data.level })

    // Forward to console log manager if available
    if (window.consoleLogManager) {
      window.consoleLogManager.addLogLine(data.line!, data.level || 'info')
    } else {
      // Fallback: log to browser console with rate limiting
      if (!this._lastConsoleLog || this._lastConsoleLog !== data.line || Date.now() - this._lastConsoleTime! > 1000) {
        console.log(`[CLI-Agent Console] ${data.line}`)
        this._lastConsoleLog = data.line!
        this._lastConsoleTime = Date.now()
      }
    }
  }

  /**
   * Handles screenshot event from backend
   */
  private handleScreenshotEvent(data: WSMessage): void {
    this.logger.debug('Received screenshot event', { action: data.action, status: data.status })

    // Forward to main app for screenshot button state management
    if (this.app) {
      if (data.action === 'captured' || data.status === 'ready') {
        this.app.setScreenshotReady?.()
      } else if (data.action === 'failed' || data.status === 'error') {
        this.app.resetScreenshotState?.()
      }
    } else {
      this.logger.warn('No app reference available for screenshot state management')
    }
  }

  /**
   * Fallback method to display user message
   */
  private createUserMessageFallback(message: string): void {
    const contentWrapper = document.getElementById('content-wrapper')
    if (!contentWrapper) return

    const currentTimestamp = Date.now()

    const messageBlock = document.createElement('div')
    messageBlock.className = 'message-block user-message'

    const timestamp = document.createElement('div')
    timestamp.className = 'agent-msg-head'
    timestamp.textContent = new Date(currentTimestamp).toLocaleTimeString()

    const content = document.createElement('div')
    content.className = 'message-content'
    content.textContent = message

    messageBlock.appendChild(timestamp)
    messageBlock.appendChild(content)

    contentWrapper.appendChild(messageBlock)

    this.scrollToBottom()

    this.logger.debug('User message displayed via fallback method')
  }

  /**
   * Fallback method to display assistant message
   */
  private createAssistantMessageFallback(message: string): void {
    const contentWrapper = document.getElementById('content-wrapper')
    if (!contentWrapper) return

    const currentTimestamp = Date.now()

    const messageBlock = document.createElement('div')
    messageBlock.className = 'message-block assistant-message'

    const timestamp = document.createElement('div')
    timestamp.className = 'agent-msg-head'
    timestamp.textContent = new Date(currentTimestamp).toLocaleTimeString()

    const content = document.createElement('div')
    content.className = 'message-content'

    const pre = document.createElement('pre')
    pre.style.cssText = 'white-space: pre-wrap; font-family: monospace; background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 0;'
    pre.textContent = message
    content.appendChild(pre)

    messageBlock.appendChild(timestamp)
    messageBlock.appendChild(content)

    contentWrapper.appendChild(messageBlock)

    this.scrollToBottom()

    this.logger.debug('Assistant message displayed via fallback method')
  }

  /**
   * Main WebSocket message handler - routes messages to appropriate handlers
   */
  public async handleMessage(data: WSMessage): Promise<void> {
    this.logger.debug('Received WebSocket message', { type: data.type, action: data.action, id: data.id, model: data.model })

    if (data.type === 'session_init') {
      this.handleSessionInit(data)
      return
    }

    if (data.type === 'chat_replaced') {
      this.emit('chat_replaced', data)
      return
    }

    if (data.type === 'chat_cleared') {
      this.emit('chat_cleared', data)
      return
    }

    if (data.type === 'chat_truncated') {
      this.emit('chat_truncated', data)
      return
    }

    if (data.type === 'message_added') {
      this.emit('message_added', data)
      return
    }

    if (data.type === 'message_deleted') {
      this.emit('message_deleted', data)
      return
    }

    if (data.type === 'state_changed') {
      this.emit('state_changed', data)
      return
    }

    if (data.type === 'message_edited') {
      this.handleMessageEdited(data)
      return
    }

    if (data.type === 'message_start') {
      this.emit('message_start', data)
      return
    }

    if (data.type === 'message_chunk') {
      this.emit('message_chunk', data)
      return
    }

    if (data.type === 'message_end') {
      this.emit('message_end', data)
      return
    }

    if (data.type === 'token_stream') {
      this.emit('token_stream', data)
      return
    }

    if (data.type === 'tool_update') {
      this.emit('tool_update', data)
      return
    }

    if (data.type === 'prompts_regenerated') {
      this.emit('prompts_regenerated', data)
      return
    }

    if (data.type === 'config' && data.action === 'connection_init') {
      this.handleConnectionConfig(data)
      return
    }

    if (data.type === 'permission_request') {
      this.handlePermissionRequest(data)
      return
    }

    if (data.type === 'user_message' && data.action === 'display') {
      this.handleUserMessage(data)
      return
    }

    if (data.type === 'assistant_message' && data.action === 'display') {
      this.handleAssistantMessage(data)
      return
    }

    if (data.type === 'console_log') {
      this.handleConsoleLog(data)
      return
    }

    if (data.type === 'screenshot') {
      this.handleScreenshotEvent(data)
      return
    }

    if (data.action === 'tool_start') {
      this.handleLiveToolStart(data)
      return
    }

    if (data.action === 'tool_complete') {
      this.handleLiveToolComplete(data)
      return
    }

    if (data.action === 'loading_tool_start') {
      this.handleLoadingToolStart(data)
      return
    }

    if (data.action === 'loading_tool_complete') {
      this.handleLoadingToolComplete(data)
      return
    }

    if (data.action === 'audio_tts' && data.type === 'audio') {
      this.handleWebSocketAudio(data)
      return
    }

    // Legacy message rendering logic
    const shouldHideLoadingIndicator =
      data.type === 'html' ||
      data.type === 'markdown' ||
      data.action === 'error' ||
      (data.content && (
        data.content.includes('<visualresponse>') ||
        data.content.includes('<response>')
      ))

    if (shouldHideLoadingIndicator) {
      this.app?.hideLoadingIndicator?.()
      this.app?.hideInterruptButton?.()
    }

    this.cleanupLiveToolsContainer()
    this.transferLoadingToolsToMessage(data.id, data.tool_info)

    setTimeout(() => {
      this.performMaintenanceCleanup()
    }, 500)

    // If ChatStateManager exists, skip legacy rendering
    if (window.chatState?.addMessage) {
      this.logger.debug('ChatStateManager present, skipping legacy message renderer')
      return
    }

    await this.renderLegacyMessage(data)
  }

  /**
   * Legacy message rendering (fallback when ChatStateManager not present)
   */
  private async renderLegacyMessage(data: WSMessage): Promise<void> {
    const messageId = 'message-' + data.id
    const existingBlock = document.getElementById(messageId)

    if (data.action === 'delete') {
      existingBlock?.remove()
      this.logger.info(`Deleted message block: ${messageId}`)
      return
    }

    if (data.action === 'clear') {
      if (this.app?.isRestoring) {
        this.logger.info('Skipping clear action during restoration')
        return
      }
      this.contentWrapper.innerHTML = ''
      this.app?.clearMessageHistory?.()
      this.logger.info('Cleared all visual responses.')
      return
    }

    this.logger.startTiming(`render_message_${messageId}`)

    let messageBlock = existingBlock
    if (!messageBlock) {
      messageBlock = document.createElement('div')
      messageBlock.className = 'message-block agent-message'
      messageBlock.id = messageId
      this.contentWrapper.appendChild(messageBlock)
    }

    const timestamp = new Date().toLocaleTimeString()

    const timestampDiv = document.createElement('div')
    timestampDiv.className = 'agent-msg-head'

    let timestampContent = `<span class="timestamp-text">${timestamp} - ID: ${data.id}`

    if (data.model) {
      timestampContent += ` <span class="model-key" title="Model used for this response">[${data.model}]</span>`
    }

    timestampContent += `</span>`
    timestampDiv.innerHTML = timestampContent
    messageBlock.appendChild(timestampDiv)

    const messageContent = document.createElement('div')
    messageContent.className = 'message-content'
    messageBlock.appendChild(messageContent)

    await this.renderer.renderContent(messageContent, data, messageId, timestampDiv)

    // Save to history if not restoring
    if (this.app?.saveMessageToHistory && !this.app.isRestoring) {
      const transientTypes = ['permission_request', 'config', 'tts_chunk']
      const isAudioMessage = data.action === 'audio_tts' && data.type === 'audio'

      if (!transientTypes.includes(data.type) && !isAudioMessage) {
        this.app.saveMessageToHistory('agent', {
          originalId: data.id,
          timestamp: Date.now(),
          messageData: data
        })
      }
    }

    this.scrollToBottom()
    this.logger.endTiming(`render_message_${messageId}`, { success: true, contentType: data.type })
  }

  /**
   * Unified autoscroll function
   */
  public scrollToBottom(): void {
    if (!this.contentWrapper) return

    const forceScrollToEnd = () => {
      this.contentWrapper.scrollTop = this.contentWrapper.scrollHeight + 100
      this.contentWrapper.scrollTo(0, this.contentWrapper.scrollHeight + 100)
      this.contentWrapper.scrollTop = this.contentWrapper.scrollHeight

      const lastChild = this.contentWrapper.lastElementChild
      if (lastChild) {
        try {
          lastChild.scrollIntoView({ behavior: 'instant', block: 'end' } as ScrollIntoViewOptions)
        } catch (e) {
          (lastChild as any).scrollIntoView(false)
        }
      }
    }

    forceScrollToEnd()

    requestAnimationFrame(() => {
      forceScrollToEnd()
      setTimeout(forceScrollToEnd, 10)
      setTimeout(forceScrollToEnd, 50)
    })
  }

  /**
   * Sends a message through the WebSocket connection
   */
  public sendMessage(message: string): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(message)
      this.logger.info('Message sent to agent', { messageLength: message.length })
      return true
    }
    this.logger.warn('Cannot send message - WebSocket not connected.')
    return false
  }

  /**
   * Handles message edit events from backend
   */
  private handleMessageEdited(data: WSMessage): void {
    const message = (data as any).message
    if (!message) return

    const messageBlock = document.querySelector(`[data-message-index="${message.index}"]`)
    if (!messageBlock) {
      console.warn('Message block not found for index:', message.index)
      return
    }

    const contentDiv = messageBlock.querySelector('.message-content')
    if (contentDiv) {
      contentDiv.textContent = message.content

      messageBlock.classList.add('message-edited')
      setTimeout(() => messageBlock.classList.remove('message-edited'), 2000)

      this.logger.info('Message edited remotely', { index: message.index })
    }
  }

  /**
   * Checks if WebSocket connection is currently open
   */
  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  /**
   * Sends permission response to backend
   */
  public sendPermissionResponse(permissionId: string, result: string, action: any = {}): boolean {
    const response = {
      type: 'permission_response',
      permission_id: permissionId,
      result: result,
      action: action,
      timestamp: Date.now()
    }

    if (this.sendMessage(JSON.stringify(response))) {
      this.logger.info('Sent permission response', { permissionId, result })

      const permissionElement = document.getElementById('permission-' + permissionId)
      permissionElement?.remove()

      window.pendingPermissionId = null

      return true
    }
    return false
  }

  /**
   * Sends interrupt signal to agent
   */
  public sendInterruptSignal(): boolean {
    const interruptMessage = {
      type: 'interrupt_signal',
      timestamp: Date.now(),
      source: 'web_ui'
    }

    if (this.sendMessage(JSON.stringify(interruptMessage))) {
      this.logger.info('Sent interrupt signal to agent')
      return true
    } else {
      this.logger.error('Failed to send interrupt signal - WebSocket not connected')
      return false
    }
  }

  /**
   * Handles live tool start event (legacy)
   */
  private handleLiveToolStart(data: WSMessage): void {
    this.logger.info('Live tool started', { tool_name: data.tool_name })

    let liveToolsContainer = document.getElementById('live-tools-container')
    if (!liveToolsContainer) {
      const loadingIndicator = document.getElementById('agent-loading-indicator')
      if (loadingIndicator) {
        liveToolsContainer = document.createElement('div')
        liveToolsContainer.id = 'live-tools-container'
        liveToolsContainer.className = 'live-tools-container'
        loadingIndicator.parentNode!.insertBefore(liveToolsContainer, loadingIndicator.nextSibling)
      }
    }

    if (liveToolsContainer) {
      const toolItem = document.createElement('div')
      toolItem.id = `live-tool-${data.tool_name}`
      toolItem.className = 'live-tool-item running'

      const toolEmojis: Record<string, string> = {
        "editfile": "✏️", "readfile": "📖", "writefile": "📝",
        "insertline": "➕", "deletelines": "🗑️", "bash": "🔧",
        "python": "🐍", "searchweb": "🌐", "showuser": "👁️",
        "memory": "🧠", "grepsearch": "🔍", "visualresponse": "🎨"
      }
      const toolEmoji = toolEmojis[data.tool_name!] || "⚙️"

      toolItem.innerHTML = `
        <div class="live-tool-header">
          <span class="live-tool-emoji">${toolEmoji}</span>
          <span class="live-tool-name">${data.tool_name}</span>
          <span class="live-tool-status running">🔄 Running...</span>
        </div>
        <div class="live-tool-args">${this.escapeHtml(data.tool_args || 'No arguments')}</div>
      `

      liveToolsContainer.appendChild(toolItem)
      this.scrollToBottom()
    }
  }

  /**
   * Handles live tool completion event (legacy)
   */
  private handleLiveToolComplete(data: WSMessage): void {
    this.logger.info('Live tool completed', { tool_name: data.tool_name, status: data.tool_status })

    const toolItem = document.getElementById(`live-tool-${data.tool_name}`)
    if (toolItem) {
      const statusSpan = toolItem.querySelector('.live-tool-status')
      if (statusSpan) {
        if (data.tool_status === 'success') {
          statusSpan.innerHTML = `✅ Completed (${data.tool_duration_ms}ms)`
          statusSpan.className = 'live-tool-status success'
        } else {
          statusSpan.innerHTML = `❌ Failed (${data.tool_duration_ms}ms)`
          statusSpan.className = 'live-tool-status error'
        }
      }

      toolItem.classList.remove('running')
      toolItem.classList.add('completed')

      setTimeout(() => {
        if (toolItem.parentNode) {
          toolItem.style.opacity = '0.5'
          toolItem.style.transform = 'translateX(20px)'
        }
      }, 2000)

      setTimeout(() => {
        if (toolItem.parentNode) {
          toolItem.remove()

          const container = document.getElementById('live-tools-container')
          if (container && container.children.length === 0) {
            container.remove()
            this.logger.info('Removed empty live tools container')
          }
        }
      }, 5000)
    }
  }

  /**
   * Cleans up live tools container
   */
  private cleanupLiveToolsContainer(): void {
    const container = document.getElementById('live-tools-container')
    if (container) {
      container.style.transition = 'opacity 0.5s ease, transform 0.5s ease'
      container.style.opacity = '0'
      container.style.transform = 'translateY(-10px)'

      setTimeout(() => {
        container.remove()
        this.logger.info('Cleaned up live tools container on message arrival')
      }, 500)
    }
  }

  /**
   * Escapes HTML special characters
   */
  private escapeHtml(text: string): string {
    if (!text) return ''
    const div = document.createElement('div')
    div.textContent = text
    return div.innerHTML
  }

  /**
   * Handles loading tool start event
   */
  private handleLoadingToolStart(data: WSMessage): void {
    this.logger.info('Loading tool started', { tool_name: data.tool_name })

    let loadingToolsHeader = document.getElementById('loading-tools-header')
    if (!loadingToolsHeader) {
      const loadingIndicator = document.getElementById('agent-loading-indicator')
      if (loadingIndicator) {
        loadingToolsHeader = document.createElement('div')
        loadingToolsHeader.id = 'loading-tools-header'
        loadingToolsHeader.className = 'loading-tools-header'
        loadingIndicator.parentNode!.insertBefore(loadingToolsHeader, loadingIndicator.nextSibling)
      }
    }

    if (!loadingToolsHeader) return

    const toolItem = document.createElement('div')
    toolItem.className = 'loading-tool-item running'
    toolItem.id = `loading-tool-${data.tool_name}`

    const startTime = Date.now()
    toolItem.dataset.startTime = String(startTime)

    const modelDisplay = data.model ? `<span class="loading-tool-model">[${this.escapeHtml(data.model)}]</span>` : ''

    toolItem.innerHTML = `
      <div class="loading-tool-main">
        <span class="loading-tool-name">${this.escapeHtml(data.tool_name!)}</span>
        ${modelDisplay}
        <span class="loading-tool-status running">
          <span class="loading-tool-spinner">⏳</span>
          <span class="loading-tool-stopwatch">0.0s</span>
        </span>
      </div>
      <div class="loading-tool-details collapsed">
        <div class="loading-tool-args">${this.escapeHtml(data.tool_args || 'No arguments')}</div>
        <div class="loading-tool-output">Output will appear here...</div>
      </div>
    `

    const mainDiv = toolItem.querySelector('.loading-tool-main') as HTMLElement
    mainDiv.addEventListener('click', () => {
      const details = toolItem.querySelector('.loading-tool-details')!
      const isCollapsed = details.classList.contains('collapsed')

      if (isCollapsed) {
        details.classList.remove('collapsed')
        details.classList.add('expanded')
      } else {
        details.classList.remove('expanded')
        details.classList.add('collapsed')
      }
    })

    loadingToolsHeader.appendChild(toolItem)
    this.scrollToBottom()

    const stopwatchSpan = toolItem.querySelector('.loading-tool-stopwatch')
    const stopwatchTimer = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000
      if (stopwatchSpan) {
        stopwatchSpan.textContent = elapsed.toFixed(1) + 's'
      }
    }, 100)

    toolItem.dataset.stopwatchTimer = String(stopwatchTimer)
  }

  /**
   * Handles WebSocket TTS audio streaming
   */
  private handleWebSocketAudio(data: WSMessage): void {
    this.logger.info('Received WebSocket TTS audio', {
      text: data.text?.substring(0, 50) + '...',
      format: data.audio_format,
      duration: data.duration,
      language: data.language
    })

    if (window.audioManager?.playWebSocketAudio && data.audio_data) {
      window.audioManager.playWebSocketAudio({
        audio_data: data.audio_data,
        text: data.text,
        audio_format: data.audio_format,
        duration: data.duration,
        language: data.language
      })
    } else {
      this.logger.warn('AudioManager not available for WebSocket audio playback or missing audio_data')
    }
  }

  /**
   * Handles loading tool completion event
   */
  private handleLoadingToolComplete(data: WSMessage): void {
    console.log('🔧 Received loading tool complete:', data)
    this.logger.info('Loading tool completed', { tool_name: data.tool_name, status: data.tool_status })

    const toolItem = document.querySelector(`[id="loading-tool-${data.tool_name}"].running`)

    if (!toolItem) {
      this.logger.warn('No loading tool item found for:', data.tool_name)
      return
    }

    const timerId = toolItem.getAttribute('data-stopwatch-timer')
    if (timerId) {
      clearInterval(parseInt(timerId))
    }

    const statusSpan = toolItem.querySelector('.loading-tool-status')
    if (statusSpan) {
      const duration = data.tool_duration_ms || 0
      const frozenTime = (duration / 1000).toFixed(1) + 's'

      if (data.tool_status === 'success') {
        statusSpan.innerHTML = `
          <span class="loading-tool-indicator">✅</span>
          <span class="loading-tool-stopwatch frozen">${frozenTime}</span>
        `
        statusSpan.className = 'loading-tool-status success'
      } else {
        statusSpan.innerHTML = `
          <span class="loading-tool-indicator">❌</span>
          <span class="loading-tool-stopwatch frozen">${frozenTime}</span>
        `
        statusSpan.className = 'loading-tool-status error'
      }
    }

    const outputDiv = toolItem.querySelector('.loading-tool-output')
    if (outputDiv) {
      if (data.tool_result) {
        outputDiv.innerHTML = `<pre>${this.escapeHtml(data.tool_result)}</pre>`
      } else if (data.tool_error) {
        outputDiv.innerHTML = `<div class="error">Error: ${this.escapeHtml(data.tool_error)}</div>`
      } else {
        outputDiv.textContent = 'Completed successfully'
      }
    }

    toolItem.classList.remove('running')
    toolItem.classList.add(data.tool_status === 'success' ? 'completed' : 'failed')

    const spinner = toolItem.querySelector('.loading-tool-spinner')
    spinner?.remove()

    toolItem.removeAttribute('data-stopwatch-timer')

    this.scrollToBottom()
  }

  /**
   * Transfers loading tools to message block
   */
  private transferLoadingToolsToMessage(messageId: string | undefined, toolInfo: any = null): void {
    const loadingToolsHeader = document.getElementById('loading-tools-header')
    if (!loadingToolsHeader || loadingToolsHeader.children.length === 0) {
      loadingToolsHeader?.remove()
      return
    }

    setTimeout(() => {
      const messageBlock = document.getElementById(`message-${messageId}`)
      if (!messageBlock) {
        this.logger.warn('No message block found for transfer:', messageId)
        return
      }

      let toolsHeader = messageBlock.querySelector('.loading-tools-header') as HTMLElement
      if (!toolsHeader) {
        toolsHeader = document.createElement('div')
        toolsHeader.className = 'loading-tools-header'

        const contentArea = messageBlock.querySelector('.message-content')
        if (contentArea) {
          contentArea.parentNode!.insertBefore(toolsHeader, contentArea)
        }
      }

      const toolsArray = Array.from(loadingToolsHeader.children)

      toolsArray.forEach(loadingTool => {
        const clonedTool = loadingTool.cloneNode(true) as HTMLElement

        const mainDiv = clonedTool.querySelector('.loading-tool-main') as HTMLElement
        if (mainDiv) {
          mainDiv.addEventListener('click', () => {
            const details = clonedTool.querySelector('.loading-tool-details') as HTMLElement
            const isVisible = details.style.display !== 'none'
            details.style.display = isVisible ? 'none' : 'block'
          })
        }

        const originalStopwatchTimer = (loadingTool as HTMLElement).dataset.stopwatchTimer
        if (originalStopwatchTimer && loadingTool.classList.contains('running')) {
          const stopwatchSpan = clonedTool.querySelector('.loading-tool-stopwatch')
          const startTime = parseInt((loadingTool as HTMLElement).dataset.startTime!)

          if (stopwatchSpan && startTime) {
            const stopwatchTimer = setInterval(() => {
              const elapsed = (Date.now() - startTime) / 1000
              if (stopwatchSpan) {
                stopwatchSpan.textContent = elapsed.toFixed(1) + 's'
              }
            }, 100)
            clonedTool.dataset.stopwatchTimer = String(stopwatchTimer)
            clonedTool.dataset.startTime = String(startTime)
          }
        }

        toolsHeader.appendChild(clonedTool)
      })

      loadingToolsHeader.remove()
      this.logger.info('Transferred loading tools to visualresponse header')
    }, 100)
  }

  /**
   * Comprehensive cleanup function
   */
  private performMaintenanceCleanup(): void {
    const orphanedHeaders = document.querySelectorAll('.loading-tools-header')
    orphanedHeaders.forEach(header => {
      if (header.children.length === 0) {
        this.logger.info('Removing empty loading tools header')
        header.remove()
      }
    })

    const stuckTools = document.querySelectorAll('.loading-tool-item.running')
    stuckTools.forEach(tool => {
      const htmlTool = tool as HTMLElement
      const startTime = parseInt(htmlTool.dataset.startTime!)
      const currentTime = Date.now()
      const elapsed = currentTime - startTime

      if (elapsed > 30000) {
        this.logger.warn('Found stuck tool, marking as failed:', htmlTool.id)

        const timerId = htmlTool.dataset.stopwatchTimer
        if (timerId) {
          clearInterval(parseInt(timerId))
        }

        const statusSpan = tool.querySelector('.loading-tool-status')
        if (statusSpan) {
          const frozenTime = (elapsed / 1000).toFixed(1) + 's'
          statusSpan.innerHTML = `
            <span class="loading-tool-indicator">⚠️</span>
            <span class="loading-tool-stopwatch frozen">${frozenTime}</span>
          `
          statusSpan.className = 'loading-tool-status error'
        }

        tool.classList.remove('running')
        tool.classList.add('failed')
        htmlTool.removeAttribute('data-stopwatch-timer')
      }
    })
  }
}

window.WebSocketManager = WebSocketManager

export default WebSocketManager
export { WebSocketManager }
