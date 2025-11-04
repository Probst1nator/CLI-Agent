/**
 * Screen Capture Manager
 *
 * Handles browser-based screen/window/tab capture using the Screen Capture API
 * with support for both Chromium and Firefox browsers.
 */

declare global {
  interface Window {
    ScreenCaptureManager?: typeof ScreenCaptureManager
  }
}

interface CaptureOptions {
  audio?: boolean
  preferCurrentTab?: boolean
}

class ScreenCaptureManager {
  private captureStream: MediaStream | null = null
  private videoElement: HTMLVideoElement
  private isCapturing: boolean = false

  constructor() {
    this.videoElement = this.initializeVideoElement()
  }

  /**
   * Initialize hidden video element for capturing frames
   */
  private initializeVideoElement(): HTMLVideoElement {
    const video = document.createElement('video')
    video.style.display = 'none'
    video.autoplay = true
    video.muted = true
    document.body.appendChild(video)
    return video
  }

  /**
   * Check if Screen Capture API is supported
   */
  public isSupported(): boolean {
    return !!(navigator.mediaDevices?.getDisplayMedia || navigator.mediaDevices?.getUserMedia)
  }

  /**
   * Request screen capture from user
   */
  public async requestScreenCapture(options: CaptureOptions = {}): Promise<MediaStream> {
    if (!this.isSupported()) {
      throw new Error('Screen Capture API is not supported in this browser')
    }

    const captureOptions: DisplayMediaStreamOptions = {
      video: {
        cursor: 'always' as any,
        displaySurface: options.preferCurrentTab ? 'browser' as any : 'monitor' as any,
      } as any,
      audio: options.audio || false,
    }

    try {
      console.log('🎥 Requesting screen capture with options:', captureOptions)

      this.captureStream = await navigator.mediaDevices.getDisplayMedia(captureOptions)

      console.log('✅ Screen capture stream acquired')
      return this.captureStream

    } catch (error) {
      console.error('❌ Screen capture failed:', error)

      if ((error as DOMException).name === 'NotAllowedError') {
        throw new Error('Screen capture permission denied by user')
      } else if ((error as DOMException).name === 'NotFoundError') {
        throw new Error('No screen capture source available')
      } else if ((error as DOMException).name === 'NotSupportedError') {
        throw new Error('Screen capture not supported in this browser')
      } else {
        throw new Error(`Screen capture error: ${(error as Error).message}`)
      }
    }
  }

  /**
   * Capture a single screenshot from the stream
   */
  public async captureScreenshot(stream: MediaStream): Promise<string> {
    if (!stream || !stream.active) {
      throw new Error('No active capture stream')
    }

    return new Promise((resolve, reject) => {
      try {
        this.videoElement.srcObject = stream

        this.videoElement.onloadedmetadata = () => {
          this.videoElement.play().then(() => {
            setTimeout(() => {
              try {
                const canvas = document.createElement('canvas')
                canvas.width = this.videoElement.videoWidth
                canvas.height = this.videoElement.videoHeight

                const ctx = canvas.getContext('2d')
                if (!ctx) {
                  reject(new Error('Failed to get 2D context'))
                  return
                }

                ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height)

                const screenshot = canvas.toDataURL('image/png')

                console.log(`📸 Screenshot captured: ${canvas.width}x${canvas.height}`)
                resolve(screenshot)

              } catch (error) {
                reject(new Error(`Failed to capture frame: ${(error as Error).message}`))
              }
            }, 100)
          }).catch(reject)
        }

        this.videoElement.onerror = (error) => {
          reject(new Error(`Video playback error: ${error}`))
        }

      } catch (error) {
        reject(new Error(`Screenshot capture error: ${(error as Error).message}`))
      }
    })
  }

  /**
   * Stop the capture stream and cleanup
   */
  public stopCapture(): void {
    if (this.captureStream) {
      this.captureStream.getTracks().forEach(track => track.stop())
      this.captureStream = null
      console.log('🛑 Screen capture stream stopped')
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null
    }

    this.isCapturing = false
  }

  /**
   * Capture screen and return screenshot
   */
  public async capture(options: CaptureOptions = {}): Promise<string> {
    if (this.isCapturing) {
      throw new Error('Capture already in progress')
    }

    this.isCapturing = true

    try {
      const stream = await this.requestScreenCapture(options)
      const screenshot = await this.captureScreenshot(stream)
      this.stopCapture()
      return screenshot

    } catch (error) {
      this.stopCapture()
      throw error
    }
  }

  /**
   * Start continuous screen capture (keeps stream alive)
   */
  public async startContinuousCapture(options: CaptureOptions = {}): Promise<MediaStream> {
    if (this.captureStream?.active) {
      console.log('Continuous capture already active')
      return this.captureStream
    }

    try {
      this.captureStream = await this.requestScreenCapture(options)
      this.videoElement.srcObject = this.captureStream

      await new Promise<void>((resolve, reject) => {
        this.videoElement.onloadedmetadata = () => {
          this.videoElement.play().then(() => resolve()).catch(reject)
        }
        this.videoElement.onerror = reject
      })

      console.log('🟢 Continuous screen capture started')
      this.isCapturing = true
      return this.captureStream

    } catch (error) {
      this.stopCapture()
      throw error
    }
  }

  /**
   * Capture a screenshot from existing continuous stream
   */
  public async captureFromStream(): Promise<string> {
    if (!this.captureStream?.active) {
      throw new Error('No active continuous capture stream. Call startContinuousCapture() first.')
    }

    try {
      const canvas = document.createElement('canvas')
      canvas.width = this.videoElement.videoWidth
      canvas.height = this.videoElement.videoHeight

      const ctx = canvas.getContext('2d')
      if (!ctx) {
        throw new Error('Failed to get 2D context')
      }

      ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height)

      const screenshot = canvas.toDataURL('image/png')
      console.log(`📸 Screenshot captured from stream: ${canvas.width}x${canvas.height}`)
      return screenshot

    } catch (error) {
      throw new Error(`Failed to capture from stream: ${(error as Error).message}`)
    }
  }

  /**
   * Check if browser is Firefox
   */
  public isFirefox(): boolean {
    return navigator.userAgent.toLowerCase().indexOf('firefox') > -1
  }

  /**
   * Check if browser is Chrome/Chromium
   */
  public isChromium(): boolean {
    return /Chrome|Chromium|Edg/.test(navigator.userAgent)
  }

  /**
   * Get browser-specific capture instructions
   */
  public getCaptureInstructions(): string {
    if (this.isFirefox()) {
      return 'Firefox: Select "Entire Screen", "Window", or "Application Window" from the dialog.'
    } else if (this.isChromium()) {
      return 'Chrome: Select "Entire Screen", "Window", or "Chrome Tab" from the dialog.'
    } else {
      return 'Select the screen, window, or tab you want to capture.'
    }
  }
}

export default ScreenCaptureManager
export { ScreenCaptureManager }
