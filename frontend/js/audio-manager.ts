/**
 * AudioManager - Browser audio recording and TTS playback
 *
 * Handles microphone recording via MediaRecorder API, uploading to backend for transcription,
 * and playing TTS audio responses with queue management.
 */

declare global {
  interface Window {
    AudioManager?: typeof AudioManager
    // Note: audioManager instance is typed in websocket.ts with specific method signature
    AudioContext: typeof AudioContext
    webkitAudioContext: typeof AudioContext
  }
}

interface AudioQueueItem {
  url: string
  metadata: {
    volume?: number
    text?: string
    format?: string
    duration?: number
    language?: string
  }
}

interface WebSocketAudioData {
  audio_data: string
  text?: string
  audio_format?: string
  duration?: number
  language?: string
}

class AudioManager {
  private recorder: MediaRecorder | null = null
  private audioContext: AudioContext | null = null
  private stream: MediaStream | null = null
  private chunks: Blob[] = []
  private playingAudio: HTMLAudioElement | null = null
  private isRecording: boolean = false
  private audioQueue: AudioQueueItem[] = []
  private isPlayingQueue: boolean = false
  private hasUserInteraction: boolean = false
  private isTTSActive: boolean = false
  private volume: number = 1.0

  // VAD (Voice Activity Detection) settings
  private vadEnabled: boolean = false
  private vadThreshold: number = 0.02
  private vadSilenceDuration: number = 2.0
  private vadMaxDuration: number = 20.0
  private vadRecording: boolean = false
  private vadSilentChunks: number = 0
  private vadRecordedChunks: Float32Array[] = []
  private vadStartTime: number = 0
  private vadProcessor: ScriptProcessorNode | null = null

  constructor() {
    console.log('AudioManager initialized')

    // Set flag on first user interaction to allow autoplay
    const enableAutoplay = () => {
      this.hasUserInteraction = true
      console.log('User interaction detected - audio autoplay enabled')
      document.removeEventListener('click', enableAutoplay)
      document.removeEventListener('keydown', enableAutoplay)
    }
    document.addEventListener('click', enableAutoplay, { once: true })
    document.addEventListener('keydown', enableAutoplay, { once: true })
  }

  /**
   * Start recording from microphone using MediaRecorder API
   */
  public async startRecording(): Promise<void> {
    try {
      if (this.isRecording) {
        console.warn('Already recording')
        return
      }

      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      this.recorder = new MediaRecorder(this.stream)
      this.chunks = []

      this.recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          this.chunks.push(e.data)
        }
      }

      this.recorder.start()
      this.isRecording = true

      console.log('Recording started')
      this.showStatus('Recording...', 'info')

    } catch (error) {
      console.error('Failed to start recording:', error)

      if ((error as DOMException).name === 'NotAllowedError' || (error as DOMException).name === 'PermissionDeniedError') {
        this.showError('Microphone access denied. Please allow microphone access in your browser settings.')
      } else if ((error as DOMException).name === 'NotFoundError') {
        this.showError('No microphone found. Please connect a microphone.')
      } else {
        this.showError('Failed to start recording: ' + (error as Error).message)
      }

      throw error
    }
  }

  /**
   * Stop recording and upload for transcription
   */
  public async stopRecording(): Promise<string> {
    try {
      if (!this.isRecording || !this.recorder) {
        console.warn('Not recording')
        return ''
      }

      return new Promise((resolve, reject) => {
        this.recorder!.onstop = async () => {
          this.isRecording = false

          if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop())
            this.stream = null
          }

          try {
            const audioBlob = new Blob(this.chunks, { type: 'audio/webm' })
            console.log('Recording stopped, blob size:', audioBlob.size)

            this.showStatus('Transcribing...', 'info')
            const text = await this.uploadAudio(audioBlob)

            resolve(text)
          } catch (error) {
            reject(error)
          }
        }

        this.recorder!.stop()
      })

    } catch (error) {
      console.error('Failed to stop recording:', error)
      this.showError('Recording upload failed: ' + (error as Error).message)
      throw error
    }
  }

  /**
   * Start recording with automatic Voice Activity Detection
   */
  public async startRecordingWithVAD(): Promise<string> {
    try {
      if (this.isRecording) {
        console.warn('Already recording')
        return ''
      }

      console.log(`Starting VAD recording (threshold: ${this.vadThreshold}, silence: ${this.vadSilenceDuration}s, max: ${this.vadMaxDuration}s)`)

      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      })

      const source = this.audioContext.createMediaStreamSource(this.stream)
      this.vadProcessor = this.audioContext.createScriptProcessor(1600, 1, 1)

      this.vadRecording = false
      this.vadSilentChunks = 0
      this.vadRecordedChunks = []
      this.vadStartTime = Date.now()
      this.isRecording = true

      const numSilentChunksToStop = Math.floor(this.vadSilenceDuration * 10)

      return new Promise((resolve, reject) => {
        this.vadProcessor!.onaudioprocess = (e) => {
          const inputData = e.inputBuffer.getChannelData(0)
          const amplitude = this.calculateAmplitude(inputData)
          const isLoud = amplitude > this.vadThreshold

          const elapsed = (Date.now() - this.vadStartTime) / 1000
          if (elapsed > this.vadMaxDuration) {
            console.log('VAD: Max duration reached, stopping')
            this.showStatus('Max recording time reached', 'warning')
            this.stopRecordingVAD().then(resolve).catch(reject)
            return
          }

          if (this.vadRecording) {
            this.vadRecordedChunks.push(new Float32Array(inputData))

            if (isLoud) {
              this.vadSilentChunks = 0
            } else {
              this.vadSilentChunks++
              if (this.vadSilentChunks > numSilentChunksToStop) {
                console.log('VAD: Silence detected, stopping')
                this.showStatus('Processing...', 'info')
                this.stopRecordingVAD().then(resolve).catch(reject)
                return
              }
            }
          } else if (isLoud) {
            console.log('VAD: Speech detected, starting recording')
            this.showStatus('Recording...', 'info')
            this.vadRecording = true
            this.vadRecordedChunks.push(new Float32Array(inputData))

            const voiceBtn = document.getElementById('voice-btn')
            if (voiceBtn) {
              voiceBtn.classList.remove('vad-listening')
              voiceBtn.classList.add('vad-recording')
            }
          }
        }

        source.connect(this.vadProcessor!)
        this.vadProcessor!.connect(this.audioContext!.destination)

        this.showStatus('Listening for speech...', 'info')
      })

    } catch (error) {
      console.error('Failed to start VAD recording:', error)

      if ((error as DOMException).name === 'NotAllowedError' || (error as DOMException).name === 'PermissionDeniedError') {
        this.showError('Microphone access denied. Please allow microphone access in your browser settings.')
      } else if ((error as DOMException).name === 'NotFoundError') {
        this.showError('No microphone found. Please connect a microphone.')
      } else {
        this.showError('Failed to start recording: ' + (error as Error).message)
      }

      throw error
    }
  }

  /**
   * Stop VAD recording and process audio
   */
  public async stopRecordingVAD(): Promise<string> {
    try {
      this.isRecording = false

      if (this.vadProcessor) {
        this.vadProcessor.disconnect()
        this.vadProcessor = null
      }

      if (this.audioContext) {
        await this.audioContext.close()
        this.audioContext = null
      }

      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop())
        this.stream = null
      }

      const voiceBtn = document.getElementById('voice-btn')
      if (voiceBtn) {
        voiceBtn.classList.remove('vad-listening', 'vad-recording')
      }

      if (this.vadRecordedChunks.length === 0) {
        console.log('No speech detected')
        this.showStatus('No speech detected', 'warning')
        return ''
      }

      const totalLength = this.vadRecordedChunks.reduce((acc, chunk) => acc + chunk.length, 0)
      const mergedAudio = new Float32Array(totalLength)
      let offset = 0
      for (const chunk of this.vadRecordedChunks) {
        mergedAudio.set(chunk, offset)
        offset += chunk.length
      }

      const audioBlob = this.float32ArrayToWav(mergedAudio, 16000)
      console.log('VAD recording complete, blob size:', audioBlob.size)

      this.showStatus('Transcribing...', 'info')
      const text = await this.uploadAudio(audioBlob)

      this.vadRecordedChunks = []
      return text

    } catch (error) {
      console.error('Failed to stop VAD recording:', error)
      this.showError('Recording processing failed: ' + (error as Error).message)
      throw error
    }
  }

  /**
   * Calculate amplitude of audio samples
   */
  private calculateAmplitude(samples: Float32Array): number {
    let sum = 0
    for (let i = 0; i < samples.length; i++) {
      sum += Math.abs(samples[i])
    }
    return sum / samples.length
  }

  /**
   * Convert Float32Array audio samples to WAV blob
   */
  private float32ArrayToWav(samples: Float32Array, sampleRate: number): Blob {
    const buffer = new ArrayBuffer(44 + samples.length * 2)
    const view = new DataView(buffer)

    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i))
      }
    }

    const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array) => {
      for (let i = 0; i < input.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, input[i]))
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, 36 + samples.length * 2, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, 1, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * 2, true)
    view.setUint16(32, 2, true)
    view.setUint16(34, 16, true)
    writeString(36, 'data')
    view.setUint32(40, samples.length * 2, true)

    floatTo16BitPCM(view, 44, samples)

    return new Blob([view], { type: 'audio/wav' })
  }

  /**
   * Upload audio blob to backend for transcription
   */
  public async uploadAudio(blob: Blob): Promise<string> {
    try {
      const formData = new FormData()
      formData.append('audio', blob, 'recording.webm')

      const url = window.backendConnector ? await window.backendConnector.buildURL('/api/audio/upload') : '/api/audio/upload'
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'X-Session-ID': window.sessionId || ''
        },
        body: formData
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Upload failed')
      }

      const data = await response.json()

      if (data.status !== 'success' || !data.text) {
        throw new Error('Transcription failed or returned empty text')
      }

      console.log('Transcription successful:', data.text)
      this.showStatus('Transcription complete', 'success')

      return data.text

    } catch (error) {
      console.error('Audio upload failed:', error)
      this.showError('Transcription failed: ' + (error as Error).message)
      throw error
    }
  }

  /**
   * Generate and play TTS audio for text
   */
  public async playTTS(text: string, language: string = 'auto'): Promise<void> {
    try {
      if (!text || !text.trim()) {
        console.warn('Empty text for TTS')
        return
      }

      // Stop any currently playing audio
      if (this.playingAudio) {
        this.playingAudio.pause()
        this.playingAudio = null
      }

      console.log('Generating TTS for text:', text.substring(0, 100) + '...')
      this.isTTSActive = true
      this.showTTSStatus('Generating speech...')

      const url = window.backendConnector ? await window.backendConnector.buildURL('/api/audio/tts') : '/api/audio/tts'
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': window.sessionId || ''
        },
        body: JSON.stringify({ text, language })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'TTS failed')
      }

      const audioBlob = await response.blob()
      const audioUrl = URL.createObjectURL(audioBlob)

      this.playingAudio = new Audio(audioUrl)
      this.playingAudio.volume = this.volume

      this.playingAudio.onended = () => {
        URL.revokeObjectURL(audioUrl)
        this.playingAudio = null
        this.isTTSActive = false
        this.hideTTSStatus()
        console.log('TTS playback finished')
      }

      this.playingAudio.onerror = (error) => {
        console.error('TTS playback error:', error)
        URL.revokeObjectURL(audioUrl)
        this.playingAudio = null
        this.isTTSActive = false
        this.hideTTSStatus()
        this.showError('Audio playback failed')
      }

      await this.playingAudio.play()
      this.showTTSStatus('Speaking...')
      console.log('TTS playback started')

    } catch (error) {
      console.error('TTS playback failed:', error)
      this.isTTSActive = false
      this.hideTTSStatus()
      this.showError('Text-to-speech failed: ' + (error as Error).message)
      throw error
    }
  }

  /**
   * Stop currently playing audio
   */
  public stopAudio(): void {
    if (this.playingAudio) {
      this.playingAudio.pause()
      this.playingAudio = null
      this.isTTSActive = false
      this.hideTTSStatus()
      console.log('Audio playback stopped')
    }

    // Clear queue
    this.audioQueue = []
    this.isPlayingQueue = false
  }

  /**
   * Show TTS status indicator in UI
   */
  private showTTSStatus(message: string): void {
    let statusEl = document.getElementById('tts-status-indicator')
    if (!statusEl) {
      statusEl = document.createElement('div')
      statusEl.id = 'tts-status-indicator'
      statusEl.className = 'tts-status-indicator'
      document.body.appendChild(statusEl)
    }
    statusEl.textContent = `🔊 ${message}`
    statusEl.style.display = 'block'
  }

  /**
   * Hide TTS status indicator
   */
  private hideTTSStatus(): void {
    const statusEl = document.getElementById('tts-status-indicator')
    if (statusEl) {
      statusEl.style.display = 'none'
    }
  }

  /**
   * Play TTS audio from WebSocket stream (base64 encoded)
   */
  public async playWebSocketAudio(audioData: WebSocketAudioData): Promise<void> {
    try {
      if (!audioData || !audioData.audio_data) {
        console.warn('Invalid WebSocket audio data')
        return
      }

      console.log('Received WebSocket TTS audio:', {
        text: audioData.text?.substring(0, 50) + '...',
        format: audioData.audio_format,
        duration: audioData.duration,
        language: audioData.language
      })

      // Decode base64 audio data
      const binaryString = atob(audioData.audio_data)
      const bytes = new Uint8Array(binaryString.length)
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i)
      }

      // Create audio blob
      const audioBlob = new Blob([bytes], { type: 'audio/wav' })
      const audioUrl = URL.createObjectURL(audioBlob)

      // Queue audio for playback
      this.queueAudio(audioUrl, audioData)

    } catch (error) {
      console.error('Failed to play WebSocket audio:', error)
      this.showError('Audio playback failed: ' + (error as Error).message)
    }
  }

  /**
   * Queue audio for sequential playback
   */
  private queueAudio(audioUrl: string, metadata: Partial<WebSocketAudioData> = {}): void {
    this.audioQueue.push({ url: audioUrl, metadata })
    console.log(`Audio queued (${this.audioQueue.length} in queue)`)

    // Start playing queue if not already playing
    if (!this.isPlayingQueue) {
      this.playNextInQueue()
    }
  }

  /**
   * Play next audio in queue
   */
  private async playNextInQueue(): Promise<void> {
    if (this.audioQueue.length === 0) {
      this.isPlayingQueue = false
      console.log('Audio queue empty')
      return
    }

    // Check for user interaction before playing
    if (!this.hasUserInteraction) {
      console.warn('Skipping audio playback - no user interaction yet (browser autoplay policy)')
      this.isPlayingQueue = false
      this.audioQueue = [] // Clear queue since we can't play
      return
    }

    this.isPlayingQueue = true
    const item = this.audioQueue.shift()!

    try {
      // Stop any currently playing audio
      if (this.playingAudio) {
        this.playingAudio.pause()
        this.playingAudio = null
      }

      this.playingAudio = new Audio(item.url)

      // Apply volume if specified
      if (item.metadata.volume !== undefined) {
        this.playingAudio.volume = Math.max(0, Math.min(1, item.metadata.volume))
      }

      this.playingAudio.onended = () => {
        URL.revokeObjectURL(item.url)
        this.playingAudio = null
        console.log('Audio playback finished')

        // Play next in queue
        this.playNextInQueue()
      }

      this.playingAudio.onerror = (error) => {
        console.error('Audio playback error:', error)
        URL.revokeObjectURL(item.url)
        this.playingAudio = null

        // Continue to next in queue
        this.playNextInQueue()
      }

      await this.playingAudio.play()
      console.log('Queue playback started')

    } catch (error) {
      console.error('Queue playback failed:', error)
      URL.revokeObjectURL(item.url)
      this.playingAudio = null

      // Continue to next in queue
      this.playNextInQueue()
    }
  }

  /**
   * Set audio volume
   */
  public setVolume(volume: number): void {
    this.volume = Math.max(0, Math.min(1, volume))
    if (this.playingAudio) {
      this.playingAudio.volume = this.volume
    }
    console.log('Audio volume set to:', this.volume)
  }

  /**
   * Show status message
   */
  private showStatus(message: string, type: string): void {
    console.log(`[Audio Status] ${type.toUpperCase()}: ${message}`)
  }

  /**
   * Show error message
   */
  private showError(message: string): void {
    console.error(`[Audio Error] ${message}`)
  }
}

export default AudioManager
export { AudioManager }
