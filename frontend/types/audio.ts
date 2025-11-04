/**
 * Audio and voice interaction types
 */

/**
 * Audio recording configuration
 */
export interface AudioRecordingConfig {
  /** MIME type for recording */
  mimeType: string
  /** Audio bitrate */
  audioBitsPerSecond?: number
  /** Sample rate */
  sampleRate?: number
}

/**
 * Transcription result
 */
export interface TranscriptionResult {
  /** Transcribed text */
  text: string
  /** Confidence score (0-1) */
  confidence?: number
  /** Language detected */
  language?: string
  /** Processing time in seconds */
  duration?: number
}

/**
 * TTS (Text-to-Speech) request
 */
export interface TTSRequest {
  /** Text to synthesize */
  text: string
  /** Voice ID or name */
  voice?: string
  /** Speech rate (0.5-2.0) */
  rate?: number
  /** Pitch adjustment (-10 to 10) */
  pitch?: number
}

/**
 * TTS response
 */
export interface TTSResponse {
  /** Audio data (base64 or URL) */
  audio: string
  /** Audio format (mp3, wav, etc.) */
  format: string
  /** Duration in seconds */
  duration?: number
}

/**
 * Voice activity detection result
 */
export interface VADResult {
  /** Whether speech is detected */
  isSpeaking: boolean
  /** Confidence score */
  confidence: number
  /** Energy level */
  energy: number
}

/**
 * Audio queue item
 */
export interface AudioQueueItem {
  /** Unique ID */
  id: string
  /** Audio data URL or base64 */
  audioData: string
  /** Source text */
  text: string
  /** Priority (higher = more urgent) */
  priority?: number
}
