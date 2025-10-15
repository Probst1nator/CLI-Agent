/**
 * AudioManager - Browser audio recording and TTS playback
 *
 * Handles microphone recording, uploading to backend for transcription,
 * and playing TTS audio responses.
 */

class AudioManager {
    constructor() {
        this.recorder = null;
        this.audioContext = null;
        this.stream = null;
        this.chunks = [];
        this.playingAudio = null;
        this.isRecording = false;
        this.audioQueue = [];
        this.isPlayingQueue = false;
        this.hasUserInteraction = false;
        this.isTTSActive = false;
        this.volume = 1.0;

        console.log('AudioManager initialized');

        // Set flag on first user interaction to allow autoplay
        const enableAutoplay = () => {
            this.hasUserInteraction = true;
            console.log('User interaction detected - audio autoplay enabled');
            document.removeEventListener('click', enableAutoplay);
            document.removeEventListener('keydown', enableAutoplay);
        };
        document.addEventListener('click', enableAutoplay, { once: true });
        document.addEventListener('keydown', enableAutoplay, { once: true });
    }

    /**
     * Start recording from microphone
     */
    async startRecording() {
        try {
            if (this.isRecording) {
                console.warn('Already recording');
                return;
            }

            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.recorder = new MediaRecorder(this.stream);
            this.chunks = [];

            this.recorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    this.chunks.push(e.data);
                }
            };

            this.recorder.start();
            this.isRecording = true;

            console.log('Recording started');
            this.showStatus('Recording...', 'info');

        } catch (error) {
            console.error('Failed to start recording:', error);

            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                this.showError('Microphone access denied. Please allow microphone access in your browser settings.');
            } else if (error.name === 'NotFoundError') {
                this.showError('No microphone found. Please connect a microphone.');
            } else {
                this.showError('Failed to start recording: ' + error.message);
            }

            throw error;
        }
    }

    /**
     * Stop recording and upload for transcription
     * @returns {Promise<string>} Transcribed text
     */
    async stopRecording() {
        try {
            if (!this.isRecording || !this.recorder) {
                console.warn('Not recording');
                return '';
            }

            return new Promise((resolve, reject) => {
                this.recorder.onstop = async () => {
                    this.isRecording = false;

                    if (this.stream) {
                        this.stream.getTracks().forEach(track => track.stop());
                        this.stream = null;
                    }

                    try {
                        const audioBlob = new Blob(this.chunks, { type: 'audio/webm' });
                        console.log('Recording stopped, blob size:', audioBlob.size);

                        this.showStatus('Transcribing...', 'info');
                        const text = await this.uploadAudio(audioBlob);

                        resolve(text);
                    } catch (error) {
                        reject(error);
                    }
                };

                this.recorder.stop();
            });

        } catch (error) {
            console.error('Failed to stop recording:', error);
            this.showError('Recording upload failed: ' + error.message);
            throw error;
        }
    }

    /**
     * Upload audio blob to backend for transcription
     * @param {Blob} blob - Audio blob to upload
     * @returns {Promise<string>} Transcribed text
     */
    async uploadAudio(blob) {
        try {
            const formData = new FormData();
            formData.append('audio', blob, 'recording.webm');

            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/audio/upload') : '/api/audio/upload';
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'X-Session-ID': window.sessionId || ''
                },
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Upload failed');
            }

            const data = await response.json();

            if (data.status !== 'success' || !data.text) {
                throw new Error('Transcription failed or returned empty text');
            }

            console.log('Transcription successful:', data.text);
            this.showStatus('Transcription complete', 'success');

            return data.text;

        } catch (error) {
            console.error('Audio upload failed:', error);
            this.showError('Transcription failed: ' + error.message);
            throw error;
        }
    }

    /**
     * Generate and play TTS audio for text
     * @param {string} text - Text to speak
     * @param {string} language - Language code (default: 'en')
     */
    async playTTS(text, language = 'auto') {
        try {
            if (!text || !text.trim()) {
                console.warn('Empty text for TTS');
                return;
            }

            // Stop any currently playing audio
            if (this.playingAudio) {
                this.playingAudio.pause();
                this.playingAudio = null;
            }

            console.log('Generating TTS for text:', text.substring(0, 100) + '...');
            this.isTTSActive = true;
            this.showTTSStatus('Generating speech...');

            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/audio/tts') : '/api/audio/tts';
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': window.sessionId || ''
                },
                body: JSON.stringify({ text, language })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'TTS failed');
            }

            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);

            this.playingAudio = new Audio(audioUrl);

            this.showTTSStatus('Speaking...');

            this.playingAudio.volume = this.volume;

            this.playingAudio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                this.playingAudio = null;
                this.isTTSActive = false;
                this.hideTTSStatus();
                console.log('TTS playback finished');
            };

            this.playingAudio.onerror = (error) => {
                console.error('Audio playback error:', error);
                URL.revokeObjectURL(audioUrl);
                this.playingAudio = null;
                this.isTTSActive = false;
                this.hideTTSStatus();
                this.showError('Audio playback failed');
            };

            await this.playingAudio.play();
            console.log('TTS playback started');

        } catch (error) {
            console.error('TTS playback failed:', error);
            this.isTTSActive = false;
            this.hideTTSStatus();
            this.showError('Text-to-speech failed: ' + error.message);
            throw error;
        }
    }

    stopAudio() {
        if (this.playingAudio) {
            this.playingAudio.pause();
            this.playingAudio = null;
            this.isTTSActive = false;
            this.hideTTSStatus();
            console.log('Audio playback stopped');
        }
    }

    showTTSStatus(message) {
        let statusEl = document.getElementById('tts-status-indicator');
        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.id = 'tts-status-indicator';
            statusEl.className = 'tts-status-indicator';
            document.body.appendChild(statusEl);
        }
        statusEl.textContent = `🔊 ${message}`;
        statusEl.style.display = 'block';
    }

    hideTTSStatus() {
        const statusEl = document.getElementById('tts-status-indicator');
        if (statusEl) {
            statusEl.style.display = 'none';
        }
    }

    /**
     * Play TTS audio from WebSocket stream (base64 encoded)
     * @param {Object} audioData - Audio payload from WebSocket
     */
    async playWebSocketAudio(audioData) {
        try {
            if (!audioData || !audioData.audio_data) {
                console.warn('Invalid WebSocket audio data');
                return;
            }

            console.log('Received WebSocket TTS audio:', {
                text: audioData.text?.substring(0, 50) + '...',
                format: audioData.audio_format,
                duration: audioData.duration,
                language: audioData.language
            });

            // Decode base64 audio data
            const binaryString = atob(audioData.audio_data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Create audio blob
            const audioBlob = new Blob([bytes], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);

            // Queue audio for playback
            this.queueAudio(audioUrl, audioData);

        } catch (error) {
            console.error('Failed to play WebSocket audio:', error);
            this.showError('Audio playback failed: ' + error.message);
        }
    }

    /**
     * Queue audio for sequential playback
     * @param {string} audioUrl - Object URL of audio blob
     * @param {Object} metadata - Audio metadata
     */
    queueAudio(audioUrl, metadata = {}) {
        this.audioQueue.push({ url: audioUrl, metadata });
        console.log(`Audio queued (${this.audioQueue.length} in queue)`);

        // Start playing queue if not already playing
        if (!this.isPlayingQueue) {
            this.playNextInQueue();
        }
    }

    /**
     * Play next audio in queue
     */
    async playNextInQueue() {
        if (this.audioQueue.length === 0) {
            this.isPlayingQueue = false;
            console.log('Audio queue empty');
            return;
        }

        // Check for user interaction before playing
        if (!this.hasUserInteraction) {
            console.warn('Skipping audio playback - no user interaction yet (browser autoplay policy)');
            this.isPlayingQueue = false;
            this.audioQueue = []; // Clear queue since we can't play
            return;
        }

        this.isPlayingQueue = true;
        const { url, metadata } = this.audioQueue.shift();

        try {
            // Stop any currently playing audio
            if (this.playingAudio) {
                this.playingAudio.pause();
                this.playingAudio = null;
            }

            this.playingAudio = new Audio(url);

            // Apply volume if specified
            if (metadata.volume !== undefined) {
                this.playingAudio.volume = Math.max(0, Math.min(1, metadata.volume));
            }

            this.playingAudio.onended = () => {
                URL.revokeObjectURL(url);
                this.playingAudio = null;
                console.log('Audio playback finished');

                // Play next in queue
                this.playNextInQueue();
            };

            this.playingAudio.onerror = (error) => {
                console.error('Audio playback error:', error);
                URL.revokeObjectURL(url);
                this.playingAudio = null;

                // Continue to next in queue on error
                this.playNextInQueue();
            };

            await this.playingAudio.play();
            console.log('Playing audio from queue');

        } catch (error) {
            console.error('Failed to play queued audio:', error);
            URL.revokeObjectURL(url);
            this.playingAudio = null;

            // Continue to next in queue on error
            this.playNextInQueue();
        }
    }

    /**
     * Stop currently playing audio and clear queue
     */
    stopAudio() {
        // Stop current playback
        if (this.playingAudio) {
            this.playingAudio.pause();
            this.playingAudio = null;
            console.log('Audio stopped');
        }

        // Clear queue and revoke URLs
        this.audioQueue.forEach(item => {
            if (item.url) {
                URL.revokeObjectURL(item.url);
            }
        });
        this.audioQueue = [];
        this.isPlayingQueue = false;

        console.log('Audio queue cleared');
    }

    /**
     * Check if microphone is available
     * @returns {Promise<boolean>}
     */
    async checkMicrophoneAccess() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioInputs = devices.filter(device => device.kind === 'audioinput');

            if (audioInputs.length === 0) {
                console.warn('No microphone devices found');
                return false;
            }

            return true;

        } catch (error) {
            console.error('Failed to check microphone access:', error);
            return false;
        }
    }

    /**
     * Get voice mode status from backend
     * @returns {Promise<boolean>}
     */
    async getVoiceStatus() {
        try {
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/audio/voice_status') : '/api/audio/voice_status';
            const response = await fetch(url, {
                headers: {
                    'X-Session-ID': window.sessionId || ''
                }
            });

            if (!response.ok) {
                throw new Error('Failed to get voice status');
            }

            const data = await response.json();
            return data.voice_mode || false;

        } catch (error) {
            console.error('Failed to get voice status:', error);
            return false;
        }
    }

    /**
     * Show status message
     */
    showStatus(message, type = 'info') {
        if (typeof showToast === 'function') {
            showToast(message, type);
        } else {
            console.log(`[${type}] ${message}`);
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        if (typeof showToast === 'function') {
            showToast(message, 'error');
        } else {
            console.error('❌', message);
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        if (this.playingAudio) {
            this.playingAudio.pause();
            this.playingAudio = null;
        }

        this.isRecording = false;
        console.log('AudioManager cleaned up');
    }

    setVolume(volumePercent) {
        this.volume = Math.max(0, Math.min(1, volumePercent / 100));
        console.log('TTS volume set to', this.volume);

        if (this.playingAudio) {
            this.playingAudio.volume = this.volume;
        }
    }

    getVolume() {
        return this.volume * 100;
    }
}

window.AudioManager = AudioManager;

window.addEventListener('beforeunload', () => {
    if (window.audioManager) {
        window.audioManager.cleanup();
    }
});
