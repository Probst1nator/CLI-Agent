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

        // VAD (Voice Activity Detection) settings
        this.vadEnabled = false;
        this.vadThreshold = 0.02;
        this.vadSilenceDuration = 2.0;
        this.vadMaxDuration = 20.0;
        this.vadRecording = false;
        this.vadSilentChunks = 0;
        this.vadRecordedChunks = [];
        this.vadStartTime = 0;
        this.vadProcessor = null;

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
     * Start recording with automatic Voice Activity Detection
     * Automatically stops after silence_duration seconds of silence
     * @returns {Promise<string>} Transcribed text
     */
    async startRecordingWithVAD() {
        try {
            if (this.isRecording) {
                console.warn('Already recording');
                return '';
            }

            console.log(`Starting VAD recording (threshold: ${this.vadThreshold}, silence: ${this.vadSilenceDuration}s, max: ${this.vadMaxDuration}s)`);

            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });

            const source = this.audioContext.createMediaStreamSource(this.stream);

            this.vadProcessor = this.audioContext.createScriptProcessor(1600, 1, 1);

            this.vadRecording = false;
            this.vadSilentChunks = 0;
            this.vadRecordedChunks = [];
            this.vadStartTime = Date.now();
            this.isRecording = true;

            const numSilentChunksToStop = Math.floor(this.vadSilenceDuration * 10);

            return new Promise((resolve, reject) => {
                this.vadProcessor.onaudioprocess = (e) => {
                    const inputData = e.inputBuffer.getChannelData(0);
                    const amplitude = this.calculateAmplitude(inputData);
                    const isLoud = amplitude > this.vadThreshold;

                    const elapsed = (Date.now() - this.vadStartTime) / 1000;
                    if (elapsed > this.vadMaxDuration) {
                        console.log('VAD: Max duration reached, stopping');
                        this.showStatus('Max recording time reached', 'warning');
                        this.stopRecordingVAD().then(resolve).catch(reject);
                        return;
                    }

                    if (this.vadRecording) {
                        this.vadRecordedChunks.push(new Float32Array(inputData));

                        if (isLoud) {
                            this.vadSilentChunks = 0;
                        } else {
                            this.vadSilentChunks++;
                            if (this.vadSilentChunks > numSilentChunksToStop) {
                                console.log('VAD: Silence detected, stopping');
                                this.showStatus('Processing...', 'info');
                                this.stopRecordingVAD().then(resolve).catch(reject);
                                return;
                            }
                        }
                    } else if (isLoud) {
                        console.log('VAD: Speech detected, starting recording');
                        this.showStatus('Recording...', 'info');
                        this.vadRecording = true;
                        this.vadRecordedChunks.push(new Float32Array(inputData));

                        const voiceBtn = document.getElementById('voice-btn');
                        if (voiceBtn) {
                            voiceBtn.classList.remove('vad-listening');
                            voiceBtn.classList.add('vad-recording');
                        }
                    }
                };

                source.connect(this.vadProcessor);
                this.vadProcessor.connect(this.audioContext.destination);

                this.showStatus('Listening for speech...', 'info');
            });

        } catch (error) {
            console.error('Failed to start VAD recording:', error);

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
     * Stop VAD recording and process audio
     * @returns {Promise<string>} Transcribed text
     */
    async stopRecordingVAD() {
        try {
            this.isRecording = false;

            if (this.vadProcessor) {
                this.vadProcessor.disconnect();
                this.vadProcessor = null;
            }

            if (this.audioContext) {
                await this.audioContext.close();
                this.audioContext = null;
            }

            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }

            const voiceBtn = document.getElementById('voice-btn');
            if (voiceBtn) {
                voiceBtn.classList.remove('vad-listening', 'vad-recording');
            }

            if (this.vadRecordedChunks.length === 0) {
                console.log('No speech detected');
                this.showStatus('No speech detected', 'warning');
                return '';
            }

            const totalLength = this.vadRecordedChunks.reduce((acc, chunk) => acc + chunk.length, 0);
            const mergedAudio = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of this.vadRecordedChunks) {
                mergedAudio.set(chunk, offset);
                offset += chunk.length;
            }

            const audioBlob = this.float32ArrayToWav(mergedAudio, 16000);
            console.log('VAD recording complete, blob size:', audioBlob.size);

            this.showStatus('Transcribing...', 'info');
            const text = await this.uploadAudio(audioBlob);

            this.vadRecordedChunks = [];
            return text;

        } catch (error) {
            console.error('Failed to stop VAD recording:', error);
            this.showError('Recording processing failed: ' + error.message);
            throw error;
        }
    }

    /**
     * Calculate amplitude of audio samples
     * @param {Float32Array} samples - Audio samples
     * @returns {number} Average absolute amplitude
     */
    calculateAmplitude(samples) {
        let sum = 0;
        for (let i = 0; i < samples.length; i++) {
            sum += Math.abs(samples[i]);
        }
        return sum / samples.length;
    }

    /**
     * Convert Float32Array audio to WAV blob
     * @param {Float32Array} samples - Audio samples
     * @param {number} sampleRate - Sample rate
     * @returns {Blob} WAV audio blob
     */
    float32ArrayToWav(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        const floatTo16BitPCM = (output, offset, input) => {
            for (let i = 0; i < input.length; i++, offset += 2) {
                const s = Math.max(-1, Math.min(1, input[i]));
                output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        floatTo16BitPCM(view, 44, samples);

        return new Blob([view], { type: 'audio/wav' });
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
