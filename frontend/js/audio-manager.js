/**
 * AudioManager - Browser audio recording and TTS playback
 *
 * Handles microphone recording via MediaRecorder API, uploading to backend for transcription,
 * and playing TTS audio responses with queue management.
 *
 * @class AudioManager
 */

class AudioManager {
    constructor() {
        /**
         * MediaRecorder instance for capturing audio
         * @type {MediaRecorder|null}
         * @private
         */
        this.recorder = null;

        /**
         * Web Audio API AudioContext for VAD processing
         * @type {AudioContext|null}
         * @private
         */
        this.audioContext = null;

        /**
         * MediaStream from getUserMedia for microphone access
         * @type {MediaStream|null}
         * @private
         */
        this.stream = null;

        /**
         * Array of Blob chunks collected during recording
         * @type {Blob[]}
         * @private
         */
        this.chunks = [];

        /**
         * Currently playing HTML Audio element
         * @type {HTMLAudioElement|null}
         * @private
         */
        this.playingAudio = null;

        /**
         * Flag indicating if recording is in progress
         * @type {boolean}
         * @private
         */
        this.isRecording = false;

        /**
         * Queue of audio items waiting to be played
         * @type {Array<{url: string, metadata: Object}>}
         * @private
         */
        this.audioQueue = [];

        /**
         * Flag indicating if queue playback is active
         * @type {boolean}
         * @private
         */
        this.isPlayingQueue = false;

        /**
         * Flag indicating if user has interacted (required for autoplay)
         * @type {boolean}
         * @private
         */
        this.hasUserInteraction = false;

        /**
         * Flag indicating if TTS is currently active
         * @type {boolean}
         * @private
         */
        this.isTTSActive = false;

        /**
         * Audio volume level (0.0 to 1.0)
         * @type {number}
         * @private
         */
        this.volume = 1.0;

        // VAD (Voice Activity Detection) settings

        /**
         * WebSocket for continuous audio streaming
         * @type {WebSocket}
         * @private
         */
        this.audioStreamWS = null;

        /**
         * AudioContext for processing streaming audio
         * @type {AudioContext}
         * @private
         */
        this.streamAudioContext = null;

        /**
         * ScriptProcessor for capturing audio chunks
         * @type {ScriptProcessorNode}
         * @private
         */
        this.streamProcessor = null;

        /**
         * Flag indicating if streaming mode is active
         * @type {boolean}
         * @private
         */
        this.isStreaming = false;

        /**
         * Callback for when streaming transcription is received
         * @type {Function}
         * @private
         */
        this.onStreamTranscription = null;

        /**
         * Flag indicating if VAD has detected speech and started recording
         * @type {boolean}
         * @private
         */
        this.vadRecording = false;

        /**
         * Counter for consecutive silent audio chunks in VAD mode
         * @type {number}
         * @private
         */
        this.vadSilentChunks = 0;

        /**
         * Array of Float32Array audio chunks recorded during VAD session
         * @type {Float32Array[]}
         * @private
         */
        this.vadRecordedChunks = [];

        /**
         * Timestamp (milliseconds) when VAD recording started
         * @type {number}
         * @private
         */
        this.vadStartTime = 0;

        /**
         * ScriptProcessorNode for VAD audio analysis
         * @type {ScriptProcessorNode|null}
         * @private
         */
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
     * Start recording from microphone using MediaRecorder API
     *
     * Initializes MediaRecorder with getUserMedia stream and collects audio chunks.
     * Event handler 'ondataavailable' appends Blob chunks to this.chunks array.
     *
     * @async
     * @returns {Promise<void>}
     * @throws {NotAllowedError} When microphone permission is denied
     * @throws {NotFoundError} When no microphone device is found
     * @throws {Error} For other recording initialization failures
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
     *
     * Stops MediaRecorder, creates audio Blob from collected chunks,
     * uploads to backend, and returns transcribed text.
     * Event handler 'onstop' is set to handle async upload process.
     *
     * @async
     * @returns {Promise<string>} Transcribed text from backend
     * @throws {Error} When upload or transcription fails
     */
    async stopRecording() {
        try {
            console.log('[AudioManager] stopRecording() called');
            console.log('[AudioManager] isRecording:', this.isRecording);
            console.log('[AudioManager] recorder:', this.recorder);

            if (!this.isRecording || !this.recorder) {
                console.warn('[AudioManager] Not recording - returning empty string');
                return '';
            }

            console.log('[AudioManager] Creating promise to wait for recorder.onstop');

            return new Promise((resolve, reject) => {
                this.recorder.onstop = async () => {
                    console.log('[AudioManager] recorder.onstop fired!');
                    this.isRecording = false;

                    if (this.stream) {
                        this.stream.getTracks().forEach(track => track.stop());
                        this.stream = null;
                    }

                    try {
                        const audioBlob = new Blob(this.chunks, { type: 'audio/webm' });
                        console.log('[AudioManager] Recording stopped, blob size:', audioBlob.size, 'bytes');
                        console.log('[AudioManager] Chunks collected:', this.chunks.length);

                        if (audioBlob.size === 0) {
                            console.error('[AudioManager] Audio blob is empty! No data recorded.');
                            throw new Error('No audio data recorded');
                        }

                        this.showStatus('Transcribing...', 'info');
                        console.log('[AudioManager] Calling uploadAudio()...');
                        const text = await this.uploadAudio(audioBlob);
                        console.log('[AudioManager] uploadAudio() returned:', text);

                        resolve(text);
                    } catch (error) {
                        console.error('[AudioManager] Error in onstop handler:', error);
                        reject(error);
                    }
                };

                console.log('[AudioManager] Calling recorder.stop()...');
                this.recorder.stop();
                console.log('[AudioManager] recorder.stop() called, waiting for onstop event');
            });

        } catch (error) {
            console.error('[AudioManager] Failed to stop recording:', error);
            this.showError('Recording upload failed: ' + error.message);
            throw error;
        }
    }

    /**
     * Start continuous audio streaming with backend VAD
     *
     * Opens WebSocket connection and streams audio chunks to backend.
     * Backend handles VAD and sends back transcriptions when silence detected.
     *
     * @async
     * @param {Function} onTranscription - Callback for when transcription is received
     * @returns {Promise<void>}
     * @throws {Error} When streaming fails to start
     */
    async startAudioStreaming(onTranscription) {
        try {
            if (this.isStreaming) {
                console.warn('[AudioManager] Already streaming');
                return;
            }

            console.log('[AudioManager] Starting audio streaming...');
            this.onStreamTranscription = onTranscription;

            // Get base URL for WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            const wsUrl = `${protocol}//${host}/api/audio/stream`;

            console.log('[AudioManager] Connecting to WebSocket:', wsUrl);

            // Open WebSocket connection
            this.audioStreamWS = new WebSocket(wsUrl);

            this.audioStreamWS.onopen = async () => {
                console.log('[AudioManager] WebSocket connected');

                try {
                    // Start microphone capture
                    this.stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            channelCount: 1,
                            sampleRate: 16000,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });

                    // Create audio context (use native sample rate to avoid mismatch)
                    this.streamAudioContext = new (window.AudioContext || window.webkitAudioContext)();

                    const source = this.streamAudioContext.createMediaStreamSource(this.stream);

                    // Store native sample rate for resampling
                    const nativeSampleRate = this.streamAudioContext.sampleRate;
                    console.log(`[AudioManager] Native sample rate: ${nativeSampleRate}Hz, target: 16000Hz`);

                    // Create script processor for capturing audio chunks
                    // Buffer size: 4096 samples
                    this.streamProcessor = this.streamAudioContext.createScriptProcessor(4096, 1, 1);

                    this.streamProcessor.onaudioprocess = (e) => {
                        if (this.audioStreamWS && this.audioStreamWS.readyState === WebSocket.OPEN) {
                            // Get audio data as Float32Array
                            const audioData = e.inputBuffer.getChannelData(0);

                            // Resample to 16kHz if needed
                            let resampledData = audioData;
                            if (nativeSampleRate !== 16000) {
                                resampledData = this.resampleAudio(audioData, nativeSampleRate, 16000);
                            }

                            // Send as binary to backend
                            this.audioStreamWS.send(resampledData.buffer);
                        }
                    };

                    // Connect audio graph
                    source.connect(this.streamProcessor);
                    this.streamProcessor.connect(this.streamAudioContext.destination);

                    this.isStreaming = true;
                    console.log('[AudioManager] Audio streaming started');

                    // Play listening notification sound
                    this.playNotificationSound('listening');

                    this.showStatus('Listening...', 'info');

                } catch (error) {
                    console.error('[AudioManager] Failed to start audio capture:', error);
                    this.audioStreamWS.close();
                    throw error;
                }
            };

            this.audioStreamWS.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('[AudioManager] Received from backend:', data);

                    if (data.type === 'processing') {
                        // Backend detected silence and is starting transcription
                        console.log('[AudioManager] Processing started');
                        this.playNotificationSound('processing');
                        this.showStatus('Processing...', 'info');
                    } else if (data.type === 'transcription') {
                        if (data.text && data.text.trim()) {
                            console.log('[AudioManager] Transcription received:', data.text);
                            if (this.onStreamTranscription) {
                                this.onStreamTranscription(data.text, data.language);
                            }
                        } else if (data.error) {
                            console.warn('[AudioManager] Backend error:', data.error);
                        }

                        // Restore listening status after transcription
                        this.showStatus('Listening...', 'info');
                    } else if (data.type === 'error') {
                        console.error('[AudioManager] Backend error:', data.error);
                        this.showError(data.error);
                    }
                } catch (error) {
                    console.error('[AudioManager] Error parsing WebSocket message:', error);
                }
            };

            this.audioStreamWS.onerror = (error) => {
                console.error('[AudioManager] WebSocket error:', error);
                this.showError('Audio streaming connection error');
            };

            this.audioStreamWS.onclose = () => {
                console.log('[AudioManager] WebSocket closed');
                this.stopAudioStreaming();
            };

        } catch (error) {
            console.error('[AudioManager] Failed to start audio streaming:', error);
            this.showError('Failed to start audio streaming: ' + error.message);
            throw error;
        }
    }

    /**
     * Stop continuous audio streaming
     */
    stopAudioStreaming() {
        console.log('[AudioManager] Stopping audio streaming...');

        // Stop audio processing
        if (this.streamProcessor) {
            this.streamProcessor.disconnect();
            this.streamProcessor = null;
        }

        // Close audio context
        if (this.streamAudioContext) {
            this.streamAudioContext.close();
            this.streamAudioContext = null;
        }

        // Stop microphone stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        // Close WebSocket
        if (this.audioStreamWS) {
            this.audioStreamWS.close();
            this.audioStreamWS = null;
        }

        this.isStreaming = false;
        this.onStreamTranscription = null;

        console.log('[AudioManager] Audio streaming stopped');
        this.showStatus('', 'info');
    }

    /**
     * Pause audio recording (during agent response/TTS playback)
     */
    pauseRecording() {
        if (this.audioStreamWS && this.audioStreamWS.readyState === WebSocket.OPEN) {
            try {
                this.audioStreamWS.send(JSON.stringify({ action: 'pause' }));
                console.log('🎤 Paused recording (agent responding)');
            } catch (error) {
                console.error('[AudioManager] Failed to pause recording:', error);
            }
        }
    }

    /**
     * Resume audio recording (after agent response/TTS playback)
     */
    resumeRecording() {
        if (this.audioStreamWS && this.audioStreamWS.readyState === WebSocket.OPEN) {
            try {
                this.audioStreamWS.send(JSON.stringify({ action: 'resume' }));
                console.log('🎤 Resumed recording');
            } catch (error) {
                console.error('[AudioManager] Failed to resume recording:', error);
            }
        }
    }

    /**
     * Start recording with automatic Voice Activity Detection
     *
     * Uses Web Audio API ScriptProcessorNode to analyze audio amplitude in real-time.
     * Waits for speech to start recording, then auto-stops after silence_duration
     * seconds of silence or max_duration seconds total.
     *
     * VAD Flow:
     * 1. Listen for amplitude > vadThreshold to start recording
     * 2. Collect Float32Array chunks while speech detected
     * 3. Count silent chunks, stop when vadSilentChunks > threshold
     * 4. Convert Float32Array to WAV blob and upload for transcription
     *
     * @async
     * @returns {Promise<string>} Transcribed text from backend
     * @throws {NotAllowedError} When microphone permission is denied
     * @throws {NotFoundError} When no microphone device is found
     * @throws {Error} For other recording or processing failures
     */
    async startRecordingWithVAD() {
        try {
            if (this.isRecording) {
                console.warn('Already recording');
                return '';
            }

            console.log(`Starting VAD recording (threshold: ${this.vadThreshold}, silence: ${this.vadSilenceDuration}s, max: ${this.vadMaxDuration}s)`);

            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Create audio context with native sample rate to avoid mismatch
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log(`[VAD] Using native sample rate: ${this.audioContext.sampleRate}Hz`);

            const source = this.audioContext.createMediaStreamSource(this.stream);

            this.vadProcessor = this.audioContext.createScriptProcessor(1600, 1, 1);

            this.vadRecording = false;
            this.vadSilentChunks = 0;
            this.vadRecordedChunks = [];
            this.vadStartTime = Date.now();
            this.vadNativeSampleRate = this.audioContext.sampleRate; // Store for resampling later
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
     *
     * Disconnects ScriptProcessorNode, merges Float32Array chunks,
     * converts to WAV format, and uploads for transcription.
     *
     * @async
     * @returns {Promise<string>} Transcribed text from backend, or empty string if no speech detected
     * @throws {Error} When audio processing or upload fails
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

            // Resample to 16kHz if needed (backend expects 16kHz)
            let finalAudio = mergedAudio;
            if (this.vadNativeSampleRate && this.vadNativeSampleRate !== 16000) {
                console.log(`[VAD] Resampling from ${this.vadNativeSampleRate}Hz to 16000Hz`);
                finalAudio = this.resampleAudio(mergedAudio, this.vadNativeSampleRate, 16000);
            }

            const audioBlob = this.float32ArrayToWav(finalAudio, 16000);
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
     * Resample audio from one sample rate to another using linear interpolation
     *
     * @param {Float32Array} audioData - Input audio samples
     * @param {number} sourceSampleRate - Source sample rate in Hz
     * @param {number} targetSampleRate - Target sample rate in Hz
     * @returns {Float32Array} Resampled audio data
     */
    resampleAudio(audioData, sourceSampleRate, targetSampleRate) {
        if (sourceSampleRate === targetSampleRate) {
            return audioData;
        }

        const ratio = sourceSampleRate / targetSampleRate;
        const newLength = Math.round(audioData.length / ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const sourceIndex = i * ratio;
            const index = Math.floor(sourceIndex);
            const fraction = sourceIndex - index;

            if (index + 1 < audioData.length) {
                // Linear interpolation
                result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
            } else {
                result[i] = audioData[index];
            }
        }

        return result;
    }

    /**
     * Convert Float32Array audio samples to WAV blob
     *
     * Creates a WAV file with proper header and converts Float32 samples
     * to 16-bit PCM format.
     *
     * WAV Format:
     * - RIFF header (12 bytes)
     * - fmt chunk (24 bytes): PCM, mono, 16-bit
     * - data chunk header (8 bytes)
     * - 16-bit PCM audio data
     *
     * @param {Float32Array} samples - Audio samples in range [-1.0, 1.0]
     * @param {number} sampleRate - Sample rate in Hz (e.g., 16000)
     * @returns {Blob} WAV audio blob with type 'audio/wav'
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
     *
     * Sends audio as multipart/form-data to /api/audio/upload endpoint.
     * Backend processes with Whisper and returns transcribed text.
     *
     * @async
     * @param {Blob} blob - Audio blob (webm or wav format)
     * @returns {Promise<string>} Transcribed text from backend
     * @throws {Error} When upload fails or transcription returns empty
     */
    async uploadAudio(blob) {
        try {
            console.log('[AudioManager] uploadAudio() called with blob size:', blob.size);
            console.log('[AudioManager] Blob type:', blob.type);

            const formData = new FormData();
            formData.append('audio', blob, 'recording.webm');
            console.log('[AudioManager] FormData created, appended audio blob');

            const endpoint = '/api/audio/upload';
            const sessionId = window.sessionId || '';
            console.log('[AudioManager] Uploading to:', endpoint);
            console.log('[AudioManager] Session ID:', sessionId);
            console.log('[AudioManager] Using backendConnector:', !!window.backendConnector);

            const response = window.backendConnector
                ? await window.backendConnector.fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'X-Session-ID': sessionId
                    },
                    body: formData
                })
                : await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'X-Session-ID': sessionId
                    },
                    body: formData
                });

            console.log('[AudioManager] Response received, status:', response.status);
            console.log('[AudioManager] Response ok:', response.ok);

            if (!response.ok) {
                console.error('[AudioManager] Response not ok, status:', response.status);
                const error = await response.json();
                console.error('[AudioManager] Error response:', error);
                throw new Error(error.error || 'Upload failed');
            }

            const data = await response.json();
            console.log('[AudioManager] Response data:', data);

            if (data.status !== 'success' || !data.text) {
                console.error('[AudioManager] Invalid response data:', data);
                throw new Error('Transcription failed or returned empty text');
            }

            console.log('[AudioManager] Transcription successful:', data.text);
            this.showStatus('Transcription complete', 'success');

            return data.text;

        } catch (error) {
            console.error('[AudioManager] Audio upload failed:', error);
            console.error('[AudioManager] Error stack:', error.stack);
            this.showError('Transcription failed: ' + error.message);
            throw error;
        }
    }

    /**
     * Generate and play TTS audio for text
     *
     * Requests TTS audio from backend (/api/audio/tts), creates HTML Audio element,
     * and plays immediately. Sets up event handlers for playback lifecycle.
     *
     * Event Handlers:
     * - onended: Cleanup and hide status indicator
     * - onerror: Log error and cleanup
     *
     * @async
     * @param {string} text - Text to convert to speech
     * @param {string} [language='auto'] - Language code (e.g., 'en', 'es') or 'auto' for detection
     * @returns {Promise<void>}
     * @throws {Error} When TTS generation or playback fails
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

            const response = window.backendConnector
                ? await window.backendConnector.fetch('/api/audio/tts', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Session-ID': window.sessionId || ''
                    },
                    body: JSON.stringify({ text, language })
                })
                : await fetch('/api/audio/tts', {
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

    /**
     * Show TTS status indicator in UI
     *
     * Creates or updates a floating status indicator element
     * showing current TTS state (e.g., "Generating speech...", "Speaking...").
     *
     * @param {string} message - Status message to display
     * @returns {void}
     */
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

    /**
     * Hide TTS status indicator
     *
     * @returns {void}
     */
    hideTTSStatus() {
        const statusEl = document.getElementById('tts-status-indicator');
        if (statusEl) {
            statusEl.style.display = 'none';
        }
    }

    /**
     * Play TTS audio from WebSocket stream (base64 encoded)
     *
     * Receives base64-encoded audio from WebSocket, decodes to binary,
     * creates Blob, and queues for playback.
     *
     * WebSocket Audio Data Format:
     * {
     *   audio_data: string,      // Base64-encoded audio
     *   text: string,            // Original text
     *   audio_format: string,    // Format (e.g., 'wav')
     *   duration: number,        // Duration in seconds
     *   language: string         // Language code
     * }
     *
     * @async
     * @param {Object} audioData - Audio payload from WebSocket
     * @param {string} audioData.audio_data - Base64-encoded audio data
     * @param {string} [audioData.text] - Original text that was synthesized
     * @param {string} [audioData.audio_format] - Audio format
     * @param {number} [audioData.duration] - Audio duration in seconds
     * @param {string} [audioData.language] - Language code
     * @returns {Promise<void>}
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
     *
     * Adds audio to queue and starts playback if not already playing.
     * Queue ensures audio plays in order without overlapping.
     *
     * @param {string} audioUrl - Object URL (blob:) of audio blob
     * @param {Object} [metadata={}] - Audio metadata
     * @param {number} [metadata.volume] - Override volume for this audio (0.0-1.0)
     * @returns {void}
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
     *
     * Recursively processes audio queue, playing each item sequentially.
     * Checks for user interaction before playing (browser autoplay policy).
     *
     * Event Handlers:
     * - onended: Cleanup and play next in queue
     * - onerror: Log error and continue to next in queue
     *
     * @async
     * @returns {Promise<void>}
     */
    async playNextInQueue() {
        if (this.audioQueue.length === 0) {
            this.isPlayingQueue = false;
            console.log('Audio queue empty');

            // Resume audio recording now that TTS playback is complete
            if (typeof this.resumeRecording === 'function') {
                this.resumeRecording();
            }

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
     *
     * Stops any active TTS playback, clears the audio queue,
     * and resets all playback state.
     *
     * @returns {void}
     */
    stopAudio() {
        // Stop current playback
        if (this.playingAudio) {
            this.playingAudio.pause();
            this.playingAudio = null;
            console.log('Audio stopped');
        }

        // Clear TTS state
        this.isTTSActive = false;
        this.hideTTSStatus();

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
     *
     * Enumerates media devices and checks for audio input devices.
     *
     * @async
     * @returns {Promise<boolean>} True if at least one microphone is found
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
     *
     * Queries /api/audio/voice_status to check if backend voice mode is enabled.
     *
     * @async
     * @returns {Promise<boolean>} True if voice mode is enabled on backend
     */
    async getVoiceStatus() {
        try {
            const response = window.backendConnector
                ? await window.backendConnector.fetch('/api/audio/voice_status', {
                    headers: {
                        'X-Session-ID': window.sessionId || ''
                    }
                })
                : await fetch('/api/audio/voice_status', {
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
     *
     * Displays status via global showToast() function if available,
     * otherwise logs to console.
     *
     * @param {string} message - Status message to display
     * @param {string} [type='info'] - Message type: 'info', 'success', 'warning', 'error'
     * @returns {void}
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
     *
     * Displays error via global showToast() function if available,
     * otherwise logs to console.
     *
     * @param {string} message - Error message to display
     * @returns {void}
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
     *
     * Stops all media streams and audio playback.
     * Should be called when AudioManager is being destroyed.
     *
     * @returns {void}
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

    /**
     * Set audio volume
     *
     * Updates volume and applies to currently playing audio if any.
     *
     * @param {number} volumePercent - Volume percentage (0-100)
     * @returns {void}
     */
    setVolume(volumePercent) {
        this.volume = Math.max(0, Math.min(1, volumePercent / 100));
        console.log('TTS volume set to', this.volume);

        if (this.playingAudio) {
            this.playingAudio.volume = this.volume;
        }
    }

    /**
     * Get current volume
     *
     * @returns {number} Volume percentage (0-100)
     */
    getVolume() {
        return this.volume * 100;
    }

    /**
     * Play a notification sound
     *
     * Generates a simple tone using Web Audio API for user feedback.
     * Uses different frequencies for different notification types.
     *
     * @param {string} type - Notification type: 'listening' (ascending tone) or 'processing' (descending tone)
     * @returns {void}
     */
    /**
     * Initialize audio context for notification sounds.
     * Must be called during user gesture to satisfy browser autoplay policies.
     */
    initNotificationAudio() {
        try {
            if (!this.notificationAudioContext) {
                this.notificationAudioContext = new (window.AudioContext || window.webkitAudioContext)();
                console.log('[AudioManager] Notification audio context initialized');
            }
            // Resume if suspended (browser autoplay policy)
            if (this.notificationAudioContext.state === 'suspended') {
                this.notificationAudioContext.resume();
            }
        } catch (error) {
            console.error('[AudioManager] Failed to initialize notification audio:', error);
        }
    }

    playNotificationSound(type = 'listening') {
        try {
            // Ensure audio context is initialized
            if (!this.notificationAudioContext) {
                console.warn('[AudioManager] Audio context not initialized - call initNotificationAudio() first');
                return;
            }

            // Resume if suspended (browser autoplay policy)
            if (this.notificationAudioContext.state === 'suspended') {
                this.notificationAudioContext.resume();
            }

            const audioContext = this.notificationAudioContext;

            // Create oscillator for tone generation
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            // Connect nodes
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            // Configure sound based on type
            if (type === 'listening') {
                // Ascending two-tone chime for "listening" state
                oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
                oscillator.frequency.setValueAtTime(1000, audioContext.currentTime + 0.1);
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
            } else if (type === 'processing') {
                // Descending two-tone chime for "processing" state
                oscillator.frequency.setValueAtTime(1000, audioContext.currentTime);
                oscillator.frequency.setValueAtTime(800, audioContext.currentTime + 0.1);
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
            }

            // Play the tone
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.2);

            console.log(`[AudioManager] Played ${type} notification sound`);

        } catch (error) {
            console.error('[AudioManager] Failed to play notification sound:', error);
            // Non-critical error - don't throw
        }
    }
}

// Expose AudioManager globally
window.AudioManager = AudioManager;

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.audioManager) {
        window.audioManager.cleanup();
    }
});
