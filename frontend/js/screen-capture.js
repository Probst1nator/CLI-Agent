/**
 * Screen Capture Manager
 *
 * Handles browser-based screen/window/tab capture using the Screen Capture API
 * with support for both Chromium and Firefox browsers.
 *
 * Features:
 * - Capture entire screen, specific windows, or browser tabs
 * - Cross-browser support (Chrome, Firefox, Edge, etc.)
 * - Permission handling and error management
 * - Screenshot conversion to base64 for upload
 */

class ScreenCaptureManager {
    constructor() {
        this.captureStream = null;
        this.videoElement = null;
        this.isCapturing = false;

        // Initialize video element for capture
        this.initializeVideoElement();
    }

    /**
     * Initialize hidden video element for capturing frames
     */
    initializeVideoElement() {
        this.videoElement = document.createElement('video');
        this.videoElement.style.display = 'none';
        this.videoElement.autoplay = true;
        this.videoElement.muted = true;
        document.body.appendChild(this.videoElement);
    }

    /**
     * Check if Screen Capture API is supported
     * @returns {boolean} True if supported
     */
    isSupported() {
        // Check for getDisplayMedia support (standard API)
        if (navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) {
            return true;
        }

        // Fallback check for older Firefox versions
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            return true;
        }

        return false;
    }

    /**
     * Request screen capture from user
     * @param {Object} options - Capture options
     * @param {boolean} options.audio - Include audio (default: false)
     * @param {string} options.preferCurrentTab - Prefer current tab (default: false)
     * @returns {Promise<MediaStream>} Screen capture stream
     */
    async requestScreenCapture(options = {}) {
        if (!this.isSupported()) {
            throw new Error('Screen Capture API is not supported in this browser');
        }

        const captureOptions = {
            video: {
                cursor: 'always', // Always show cursor in capture
                displaySurface: options.preferCurrentTab ? 'browser' : 'monitor', // Hint for what to capture
            },
            audio: options.audio || false,
            // Firefox-specific options
            ...(this.isFirefox() && {
                mediaSource: 'screen' // Firefox requires this
            })
        };

        try {
            console.log('🎥 Requesting screen capture with options:', captureOptions);

            // Request display media (works in Chrome, Firefox, Edge)
            this.captureStream = await navigator.mediaDevices.getDisplayMedia(captureOptions);

            console.log('✅ Screen capture stream acquired');
            return this.captureStream;

        } catch (error) {
            console.error('❌ Screen capture failed:', error);

            if (error.name === 'NotAllowedError') {
                throw new Error('Screen capture permission denied by user');
            } else if (error.name === 'NotFoundError') {
                throw new Error('No screen capture source available');
            } else if (error.name === 'NotSupportedError') {
                throw new Error('Screen capture not supported in this browser');
            } else {
                throw new Error(`Screen capture error: ${error.message}`);
            }
        }
    }

    /**
     * Capture a single screenshot from the stream
     * @param {MediaStream} stream - The capture stream
     * @returns {Promise<string>} Base64 encoded screenshot
     */
    async captureScreenshot(stream) {
        if (!stream || !stream.active) {
            throw new Error('No active capture stream');
        }

        return new Promise((resolve, reject) => {
            try {
                // Set video source to stream
                this.videoElement.srcObject = stream;

                // Wait for video to load and play
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play().then(() => {
                        // Give it a moment to render first frame
                        setTimeout(() => {
                            try {
                                // Create canvas for screenshot
                                const canvas = document.createElement('canvas');
                                canvas.width = this.videoElement.videoWidth;
                                canvas.height = this.videoElement.videoHeight;

                                // Draw video frame to canvas
                                const ctx = canvas.getContext('2d');
                                ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);

                                // Convert to base64
                                const screenshot = canvas.toDataURL('image/png');

                                console.log(`📸 Screenshot captured: ${canvas.width}x${canvas.height}`);
                                resolve(screenshot);

                            } catch (error) {
                                reject(new Error(`Failed to capture frame: ${error.message}`));
                            }
                        }, 100); // Small delay to ensure frame is rendered
                    }).catch(reject);
                };

                this.videoElement.onerror = (error) => {
                    reject(new Error(`Video playback error: ${error}`));
                };

            } catch (error) {
                reject(new Error(`Screenshot capture error: ${error.message}`));
            }
        });
    }

    /**
     * Stop the capture stream and cleanup
     */
    stopCapture() {
        if (this.captureStream) {
            this.captureStream.getTracks().forEach(track => track.stop());
            this.captureStream = null;
            console.log('🛑 Screen capture stream stopped');
        }

        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }

        this.isCapturing = false;
    }

    /**
     * Capture screen and return screenshot
     * @param {Object} options - Capture options
     * @returns {Promise<string>} Base64 encoded screenshot
     */
    async capture(options = {}) {
        if (this.isCapturing) {
            throw new Error('Capture already in progress');
        }

        this.isCapturing = true;

        try {
            // Request screen capture
            const stream = await this.requestScreenCapture(options);

            // Capture screenshot
            const screenshot = await this.captureScreenshot(stream);

            // Stop capture
            this.stopCapture();

            return screenshot;

        } catch (error) {
            this.stopCapture();
            throw error;
        }
    }

    /**
     * Start continuous screen capture (keeps stream alive)
     * @param {Object} options - Capture options
     * @returns {Promise<MediaStream>} Screen capture stream
     */
    async startContinuousCapture(options = {}) {
        if (this.captureStream && this.captureStream.active) {
            console.log('Continuous capture already active');
            return this.captureStream;
        }

        try {
            this.captureStream = await this.requestScreenCapture(options);
            this.videoElement.srcObject = this.captureStream;

            await new Promise((resolve, reject) => {
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play().then(resolve).catch(reject);
                };
                this.videoElement.onerror = reject;
            });

            console.log('🟢 Continuous screen capture started');
            this.isCapturing = true;
            return this.captureStream;

        } catch (error) {
            this.stopCapture();
            throw error;
        }
    }

    /**
     * Capture a screenshot from existing continuous stream
     * @returns {Promise<string>} Base64 encoded screenshot
     */
    async captureFromStream() {
        if (!this.captureStream || !this.captureStream.active) {
            throw new Error('No active continuous capture stream. Call startContinuousCapture() first.');
        }

        try {
            const canvas = document.createElement('canvas');
            canvas.width = this.videoElement.videoWidth;
            canvas.height = this.videoElement.videoHeight;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);

            const screenshot = canvas.toDataURL('image/png');
            console.log(`📸 Screenshot captured from stream: ${canvas.width}x${canvas.height}`);
            return screenshot;

        } catch (error) {
            throw new Error(`Failed to capture from stream: ${error.message}`);
        }
    }

    /**
     * Check if browser is Firefox
     * @returns {boolean} True if Firefox
     */
    isFirefox() {
        return navigator.userAgent.toLowerCase().indexOf('firefox') > -1;
    }

    /**
     * Check if browser is Chrome/Chromium
     * @returns {boolean} True if Chrome/Chromium
     */
    isChromium() {
        return /Chrome|Chromium|Edg/.test(navigator.userAgent);
    }

    /**
     * Get browser-specific capture instructions
     * @returns {string} Instructions for user
     */
    getCaptureInstructions() {
        if (this.isFirefox()) {
            return 'Firefox: Select "Entire Screen", "Window", or "Application Window" from the dialog.';
        } else if (this.isChromium()) {
            return 'Chrome: Select "Entire Screen", "Window", or "Chrome Tab" from the dialog.';
        } else {
            return 'Select the screen, window, or tab you want to capture.';
        }
    }
}

// Export for use in main.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ScreenCaptureManager;
}
