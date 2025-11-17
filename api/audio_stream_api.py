"""
WebSocket audio streaming API for continuous voice input.

Allows browser to stream audio chunks continuously while backend
handles VAD (Voice Activity Detection) and transcription, matching
CLI voice input behavior.
"""

import logging
import time
from typing import Any, Optional

import numpy as np

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    from termcolor import colored
except ImportError:
    # Fallback if termcolor not available
    def colored(text, color):
        return text

logger = logging.getLogger(__name__)

# VAD parameters (matching CLI mode)
DEFAULT_THRESHOLD = 0.02
DEFAULT_SILENCE_DURATION = 2.0
DEFAULT_MAX_DURATION = 60.0
SAMPLE_RATE = 16000


class AudioStreamHandler:
    """Handles continuous audio streaming with backend VAD"""

    def __init__(self):
        self.audio_buffer = []
        self.recording = False
        self.silent_chunks = 0
        self.start_time = None
        self.threshold = DEFAULT_THRESHOLD
        self.silence_duration = DEFAULT_SILENCE_DURATION
        self.max_duration = DEFAULT_MAX_DURATION
        self.num_silent_chunks_to_stop = int(self.silence_duration * 10)
        self.printed_listening = False  # Track if we've printed the listening message
        self.paused = False  # Pause recording during agent response/TTS playback
        self.resume_time = None  # Track when recording was resumed (for grace period)
        self.grace_period = 0.4  # Ignore audio for 400ms after resume (acoustic echo clearance)

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio chunk with VAD logic (matches CLI record_audio behavior).

        Returns:
            Complete audio recording when silence detected, None otherwise
        """
        # Skip processing if paused (agent responding or TTS playing)
        if self.paused:
            return None

        # Grace period: ignore audio immediately after resume to avoid capturing acoustic echo
        if self.resume_time and (time.time() - self.resume_time) < self.grace_period:
            return None  # Silently ignore audio during grace period

        # Clear resume time after grace period expires
        if self.resume_time and (time.time() - self.resume_time) >= self.grace_period:
            self.resume_time = None

        # Print initial listening message (like CLI mode)
        if not self.printed_listening:
            print(f"<{colored('Recording', 'red')}> Listening...")
            self.printed_listening = True

        # Calculate amplitude (mean absolute value)
        is_loud = np.abs(audio_chunk).mean() > self.threshold

        # Check max duration
        if self.start_time and (time.time() - self.start_time) > self.max_duration:
            print("\n<Recording> Reached maximum recording time.")
            return self._finalize_recording()

        if self.recording:
            # Currently recording
            self.audio_buffer.append(audio_chunk)

            if is_loud:
                self.silent_chunks = 0
            else:
                self.silent_chunks += 1

            # Check if silence threshold reached
            if self.silent_chunks > self.num_silent_chunks_to_stop:
                print("\n<Recording> Detected end of speech.")
                return self._finalize_recording()

        elif is_loud:
            # Speech detected, start recording (like CLI mode)
            print("<Recording> Speech detected, starting...", end="", flush=True)
            self.recording = True
            self.start_time = time.time()
            self.audio_buffer = [audio_chunk]
            self.silent_chunks = 0

        return None

    def _finalize_recording(self) -> Optional[np.ndarray]:
        """Finalize recording and return audio data"""
        if not self.audio_buffer:
            logger.warning("No audio in buffer")
            return None

        # Concatenate all chunks
        audio_data = np.concatenate(self.audio_buffer, axis=0)

        # Reset state
        self.audio_buffer = []
        self.recording = False
        self.silent_chunks = 0
        self.start_time = None

        logger.info(f"Finalized recording: {len(audio_data)} samples, {len(audio_data)/SAMPLE_RATE:.2f}s duration")
        return audio_data

    def pause(self):
        """Pause audio processing during agent response/TTS playback"""
        self.paused = True
        print(f"<{colored('Recording', 'yellow')}> Paused (agent responding)")
        logger.debug("Audio stream paused")

    def resume(self):
        """Resume audio processing after agent response/TTS playback"""
        self.paused = False
        self.printed_listening = False  # Allow "Listening..." to print again

        # Reset VAD state to avoid capturing acoustic echo/noise from TTS/beep
        self.recording = False
        self.audio_buffer = []
        self.silent_chunks = 0
        self.start_time = None

        # Set grace period to ignore immediate acoustic echo
        self.resume_time = time.time()

        print(f"<{colored('Recording', 'green')}> Resumed")
        logger.debug(f"Audio stream resumed - VAD state reset, {int(self.grace_period * 1000)}ms grace period active")


async def websocket_audio_stream(request: Any) -> Any:
    """
    WebSocket endpoint for continuous audio streaming.

    Protocol:
    - Client sends binary audio chunks (Float32 PCM, 16kHz, mono)
    - Server applies VAD and buffers audio
    - Server transcribes when silence detected
    - Server sends back: {"type": "transcription", "text": "...", "language": "..."}
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    logger.info("WebSocket audio stream connected")

    handler = AudioStreamHandler()

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                # Handle control messages (pause/resume/reset)
                try:
                    import json
                    data = json.loads(msg.data)
                    action = data.get('action')
                    msg_type = data.get('type')  # Legacy support

                    if action == 'pause':
                        handler.pause()
                    elif action == 'resume':
                        handler.resume()
                    elif action == 'reset' or msg_type == 'reset':
                        handler = AudioStreamHandler()
                        logger.info("Audio stream handler reset")
                    else:
                        logger.warning(f"Unknown audio stream message: action={action}, type={msg_type}")

                except Exception as e:
                    logger.error(f"Error handling audio stream control message: {e}")

            elif msg.type == aiohttp.WSMsgType.BINARY:
                try:
                    # Convert binary data to numpy array
                    # Browser sends Float32Array as binary
                    audio_chunk = np.frombuffer(msg.data, dtype=np.float32)

                    if len(audio_chunk) == 0:
                        continue

                    # Process chunk with VAD
                    complete_audio = handler.process_chunk(audio_chunk)

                    if complete_audio is not None:
                        # Silence detected, transcribe
                        logger.debug(f"Transcribing {len(complete_audio)} samples")

                        # Notify frontend that processing has started
                        await ws.send_json({
                            "type": "processing",
                            "message": "Transcribing audio..."
                        })

                        try:
                            from shared.utils_audio import (
                                initialize_faster_whisper_model,
                                transcribe_audio,
                            )

                            # Initialize Faster-Whisper if not already done
                            initialize_faster_whisper_model("small")

                            # Transcribe (this will print the CLI-style logs)
                            text, language = transcribe_audio(
                                complete_audio,
                                SAMPLE_RATE,
                                use_faster_whisper=True
                            )

                            if text and text.strip():
                                # Print user reply message (like CLI mode)
                                print(f"🎤 User replied: \"{text.strip()}\"")

                                # Pause immediately to avoid capturing agent response/TTS/beep
                                # Frontend will resume after TTS + beep finish (or immediately if no TTS)
                                handler.pause()

                                # Send transcription back to client
                                await ws.send_json({
                                    "type": "transcription",
                                    "text": text.strip(),
                                    "language": language
                                })
                                logger.debug(f"Sent transcription: {text[:100]}...")
                            else:
                                # Empty transcription
                                print(f"<{colored('Warning', 'yellow')}> No speech detected.")
                                await ws.send_json({
                                    "type": "transcription",
                                    "text": "",
                                    "language": "unknown",
                                    "error": "No speech detected"
                                })
                                # Reset handler for next recording
                                handler = AudioStreamHandler()

                        except Exception as e:
                            logger.error(f"Transcription failed: {e}", exc_info=True)
                            await ws.send_json({
                                "type": "error",
                                "error": str(e)
                            })
                            # Reset handler even on error
                            handler = AudioStreamHandler()

                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    await ws.send_json({
                        "type": "error",
                        "error": f"Audio processing failed: {str(e)}"
                    })

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")

    except Exception as e:
        logger.error(f"WebSocket audio stream error: {e}", exc_info=True)

    finally:
        logger.info("WebSocket audio stream closed")

    return ws


def setup_routes(app: Any):
    """Register WebSocket audio streaming route"""
    app.router.add_get('/api/audio/stream', websocket_audio_stream)
    logger.info("Audio streaming WebSocket route registered")
