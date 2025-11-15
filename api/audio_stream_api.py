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

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio chunk with VAD logic (matches CLI record_audio behavior).

        Returns:
            Complete audio recording when silence detected, None otherwise
        """
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
            if msg.type == aiohttp.WSMsgType.BINARY:
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

                                # Send transcription back to client
                                await ws.send_json({
                                    "type": "transcription",
                                    "text": text.strip(),
                                    "language": language
                                })
                                logger.debug(f"Sent transcription: {text[:100]}...")

                                # Reset handler for next recording
                                handler = AudioStreamHandler()
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

            elif msg.type == aiohttp.WSMsgType.TEXT:
                # Handle control messages if needed
                try:
                    data = msg.json()
                    if data.get("type") == "reset":
                        handler = AudioStreamHandler()
                        logger.info("Audio stream handler reset")
                except Exception as e:
                    logger.error(f"Error handling text message: {e}")

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
