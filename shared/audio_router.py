"""
Audio routing module for CLI-Agent.

Routes TTS audio playback between:
- Python native playback (CLI mode)
- WebSocket streaming to frontend (GUI/API mode)

This allows seamless audio playback in both local and web-based interfaces.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def route_tts_audio(
    text: str,
    audio_data: Optional[bytes] = None,
    voice: str = "default",
    speed: float = 1.0,
    volume: float = 0.8,
    language: str = "auto",
    **kwargs: Any
) -> bool:
    """
    Route TTS audio to appropriate playback mechanism based on current mode.

    In GUI/API mode: Streams audio through WebSocket to frontend
    In CLI mode: Plays audio natively through Python audio libraries

    Args:
        text: Text that was converted to speech
        audio_data: Pre-generated audio bytes (WAV format)
        voice: Voice model used
        speed: Speech speed
        volume: Audio volume
        language: Language code
        **kwargs: Additional parameters passed to TTS engine

    Returns:
        bool: True if audio was successfully routed, False otherwise
    """
    from core.globals import g

    # Check if we're in GUI/API mode
    gui_mode = getattr(g, 'GUI_MODE', False)

    if gui_mode:
        # Route through WebSocket to frontend
        return await _stream_to_frontend(text, audio_data, language, speed, volume)
    else:
        # Use native Python audio playback
        return await _play_native(text, audio_data, voice, speed, volume, language, **kwargs)


async def _stream_to_frontend(
    text: str,
    audio_data: Optional[bytes],
    language: str,
    speed: float,
    volume: float
) -> bool:
    """
    Stream TTS audio to frontend via WebSocket.

    Args:
        text: Original text
        audio_data: WAV audio bytes
        language: Language code
        speed: Speech speed
        volume: Volume level

    Returns:
        bool: Success status
    """
    try:
        from tools.visualresponse import visualresponse

        if not audio_data:
            logger.warning("No audio data provided for frontend streaming")
            return False

        # Get active WebSocket session
        session = visualresponse._get_session()
        if not session or not session.is_healthy():
            logger.warning("No active WebSocket session for audio streaming")
            return False

        # Encode audio as base64 for WebSocket transmission
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Calculate approximate duration (rough estimate for WAV)
        # WAV format: 44100 Hz * 2 bytes * 1 channel = 88200 bytes per second
        duration_seconds = len(audio_data) / 88200 if len(audio_data) > 44 else 0

        # Send audio payload via WebSocket
        audio_payload = {
            "id": f"tts-{int(asyncio.get_event_loop().time() * 1000000)}",
            "action": "audio_tts",
            "type": "audio",
            "text": text[:200],  # Truncate for metadata
            "audio_data": audio_base64,
            "audio_format": "wav",
            "language": language,
            "speed": speed,
            "volume": volume,
            "duration": duration_seconds,
            "timestamp": asyncio.get_event_loop().time()
        }

        session.send_update(audio_payload)
        logger.debug(f"Streamed {len(audio_data)} bytes of TTS audio to frontend")
        return True

    except Exception as e:
        logger.error(f"Failed to stream audio to frontend: {e}")
        return False


async def _play_native(
    text: str,
    audio_data: Optional[bytes],
    voice: str,
    speed: float,
    volume: float,
    language: str,
    **kwargs: Any
) -> bool:
    """
    Play audio using native Python audio libraries.

    Falls back to generating audio if not provided.

    Args:
        text: Text to speak
        audio_data: Pre-generated audio bytes (optional)
        voice: Voice model
        speed: Speech speed
        volume: Volume level
        language: Language code
        **kwargs: Additional TTS parameters

    Returns:
        bool: Success status
    """
    try:
        from shared.utils_audio import text_to_speech

        if audio_data:
            # Play pre-generated audio
            import tempfile

            # Write audio bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            try:
                # Play the file
                from shared.utils_audio import _play_audio_file_simple
                await asyncio.to_thread(_play_audio_file_simple, tmp_path, volume)
                return True
            finally:
                # Clean up temp file
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
        else:
            # Generate and play audio
            success = await asyncio.to_thread(
                text_to_speech,
                text=text,
                voice=voice,
                speed=speed,
                volume=volume,
                play=True,
                language=language,
                **kwargs
            )
            return success

    except Exception as e:
        logger.error(f"Failed to play native audio: {e}")
        return False


async def generate_tts_audio(
    text: str,
    voice: str = "default",
    speed: float = 1.0,
    volume: float = 0.8,
    language: str = "auto",
    **kwargs: Any
) -> Optional[bytes]:
    """
    Generate TTS audio without playing it.

    Returns audio bytes in WAV format for routing/streaming.

    Args:
        text: Text to convert to speech
        voice: Voice model
        speed: Speech speed
        volume: Volume level
        language: Language code
        **kwargs: Additional TTS parameters

    Returns:
        bytes: WAV audio data, or None if generation failed
    """
    try:
        import tempfile
        from pathlib import Path

        from shared.utils_audio import text_to_speech

        # Generate audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Generate audio (don't play it)
            success = await asyncio.to_thread(
                text_to_speech,
                text=text,
                voice=voice,
                speed=speed,
                volume=volume,
                output_path=tmp_path,
                play=False,
                language=language,
                **kwargs
            )

            if not success:
                logger.warning("TTS generation returned False")
                return None

            # Read generated audio file
            audio_bytes = Path(tmp_path).read_bytes()
            logger.debug(f"Generated {len(audio_bytes)} bytes of TTS audio")
            return audio_bytes

        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Failed to generate TTS audio: {e}")
        return None
