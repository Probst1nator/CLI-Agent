"""
Audio API endpoints for CLI-Agent.

Provides audio recording transcription and text-to-speech endpoints
for browser-based voice interaction.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


async def upload_audio(request: Any) -> Any:
    """POST /api/audio/upload

    Upload audio blob from browser microphone and transcribe it.

    Body: FormData with 'audio' field (audio/webm or audio/wav)
    Returns: {"status": "success", "text": "transcribed text"}
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        # Read multipart data
        reader = await request.multipart()
        audio_data = None
        audio_filename = "recording.webm"

        while True:
            part = await reader.next()
            if part is None:
                break

            if part.name == 'audio':
                audio_data = await part.read()
                audio_filename = part.filename or audio_filename
                break

        if not audio_data:
            return web.json_response({
                'status': 'error',
                'error': 'No audio data provided',
                'code': 'INVALID_REQUEST'
            }, status=400)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_filename).suffix) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            # Load audio file and transcribe
            import soundfile as sf
            from shared.utils_audio import transcribe_audio

            # Load audio from temporary file
            audio_data, sample_rate = sf.read(tmp_path)

            # Transcribe audio (returns tuple of text and language)
            text, detected_language = await asyncio.to_thread(
                transcribe_audio,
                audio_data,
                sample_rate
            )

            if not text or not text.strip():
                return web.json_response({
                    'status': 'error',
                    'error': 'Transcription failed or returned empty text',
                    'code': 'TRANSCRIPTION_FAILED'
                }, status=500)

            logger.info(f"Transcribed audio ({detected_language}): {text[:100]}...")

            return web.json_response({
                'status': 'success',
                'text': text.strip(),
                'language': detected_language
            })

        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

    except Exception as e:
        logger.error(f"Audio upload failed: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'TRANSCRIPTION_FAILED'
        }, status=500)


async def generate_tts(request: Any) -> Any:
    """POST /api/audio/tts

    Generate speech from text using TTS.

    Body: {"text": "...", "language": "en"}
    Returns: audio/wav stream
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        # Parse request body
        try:
            data = await request.json()
        except Exception:
            return web.json_response({
                'status': 'error',
                'error': 'Invalid JSON body',
                'code': 'INVALID_REQUEST'
            }, status=400)

        text = data.get('text', '').strip()
        language = data.get('language', 'en')

        if not text:
            return web.json_response({
                'status': 'error',
                'error': 'Text is required',
                'code': 'INVALID_REQUEST'
            }, status=400)

        logger.info(f"Generating TTS for text: {text[:100]}... (language: {language})")

        # Generate TTS audio
        try:
            from shared.audio_router import generate_tts_audio
            from core.globals import g

            # Use online or local TTS based on mode
            online_mode = getattr(g, 'online_mode', False)

            audio_data = await generate_tts_audio(
                text=text,
                language=language,
                use_piper=not online_mode  # Use Piper for local, otherwise use online TTS
            )

            if not audio_data:
                return web.json_response({
                    'status': 'error',
                    'error': 'TTS generation failed',
                    'code': 'TTS_GENERATION_FAILED'
                }, status=500)

            # Return audio as WAV stream
            return web.Response(
                body=audio_data,
                content_type='audio/wav',
                headers={
                    'Content-Disposition': 'inline; filename="tts.wav"'
                }
            )

        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)
            return web.json_response({
                'status': 'error',
                'error': str(e),
                'code': 'TTS_GENERATION_FAILED'
            }, status=500)

    except Exception as e:
        logger.error(f"TTS request failed: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def get_voice_status(request: Any) -> Any:
    """GET /api/audio/voice_status

    Get current voice mode status.

    Returns: {"status": "success", "voice_mode": true/false}
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        voice_mode = getattr(g, 'voice_mode', False)

        return web.json_response({
            'status': 'success',
            'voice_mode': voice_mode
        })

    except Exception as e:
        logger.error(f"Failed to get voice status: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


def setup_routes(app: Any):
    """Register all audio API routes"""
    app.router.add_post('/api/audio/upload', upload_audio)
    app.router.add_post('/api/audio/tts', generate_tts)
    app.router.add_get('/api/audio/voice_status', get_voice_status)
    logger.info("Audio API routes registered")
