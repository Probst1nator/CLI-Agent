"""
CLI-Agent Remote Host Server.

This module provides a Flask-based server that exposes endpoints for various
CLI-Agent services that can be run remotely, including wake word detection,
speech processing, and potentially other services in the future.
"""

import logging
import os
from flask import Flask, request, jsonify
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Import services conditionally to allow for partial functionality
try:
    from .services.wake_word_service import WakeWordService
    wake_word_service = WakeWordService()
    WAKE_WORD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Wake word detection service unavailable: {e}")
    wake_word_service = None
    WAKE_WORD_AVAILABLE = False

# Import Whisper transcription service
try:
    from .services.whisper_service import WhisperService
    whisper_service = WhisperService()
    WHISPER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Whisper transcription service unavailable: {e}")
    whisper_service = None
    WHISPER_AVAILABLE = False

# Health status tracking
SERVICES_STATUS = {
    "wake_word": WAKE_WORD_AVAILABLE,
    "whisper": WHISPER_AVAILABLE,
    # Add more services here as they become available
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with status of available services."""
    return jsonify({
        "status": "healthy",
        "services": SERVICES_STATUS,
        "timestamp": time.time()
    })

@app.route('/services', methods=['GET'])
def list_services():
    """List available services and their status."""
    services_docs = {
        "wake_word": {
            "endpoints": [
                {
                    "path": "/wake_word/detect",
                    "method": "POST",
                    "description": "Detect wake word in audio data"
                }
            ]
        }
    }
    
    if WHISPER_AVAILABLE:
        services_docs["whisper"] = {
            "endpoints": [
                {
                    "path": "/whisper/transcribe",
                    "method": "POST",
                    "description": "Transcribe audio using Whisper models"
                },
                {
                    "path": "/whisper/models",
                    "method": "GET",
                    "description": "Get available Whisper models"
                }
            ]
        }
    
    return jsonify({
        "services": SERVICES_STATUS,
        "docs": services_docs
    })

# Wake word detection endpoints
@app.route('/wake_word/detect', methods=['POST'])
def detect_wake_word():
    """
    Endpoint to detect wake word from audio data.
    
    Expects JSON with:
    {
        "audio_data": "<base64 encoded audio data>",
        "sample_rate": 16000  # Optional, defaults to 16000
    }
    
    Returns:
    {
        "detected": true/false,
        "wake_word": "<detected wake word or null>"
    }
    """
    if not WAKE_WORD_AVAILABLE:
        return jsonify({"error": "Wake word detection service is not available"}), 503
        
    try:
        # Get audio data from request
        data = request.get_json()
        
        if not data or "audio_data" not in data:
            return jsonify({"error": "Missing audio_data"}), 400
            
        audio_data_b64 = data["audio_data"]
        sample_rate = data.get("sample_rate", 16000)
        
        # Process audio data
        result = wake_word_service.process_audio(audio_data_b64, sample_rate)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in wake word detection endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Whisper transcription endpoints
@app.route('/whisper/models', methods=['GET'])
def list_whisper_models():
    """Get information about available Whisper models."""
    if not WHISPER_AVAILABLE:
        return jsonify({"error": "Whisper transcription service is not available"}), 503
        
    try:
        models = whisper_service.get_available_models()
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Error listing Whisper models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/whisper/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio using Whisper.
    
    Expects JSON with:
    {
        "audio_data": "<base64 encoded audio data>",
        "sample_rate": 16000,
        "model": "large-v2"  # Optional, defaults to large-v2
    }
    
    Returns:
    {
        "success": true/false,
        "transcription": "<transcribed text>",
        "language": "<detected language>",
        "processing_time": <seconds>,
        "model": "<model used>"
    }
    """
    if not WHISPER_AVAILABLE:
        return jsonify({"error": "Whisper transcription service is not available"}), 503
        
    try:
        # Get audio data from request
        data = request.get_json()
        
        if not data or "audio_data" not in data:
            return jsonify({"error": "Missing audio_data"}), 400
            
        audio_data_b64 = data["audio_data"]
        sample_rate = data.get("sample_rate", 16000)
        model_name = data.get("model", "large-v2")
        
        # Process audio data for transcription
        result = whisper_service.transcribe_audio(audio_data_b64, sample_rate, model_name)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in transcription endpoint: {e}")
        return jsonify({"error": str(e)}), 500

def create_app() -> Flask:
    """Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application
    """
    return app

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Start server
    logger.info(f"Starting CLI-Agent Remote Host server on {host}:{port}")
    app.run(host=host, port=port) 