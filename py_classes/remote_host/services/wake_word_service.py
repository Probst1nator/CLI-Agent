"""
Wake Word Detection Service.

This module provides a service for detecting wake words using the Vosk speech recognition model.
"""

import logging
import os
import base64
import json
from typing import Dict, Any, List
import numpy as np
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)

class WakeWordService:
    """
    Service for detecting wake words in audio data.
    
    This service provides methods for:
    - Loading Vosk models for speech recognition
    - Processing audio data to detect wake words
    """
    
    def __init__(self):
        """Initialize the WakeWordService with a small speech recognition model."""
        try:
            # Load the Vosk model for English
            self.model = Model(lang="en-us")
            self.ready = True
            logger.info("Wake word service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize wake word service: {e}")
            self.model = None
            self.ready = False
            
    def is_ready(self) -> bool:
        """
        Check if the service is ready for use.
        
        Returns:
            bool: True if the service is ready
        """
        return self.ready and self.model is not None
            
    def process_audio(self, audio_data_b64: str, sample_rate: int) -> Dict[str, Any]:
        """
        Process audio data to detect wake words.
        
        Args:
            audio_data_b64: Base64 encoded audio data
            sample_rate: Sample rate of the audio data
            
        Returns:
            Dict: Detection results
        """
        if not self.is_ready():
            return {
                "detected": False,
                "error": "Wake word service is not ready"
            }
            
        try:
            # Define the wake words to detect
            wake_words: List[str] = [
                # Single words
                "computer", "nova",
                # Full phrases 
                "hey computer", "hey nova",
                "ok computer", "ok nova",
                "okay computer", "okay nova"
            ]
            
            # Create recognizer with provided sample rate
            rec = KaldiRecognizer(self.model, sample_rate)
            rec.SetWords(True)
            
            # Decode audio data
            audio_bytes = base64.b64decode(audio_data_b64)
            
            # Process the entire audio segment
            if rec.AcceptWaveform(audio_bytes):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower().strip()
                
                if text:
                    logger.info(f"Detected speech: '{text}'")
                    
                    # Check for wake words
                    if text.count(" ") <= 4:  # Skip if too many words
                        for wake_word in wake_words:
                            if wake_word in text:
                                return {
                                    "detected": True,
                                    "wake_word": wake_word,
                                    "text": text
                                }
            
            # Check final result if nothing detected yet
            final_result = json.loads(rec.FinalResult())
            text = final_result.get("text", "").lower().strip()
            
            if text:
                logger.info(f"Final detected speech: '{text}'")
                
                # Check for wake words
                if text.count(" ") <= 4:  # Skip if too many words
                    for wake_word in wake_words:
                        if wake_word in text:
                            return {
                                "detected": True,
                                "wake_word": wake_word,
                                "text": text
                            }
            
            # No wake word detected
            return {
                "detected": False,
                "text": text if text else None
            }
                
        except Exception as e:
            logger.error(f"Error processing audio for wake word detection: {e}")
            return {
                "detected": False,
                "error": str(e)
            } 