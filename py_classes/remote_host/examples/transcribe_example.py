#!/usr/bin/env python3
"""
Example script to demonstrate using the remote Whisper transcription service.

This script:
1. Records audio from the microphone
2. Sends it to the remote host for transcription using Whisper
3. Prints the transcription results
"""

import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Tuple

# Add parent directory to path to import RemoteHostClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from py_classes.remote_host.cls_remote_host_client import RemoteHostClient

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Record audio from the microphone.
    
    Args:
        duration: Duration to record in seconds
        sample_rate: Sample rate for recording
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    print(f"Recording for {duration} seconds...")
    
    # Record audio
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        blocking=True
    )
    
    print("Recording complete!")
    
    return audio_data, sample_rate

def list_available_models(client: RemoteHostClient) -> None:
    """List available Whisper models on the remote host."""
    result = client.list_whisper_models()
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
        
    models = result.get("models", {})
    if not models:
        print("No models information available")
        return
        
    print("\nAvailable Whisper Models:")
    print("-----------------------")
    for name, info in models.items():
        loaded_status = "✓ (loaded)" if info.get("loaded", False) else "✗ (not loaded)"
        print(f"{name:10} - {info.get('params', ''):8} - {info.get('description', ''):40} - {loaded_status}")
    print()

def main():
    # Initialize the remote host client
    client = RemoteHostClient()
    
    # Check if server is available
    health = client.check_server_health()
    if not health:
        print("Error: Could not connect to remote host server")
        return
        
    print(f"Connected to remote host server: {health.get('status', 'unknown')}")
    
    # Check if Whisper service is available
    if not client.is_service_available("whisper"):
        print("Error: Whisper service is not available on the remote host")
        return
        
    # List available models
    list_available_models(client)
    
    # Ask user to select a model
    model = input("Enter model to use (default: large-v2): ").strip()
    if not model:
        model = "large-v2"
        
    # Record audio
    audio_data, sample_rate = record_audio(duration=5.0)
    
    # Save audio for debugging
    sf.write('recorded_audio.wav', audio_data, sample_rate)
    print("Saved recording to recorded_audio.wav")
    
    # Send to server for transcription
    print(f"Sending audio to server for transcription with model '{model}'...")
    result = client.transcribe_audio(audio_data, sample_rate, model)
    
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error during transcription')}")
        return
        
    # Print results
    print("\nTranscription Results:")
    print("-----------------------")
    print(f"Transcription: {result.get('transcription', '')}")
    print(f"Language: {result.get('language', 'unknown')}")
    print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
    print(f"Model Used: {result.get('model', 'unknown')}")
    
    # Print segments if available
    segments = result.get("segments", [])
    if segments:
        print("\nDetailed Segments:")
        for i, segment in enumerate(segments):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '')
            print(f"[{start:.1f}s - {end:.1f}s]: {text}")

if __name__ == "__main__":
    main() 