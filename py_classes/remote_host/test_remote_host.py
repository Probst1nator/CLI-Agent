#!/usr/bin/env python
"""
Test the CLI-Agent Remote Host functionality.

This script demonstrates how to use the PyAiHost with remote host services.
"""

import os
import sys
import time
import argparse
from typing import Optional

# Add parent directory to path to allow importing from py_classes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py_classes.ai_providers.cls_pyaihost_interface import PyAiHost
import soundfile as sf

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test CLI-Agent Remote Host functionality")
    
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use remote host for services"
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        help="URL of the remote host server (default: use CLI_AGENT_REMOTE_HOST env var)"
    )
    
    parser.add_argument(
        "--test",
        choices=["wake-word", "all"],
        default="wake-word",
        help="Which test to run (default: wake-word)"
    )
    
    return parser.parse_args()

def test_wake_word(use_remote: bool = False, server_url: Optional[str] = None):
    """Test wake word detection."""
    print("\n=== Testing Wake Word Detection ===")
    
    if use_remote:
        print("Using REMOTE wake word detection")
        
        # Set server URL if provided
        if server_url:
            os.environ["CLI_AGENT_REMOTE_HOST"] = server_url
            
        # Initialize the remote host client
        PyAiHost.initialize_remote_host_client()
        
        # Check if wake word service is available
        if not PyAiHost.remote_host_client.is_service_available("wake_word"):
            print("ERROR: Wake word service not available on remote host!")
            return False
    else:
        print("Using LOCAL wake word detection")
    
    print("Waiting for wake word...")
    wake_word = PyAiHost.wait_for_wake_word(use_remote=use_remote)
    
    if wake_word:
        print(f"Wake word detected: {wake_word}")
        return True
    else:
        print("No wake word detected or detection failed.")
        return False

def test_full_voice_interaction(use_remote: bool = False, server_url: Optional[str] = None):
    """Test full voice interaction flow."""
    # Set server URL if provided
    if server_url:
        os.environ["CLI_AGENT_REMOTE_HOST"] = server_url
    
    print("\n=== Starting Full Voice Interaction Test ===")
    
    if use_remote:
        print("Using REMOTE services")
        # Initialize the remote host client
        PyAiHost.initialize_remote_host_client()
    else:
        print("Using LOCAL services")
    
    # Test wake word detection
    if not test_wake_word(use_remote=use_remote):
        return
    
    print("You will hear an ascending tone. After the tone, please speak...")
    time.sleep(1)
    PyAiHost.play_notification()
    
    # Record audio
    audio_file = "recorded_speech.wav"
    audio_data, sample_rate = PyAiHost.record_audio(
        use_wake_word=False,  # Already detected wake word
        use_remote=use_remote
    )
    
    if len(audio_data) == 0:
        print("No audio recorded.")
        return
    
    sf.write(audio_file, audio_data, sample_rate)
    
    # Transcribe the recorded audio
    transcribed_text, detected_lang = PyAiHost.transcribe_audio(
        audio_path=audio_file,
        whisper_model_key='small'
    )
    
    print(f"\nTranscribed Text: {transcribed_text}")
    print(f"Detected Language: {detected_lang}")
    
    if transcribed_text:
        print("\n=== Testing Speech Synthesis ===")
        
        print("\nPlaying synthesis...")
        PyAiHost.text_to_speech(
            text=transcribed_text,
            voice='af_heart',
            play=True
        )
    else:
        print("\nNo text was transcribed, skipping speech synthesis.")

def main():
    """Run the test."""
    # Parse arguments
    args = parse_args()
    
    # Run the requested test
    if args.test == "wake-word":
        test_wake_word(use_remote=args.remote, server_url=args.server_url)
    elif args.test == "all":
        test_full_voice_interaction(use_remote=args.remote, server_url=args.server_url)

if __name__ == "__main__":
    main() 