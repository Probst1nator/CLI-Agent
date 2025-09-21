#!/usr/bin/env python3
"""
Quick test of Piper TTS interface to verify functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.providers.cls_piper_interface import get_piper_interface

def test_piper_basic():
    """Test basic Piper TTS functionality."""
    print("🧪 Testing Piper TTS Interface")
    print("=" * 40)

    # Get Piper interface
    piper = get_piper_interface()

    # Check availability
    print(f"📋 Piper available: {piper.is_available()}")

    if not piper.is_available():
        print("❌ Piper TTS not available")
        return False

    # Test model info
    info = piper.get_model_info()
    print(f"📊 Model info: {info}")

    # Test available voices
    voices = piper.get_available_voices()
    print(f"🎤 Available voices: {voices}")

    # Test German speech generation
    print("\n🇩🇪 Testing German speech generation...")
    german_text = "Kannst du mir erklären was Wasserstoffbrückenbindungen sind?"

    try:
        audio_files = piper.speak(
            text=german_text,
            language="de",
            gender="male",
            quality="medium",
            output_path="/tmp/piper_test_german.wav",
            play=False
        )

        if audio_files:
            print(f"✅ German audio generated: {audio_files[0]}")

            # Check file exists and has content
            audio_file = Path(audio_files[0])
            if audio_file.exists():
                size = audio_file.stat().st_size
                print(f"📁 File size: {size} bytes")

                if size > 1000:  # Reasonable audio file should be > 1KB
                    print("✅ Audio file has reasonable size")
                else:
                    print("⚠️  Audio file seems too small")
            else:
                print("❌ Audio file not found")
        else:
            print("❌ No audio files generated")

    except Exception as e:
        print(f"❌ German speech generation failed: {e}")
        return False

    # Test English speech generation
    print("\n🇺🇸 Testing English speech generation...")
    english_text = "Can you explain what hydrogen bonds are?"

    try:
        audio_files = piper.speak(
            text=english_text,
            language="en",
            gender="female",
            quality="medium",
            output_path="/tmp/piper_test_english.wav",
            play=False
        )

        if audio_files:
            print(f"✅ English audio generated: {audio_files[0]}")

            # Check file exists and has content
            audio_file = Path(audio_files[0])
            if audio_file.exists():
                size = audio_file.stat().st_size
                print(f"📁 File size: {size} bytes")

                if size > 1000:
                    print("✅ Audio file has reasonable size")
                else:
                    print("⚠️  Audio file seems too small")
            else:
                print("❌ Audio file not found")
        else:
            print("❌ No audio files generated")

    except Exception as e:
        print(f"❌ English speech generation failed: {e}")
        return False

    print("\n🎯 Piper TTS test completed!")
    return True

if __name__ == "__main__":
    success = test_piper_basic()
    exit(0 if success else 1)