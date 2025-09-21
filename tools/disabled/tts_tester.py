#!/usr/bin/env python3
"""
TTS Testing and Configuration Utility

This script allows you to test all available TTS engines, compare their quality,
adjust settings, and configure your preferred TTS setup.

Usage:
    python test_tts_system.py                  # Interactive menu
    python test_tts_system.py --test-all       # Test all engines
    python test_tts_system.py --benchmark      # Performance benchmark
    python test_tts_system.py --config         # Configuration wizard
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple

# Add the project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.utils_audio import (
    _try_openai_tts, _try_elevenlabs_tts,
    _try_coqui_tts, _try_modern_hf_tts, _try_improved_system_tts
)
from core.providers.cls_piper_interface import get_piper_interface
from termcolor import colored

class TTSTestSuite:
    """Comprehensive TTS testing and configuration utility."""

    def __init__(self):
        self.test_phrases = [
            "Hello, this is a test of the text-to-speech system.",
            "Please state your choice.",
            "Permission granted. Proceeding with the action.",
            "The agent is processing your request. Please wait.",
            "Error detected. Would you like to retry or cancel?",
            "Welcome to the CLI Agent voice interface system."
        ]

        self.engines = {
            "openai": ("OpenAI TTS", _try_openai_tts),
            "elevenlabs": ("ElevenLabs", _try_elevenlabs_tts),
            "piper": ("Piper TTS", self._test_piper),
            "coqui": ("Coqui XTTS", _try_coqui_tts),
            "huggingface": ("Modern HF Models", _try_modern_hf_tts),
            "system": ("System TTS", _try_improved_system_tts)
        }

        self.voice_options = {
            "openai": ["nova", "alloy", "echo", "fable", "onyx", "shimmer"],
            "elevenlabs": ["rachel", "drew", "clyde", "dave"],
            "piper": ["default", "de_DE-thorsten-high", "en_GB-alba-medium", "en_GB-alan-medium"],
            "general": ["default", "female", "male"]
        }

    def _test_piper(self, text: str, **kwargs) -> bool:
        """Test Piper TTS specifically."""
        try:
            piper = get_piper_interface()
            if not piper.is_available():
                return False
            result = piper.speak(text=text, play=True, **kwargs)
            return result is not None
        except Exception:
            return False

    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{colored('=' * 60, 'cyan')}")
        print(f"{colored(title.center(60), 'cyan', attrs=['bold'])}")
        print(f"{colored('=' * 60, 'cyan')}")

    def check_environment(self) -> Dict[str, bool]:
        """Check which TTS engines are properly configured."""
        print(self.colored_text("🔍 Checking TTS Engine Availability...", "yellow"))

        status = {}

        # Check API keys
        status['openai_key'] = bool(os.getenv('OPENAI_API_KEY'))
        status['elevenlabs_key'] = bool(os.getenv('ELEVENLABS_API_KEY'))

        # Check library availability
        try:
            import openai
            status['openai_lib'] = True
        except ImportError:
            status['openai_lib'] = False

        try:
            import requests
            status['requests'] = True
        except ImportError:
            status['requests'] = False

        try:
            from TTS.api import TTS
            status['coqui'] = True
        except ImportError:
            status['coqui'] = False

        try:
            from transformers import pipeline
            status['transformers'] = True
        except ImportError:
            status['transformers'] = False

        try:
            from core.providers.cls_piper_interface import get_piper_interface
            piper = get_piper_interface()
            status['piper'] = piper.is_available()
        except Exception:
            status['piper'] = False

        # Print status report
        print(f"\n{colored('Environment Status:', 'white', attrs=['bold'])}")
        print(f"  OpenAI API Key: {self._status_icon(status['openai_key'])}")
        print(f"  ElevenLabs API Key: {self._status_icon(status['elevenlabs_key'])}")
        print(f"  OpenAI Library: {self._status_icon(status['openai_lib'])}")
        print(f"  Requests Library: {self._status_icon(status['requests'])}")
        print(f"  Coqui TTS: {self._status_icon(status['coqui'])}")
        print(f"  Transformers: {self._status_icon(status['transformers'])}")
        print(f"  Piper TTS: {self._status_icon(status['piper'])}")

        return status

    def _status_icon(self, available: bool) -> str:
        """Return colored status icon."""
        return colored("✓ Available", "green") if available else colored("✗ Missing", "red")

    def colored_text(self, text: str, color: str, attrs: Optional[List[str]] = None) -> str:
        """Return colored text."""
        return colored(text, color, attrs=attrs or [])

    def test_single_engine(self, engine_name: str, phrase: str = None, voice: str = "default",
                          speed: float = 1.0, save_output: bool = False) -> Tuple[bool, float]:
        """Test a single TTS engine with timing."""
        if engine_name not in self.engines:
            print(f"Unknown engine: {engine_name}")
            return False, 0.0

        phrase = phrase or self.test_phrases[0]
        engine_display, engine_func = self.engines[engine_name]

        print(f"\n{self.colored_text(f'Testing {engine_display}...', 'yellow')}")
        print(f"  Text: '{phrase[:50]}{'...' if len(phrase) > 50 else ''}'")
        print(f"  Voice: {voice}, Speed: {speed}")

        # Prepare output path if saving
        output_path = None
        if save_output:
            output_path = f"tts_test_{engine_name}_{int(time.time())}.wav"

        start_time = time.time()

        try:
            if engine_name == "system":
                # System TTS doesn't use the same parameters
                success = engine_func(phrase)
            else:
                success = engine_func(
                    text=phrase,
                    voice=voice,
                    speed=speed,
                    output_path=output_path,
                    play=True
                )

            elapsed = time.time() - start_time

            if success:
                print(f"  {self.colored_text('✓ Success', 'green')} ({elapsed:.2f}s)")
                if save_output and output_path and os.path.exists(output_path):
                    print(f"  Audio saved to: {output_path}")
            else:
                print(f"  {self.colored_text('✗ Failed', 'red')} ({elapsed:.2f}s)")

            return success, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  {self.colored_text('✗ Error', 'red')}: {e} ({elapsed:.2f}s)")
            return False, elapsed

    def test_all_engines(self, phrase: str = None, save_outputs: bool = False):
        """Test all available TTS engines."""
        self.print_header("Testing All TTS Engines")

        phrase = phrase or self.test_phrases[0]
        results = {}

        print(f"Test phrase: {self.colored_text(phrase, 'white', ['bold'])}")

        for engine_name, (engine_display, _) in self.engines.items():
            success, elapsed = self.test_single_engine(
                engine_name, phrase, save_output=save_outputs
            )
            results[engine_name] = {"success": success, "time": elapsed, "name": engine_display}

            # Add a small delay between tests
            time.sleep(0.5)

        # Print summary
        print(f"\n{self.colored_text('Test Results Summary:', 'white', ['bold'])}")
        print(f"{'Engine':<20} {'Status':<10} {'Time':<10}")
        print("-" * 40)

        for engine_name, result in results.items():
            status_text = colored("PASS", "green") if result["success"] else colored("FAIL", "red")
            print(f"{result['name']:<20} {status_text:<15} {result['time']:.2f}s")

        # Recommendations
        working_engines = [name for name, result in results.items() if result["success"]]
        if working_engines:
            print(f"\n{self.colored_text('Working engines found:', 'green')} {', '.join(working_engines)}")
        else:
            print(f"\n{self.colored_text('No working engines found! Check your setup.', 'red')}")

    def voice_comparison(self, engine: str = "openai"):
        """Compare different voices from the same engine."""
        if engine not in self.engines:
            print(f"Unknown engine: {engine}")
            return

        self.print_header(f"Voice Comparison - {self.engines[engine][0]}")

        voices = self.voice_options.get(engine, self.voice_options["general"])
        test_phrase = "Hello, this is a voice comparison test. How do I sound?"

        print(f"Testing voices: {', '.join(voices)}")
        print(f"Phrase: {self.colored_text(test_phrase, 'white', ['bold'])}")

        for i, voice in enumerate(voices, 1):
            print(f"\n{self.colored_text(f'Voice {i}: {voice}', 'cyan')}")
            input("Press Enter to play this voice...")

            success, elapsed = self.test_single_engine(engine, test_phrase, voice=voice)

            if not success:
                print(f"  Voice '{voice}' failed, skipping...")
                continue

            rating = input("  Rate this voice (1-5, or 's' to skip): ").strip()
            if rating.isdigit():
                print(f"  You rated '{voice}': {rating}/5")

    def speed_test(self, engine: str = "piper"):
        """Test different speech speeds."""
        if engine not in self.engines:
            print(f"Unknown engine: {engine}")
            return

        self.print_header(f"Speed Test - {self.engines[engine][0]}")

        speeds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        test_phrase = "This is a speech speed test. Listen carefully to the pace."

        for speed in speeds:
            print(f"\n{self.colored_text(f'Speed: {speed}x', 'cyan')}")
            input("Press Enter to play...")

            success, elapsed = self.test_single_engine(engine, test_phrase, speed=speed)
            if not success:
                print(f"  Speed {speed}x failed")

    def benchmark_performance(self):
        """Benchmark TTS engine performance."""
        self.print_header("Performance Benchmark")

        phrases = self.test_phrases[:3]  # Use first 3 phrases
        engines_to_test = []

        # Only test engines that are likely to work
        status = self.check_environment()
        if status.get('openai_key') and status.get('openai_lib'):
            engines_to_test.append('openai')
        if status.get('piper'):
            engines_to_test.append('piper')
        if status.get('transformers'):
            engines_to_test.append('huggingface')
        engines_to_test.append('system')  # Always available

        print(f"\nBenchmarking engines: {', '.join(engines_to_test)}")
        print(f"Using {len(phrases)} test phrases")

        results = {}

        for engine in engines_to_test:
            print(f"\n{self.colored_text(f'Benchmarking {engine}...', 'yellow')}")
            times = []
            successes = 0

            for i, phrase in enumerate(phrases, 1):
                print(f"  Phrase {i}/{len(phrases)}...", end=" ")
                success, elapsed = self.test_single_engine(engine, phrase)
                if success:
                    times.append(elapsed)
                    successes += 1
                    print(f"{elapsed:.2f}s")
                else:
                    print("FAILED")

            if times:
                avg_time = sum(times) / len(times)
                results[engine] = {
                    "avg_time": avg_time,
                    "success_rate": successes / len(phrases),
                    "total_successes": successes
                }

        # Print benchmark results
        print(f"\n{self.colored_text('Benchmark Results:', 'white', ['bold'])}")
        print(f"{'Engine':<15} {'Avg Time':<12} {'Success Rate':<15} {'Total Success'}")
        print("-" * 60)

        for engine, result in sorted(results.items(), key=lambda x: x[1]['avg_time']):
            success_rate = f"{result['success_rate']*100:.0f}%"
            print(f"{engine:<15} {result['avg_time']:.2f}s{'':<6} {success_rate:<15} {result['total_successes']}/{len(phrases)}")

    def configuration_wizard(self):
        """Interactive configuration wizard."""
        self.print_header("TTS Configuration Wizard")

        print("This wizard will help you set up optimal TTS settings.\n")

        # Check current environment
        status = self.check_environment()

        # API Key setup
        if not status.get('openai_key'):
            setup_openai = input(f"\n{self.colored_text('Setup OpenAI TTS?', 'yellow')} (highest quality) [y/N]: ").lower().startswith('y')
            if setup_openai:
                api_key = input("Enter your OpenAI API key: ").strip()
                if api_key:
                    print(f"Add this to your environment: {self.colored_text(f'export OPENAI_API_KEY=\"{api_key}\"', 'green')}")

        if not status.get('elevenlabs_key'):
            setup_elevenlabs = input(f"\n{self.colored_text('Setup ElevenLabs TTS?', 'yellow')} (premium natural voices) [y/N]: ").lower().startswith('y')
            if setup_elevenlabs:
                api_key = input("Enter your ElevenLabs API key: ").strip()
                if api_key:
                    print(f"Add this to your environment: {self.colored_text(f'export ELEVENLABS_API_KEY=\"{api_key}\"', 'green')}")

        # Library installation recommendations
        print(f"\n{self.colored_text('Recommended installations:', 'white', ['bold'])}")

        if not status.get('coqui'):
            print(f"  For Coqui TTS: {self.colored_text('pip install TTS', 'green')}")

        if not status.get('transformers'):
            print(f"  For HuggingFace models: {self.colored_text('pip install transformers torch', 'green')}")

        # Test recommended setup
        test_setup = input(f"\n{self.colored_text('Test your current setup?', 'yellow')} [Y/n]: ")
        if not test_setup.lower().startswith('n'):
            self.test_all_engines()

    def interactive_menu(self):
        """Main interactive menu."""
        while True:
            self.print_header("TTS Testing & Configuration Utility")

            menu_options = [
                ("1", "Check Environment", self.check_environment),
                ("2", "Test All Engines", lambda: self.test_all_engines(save_outputs=True)),
                ("3", "Test Single Engine", self.test_single_engine_menu),
                ("4", "Voice Comparison", self.voice_comparison_menu),
                ("5", "Speed Test", self.speed_test_menu),
                ("6", "Performance Benchmark", self.benchmark_performance),
                ("7", "Configuration Wizard", self.configuration_wizard),
                ("8", "Custom Test", self.custom_test),
                ("q", "Quit", None)
            ]

            for option, description, _ in menu_options:
                print(f"  {self.colored_text(option, 'cyan')}: {description}")

            choice = input(f"\n{self.colored_text('Choose an option:', 'white')} ").strip().lower()

            if choice == 'q':
                print("Goodbye!")
                break

            # Find and execute the chosen option
            for option, description, func in menu_options:
                if choice == option.lower() and func:
                    try:
                        func()
                    except KeyboardInterrupt:
                        print(f"\n{self.colored_text('Interrupted by user', 'yellow')}")
                    except Exception as e:
                        print(f"\n{self.colored_text(f'Error: {e}', 'red')}")

                    input(f"\n{self.colored_text('Press Enter to continue...', 'white')}")
                    break
            else:
                if choice != 'q':
                    print(f"{self.colored_text('Invalid choice, try again.', 'red')}")

    def test_single_engine_menu(self):
        """Menu for testing a single engine."""
        print(f"\n{self.colored_text('Available engines:', 'white')}")
        for i, (engine, (display, _)) in enumerate(self.engines.items(), 1):
            print(f"  {i}: {display} ({engine})")

        try:
            choice = int(input("Choose engine number: ")) - 1
            engine_names = list(self.engines.keys())
            if 0 <= choice < len(engine_names):
                engine = engine_names[choice]
                phrase = input("Enter test phrase (or press Enter for default): ").strip()
                phrase = phrase or self.test_phrases[0]
                self.test_single_engine(engine, phrase, save_output=True)
        except (ValueError, IndexError):
            print("Invalid choice.")

    def voice_comparison_menu(self):
        """Menu for voice comparison."""
        engines_with_voices = ["openai", "elevenlabs", "piper"]
        print(f"\n{self.colored_text('Engines with voice options:', 'white')}")
        for i, engine in enumerate(engines_with_voices, 1):
            print(f"  {i}: {self.engines[engine][0]} ({engine})")

        try:
            choice = int(input("Choose engine number: ")) - 1
            if 0 <= choice < len(engines_with_voices):
                self.voice_comparison(engines_with_voices[choice])
        except (ValueError, IndexError):
            print("Invalid choice.")

    def speed_test_menu(self):
        """Menu for speed testing."""
        print(f"\n{self.colored_text('Available engines:', 'white')}")
        for i, (engine, (display, _)) in enumerate(self.engines.items(), 1):
            print(f"  {i}: {display} ({engine})")

        try:
            choice = int(input("Choose engine number: ")) - 1
            engine_names = list(self.engines.keys())
            if 0 <= choice < len(engine_names):
                self.speed_test(engine_names[choice])
        except (ValueError, IndexError):
            print("Invalid choice.")

    def custom_test(self):
        """Custom test with user-defined parameters."""
        print(f"\n{self.colored_text('Custom TTS Test', 'white', ['bold'])}")

        # Get engine
        print("Available engines:")
        for i, (engine, (display, _)) in enumerate(self.engines.items(), 1):
            print(f"  {i}: {display}")

        try:
            engine_choice = int(input("Choose engine: ")) - 1
            engine_names = list(self.engines.keys())
            engine = engine_names[engine_choice]
        except (ValueError, IndexError):
            print("Invalid engine choice.")
            return

        # Get parameters
        text = input("Enter text to speak: ").strip()
        if not text:
            text = self.test_phrases[0]

        voice = input("Enter voice (default: 'default'): ").strip() or "default"

        try:
            speed = float(input("Enter speed (default: 1.0): ").strip() or "1.0")
        except ValueError:
            speed = 1.0

        save = input("Save audio file? [y/N]: ").strip().lower().startswith('y')

        print(f"\n{self.colored_text('Running custom test...', 'yellow')}")
        self.test_single_engine(engine, text, voice=voice, speed=speed, save_output=save)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TTS Testing and Configuration Utility")
    parser.add_argument("--test-all", action="store_true", help="Test all available engines")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--config", action="store_true", help="Run configuration wizard")
    parser.add_argument("--engine", help="Test specific engine")
    parser.add_argument("--text", help="Custom text to test")
    parser.add_argument("--voice", default="default", help="Voice to use")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--save", action="store_true", help="Save audio outputs")

    args = parser.parse_args()
    tts_tester = TTSTestSuite()

    if args.test_all:
        tts_tester.test_all_engines(phrase=args.text, save_outputs=args.save)
    elif args.benchmark:
        tts_tester.benchmark_performance()
    elif args.config:
        tts_tester.configuration_wizard()
    elif args.engine:
        tts_tester.test_single_engine(
            args.engine, phrase=args.text, voice=args.voice,
            speed=args.speed, save_output=args.save
        )
    else:
        tts_tester.interactive_menu()


if __name__ == "__main__":
    main()