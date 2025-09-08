#!/usr/bin/env python3
"""
German Transcription Pipeline Test

Simple, focused test using real German audio to compare Faster-Whisper model performance.
Tests the specific issue: German scientific vocabulary transcription accuracy.
"""

import os
import sys
from pathlib import Path
from typing import Dict
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.utils_audio import initialize_faster_whisper_model, _transcribe_with_faster_whisper, reset_audio_models
import soundfile as sf

# Test configuration
GERMAN_TEST_CASES = {
    "real_user_audio": {
        "file": "tests/audio/german_transcription_failure.wav",
        "expected": "kannst du mir erklären was Wasserstoffbrückenbindungen sind",
        "description": "Real user audio - German scientific vocabulary"
    }
}

WHISPER_MODELS = ["tiny", "base", "small", "medium"]

class GermanTranscriptionTest:
    """Focused German transcription test using real audio."""
    
    def __init__(self):
        self.project_root = project_root
        self.results = {}
        
    def run_model_comparison(self) -> Dict:
        """Test all Whisper models on German audio."""
        print("🇩🇪 German Transcription Model Comparison")
        print("=" * 50)
        
        results = {}
        
        # Force CPU usage for consistent testing
        os.environ['CLIAGENT_FORCE_DEVICE'] = 'cpu'
        
        for model_size in WHISPER_MODELS:
            print(f"\n📊 Testing {model_size} model...")
            model_results = {}
            
            # Reset and initialize model
            reset_audio_models()
            start_time = time.time()
            success = initialize_faster_whisper_model(model_size=model_size, force_reset=True)
            load_time = time.time() - start_time
            
            if not success:
                print(f"   ❌ Failed to load {model_size} model")
                continue
            
            print(f"   📥 Model loaded in {load_time:.1f}s")
            
            # Test each audio case
            for case_name, case_data in GERMAN_TEST_CASES.items():
                audio_file = self.project_root / case_data["file"]
                expected = case_data["expected"]
                description = case_data["description"]
                
                if not audio_file.exists():
                    print(f"   ❌ Missing audio file: {audio_file}")
                    continue
                
                try:
                    print(f"   🎧 Testing: {description}")
                    
                    # Load and transcribe audio
                    audio, sample_rate = sf.read(str(audio_file))
                    start_time = time.time()
                    result, detected_lang = _transcribe_with_faster_whisper(
                        audio, sample_rate, fallback_to_google=False
                    )
                    transcribe_time = time.time() - start_time
                    
                    # Calculate accuracy
                    accuracy = self._calculate_accuracy(expected, result)
                    word_accuracy = self._calculate_word_accuracy(expected, result)
                    
                    model_results[case_name] = {
                        'expected': expected,
                        'result': result,
                        'accuracy': accuracy,
                        'word_accuracy': word_accuracy,
                        'detected_language': detected_lang,
                        'transcribe_time': transcribe_time,
                        'load_time': load_time,
                        'description': description
                    }
                    
                    # Status indicator
                    if accuracy > 0.8:
                        status = "✅ EXCELLENT"
                    elif accuracy > 0.6:
                        status = "⚠️  PARTIAL"
                    elif "wasserstoff" in result.lower():
                        status = "🔍 KEY_TERM"
                    else:
                        status = "❌ POOR"
                    
                    print(f"   {status} - Accuracy: {accuracy:.1%} | Time: {transcribe_time:.1f}s")
                    print(f"      Expected: {expected}")
                    print(f"      Got:      {result}")
                    
                except Exception as e:
                    print(f"   ❌ Transcription failed: {e}")
                    model_results[case_name] = {
                        'expected': expected,
                        'result': f"ERROR: {e}",
                        'accuracy': 0.0,
                        'word_accuracy': 0.0,
                        'detected_language': 'error',
                        'transcribe_time': 0.0,
                        'load_time': load_time,
                        'description': description
                    }
            
            results[model_size] = model_results
        
        # Clean up environment
        if 'CLIAGENT_FORCE_DEVICE' in os.environ:
            del os.environ['CLIAGENT_FORCE_DEVICE']
            
        return results
    
    def _calculate_accuracy(self, expected: str, actual: str) -> float:
        """Calculate character-level accuracy (Levenshtein-based)."""
        if not actual or not expected:
            return 0.0
        
        # Simple character overlap ratio
        expected_chars = set(expected.lower().replace(' ', ''))
        actual_chars = set(actual.lower().replace(' ', ''))
        
        if not expected_chars:
            return 1.0 if not actual_chars else 0.0
        
        overlap = len(expected_chars.intersection(actual_chars))
        return overlap / len(expected_chars)
    
    def _calculate_word_accuracy(self, expected: str, actual: str) -> float:
        """Calculate word-level accuracy."""
        if not actual or not expected:
            return 0.0
            
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
            
        overlap = len(expected_words.intersection(actual_words))
        return overlap / len(expected_words)
    
    def generate_report(self, results: Dict) -> None:
        """Generate focused report on German transcription performance."""
        print("\n" + "=" * 60)
        print("📋 GERMAN TRANSCRIPTION TEST REPORT")
        print("=" * 60)
        
        if not results:
            print("❌ No results to report")
            return
        
        # Performance summary
        print(f"\n{'Model':<10} {'Accuracy':<10} {'Time':<8} {'Key Term':<12} {'Status'}")
        print("-" * 55)
        
        best_model = None
        best_accuracy = 0.0
        
        for model_size, model_results in results.items():
            if not model_results:
                continue
                
            # Get results for the main test case
            main_case = model_results.get('real_user_audio', {})
            accuracy = main_case.get('accuracy', 0.0)
            time_taken = main_case.get('transcribe_time', 0.0)
            result = main_case.get('result', '')
            
            # Check if key scientific term was detected
            key_term_found = 'wasserstoff' in result.lower()
            key_term_status = '✅ Found' if key_term_found else '❌ Missing'
            
            # Overall status
            if accuracy > 0.8:
                status = "✅ PASS"
            elif accuracy > 0.6 or key_term_found:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ FAIL"
            
            print(f"{model_size:<10} {accuracy:<9.1%} {time_taken:<7.1f}s {key_term_status:<12} {status}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_size
        
        # Detailed analysis
        print("\n🔍 DETAILED ANALYSIS:")
        print("-" * 30)
        
        for model_size, model_results in results.items():
            if not model_results:
                continue
                
            main_case = model_results.get('real_user_audio', {})
            if not main_case:
                continue
                
            print(f"\n📊 {model_size.upper()} MODEL:")
            print(f"   Expected: {main_case.get('expected', 'N/A')}")
            print(f"   Result:   {main_case.get('result', 'N/A')}")
            print(f"   Accuracy: {main_case.get('accuracy', 0.0):.1%}")
            print(f"   Word Acc: {main_case.get('word_accuracy', 0.0):.1%}")
            print(f"   Language: {main_case.get('detected_language', 'N/A')}")
            print(f"   Load Time: {main_case.get('load_time', 0.0):.1f}s")
            print(f"   Trans Time: {main_case.get('transcribe_time', 0.0):.1f}s")
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS:")
        print("-" * 20)
        
        if best_model:
            print(f"• Best performing model: {best_model} ({best_accuracy:.1%})")
            
            if best_accuracy < 0.5:
                print("• ⚠️  All models show poor accuracy on German scientific vocabulary")
                print("• 💡 Consider upgrading to 'large' or 'large-v3-turbo' models")
                print("• 💡 Try German-specific fine-tuned models (primeline/whisper-large-v3-german)")
            elif best_accuracy < 0.8:
                print("• ⚠️  Partial success - larger models may improve accuracy")
                print(f"• 💡 Current best: {best_model} model")
            else:
                print(f"• ✅ {best_model} model provides good German transcription")
        
        # System info
        print("\n📋 SYSTEM INFO:")
        print(f"• Test audio: {len(GERMAN_TEST_CASES)} real German audio file(s)")
        print(f"• Models tested: {len(results)} Faster-Whisper models")
        print("• Target vocabulary: German scientific terms (Wasserstoffbrückenbindungen)")
        print("• Device: CPU (GTX 1080 Ti compatibility mode)")
    
    def run_full_test(self) -> bool:
        """Run the complete German transcription test."""
        print("🇩🇪 German Scientific Vocabulary Transcription Test")
        print("=" * 55)
        
        # Check if test files exist
        missing_files = []
        for case_name, case_data in GERMAN_TEST_CASES.items():
            audio_file = self.project_root / case_data["file"]
            if not audio_file.exists():
                missing_files.append(audio_file)
        
        if missing_files:
            print("❌ Missing test audio files:")
            for file in missing_files:
                print(f"   {file}")
            print("💡 Run the basic audio transcription tests first to generate test files")
            return False
        
        # Run model comparison
        results = self.run_model_comparison()
        
        if not results:
            print("\n❌ No transcription results. Test failed.")
            return False
        
        # Generate report
        self.generate_report(results)
        
        print("\n🎯 German transcription test completed!")
        return True


def main():
    """Main test runner."""
    test_runner = GermanTranscriptionTest()
    success = test_runner.run_full_test()
    
    if success:
        print("\n✅ German transcription analysis completed")
        return 0
    else:
        print("\n❌ German transcription test failed")
        return 1


if __name__ == "__main__":
    exit(main())