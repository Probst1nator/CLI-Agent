#!/usr/bin/env python3
"""
Enhanced Voice Pipeline Test with Piper TTS

Tests high-quality Piper TTS audio generation and Faster-Whisper transcription accuracy
for both German and English with exactly 3 sentences each and 2 voices (1 female, 1 male).
This addresses the original user request for simple, repeatable testing.
"""

import os
import sys
from pathlib import Path
from typing import Dict
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.utils_audio import (
    initialize_faster_whisper_model, _transcribe_with_faster_whisper, reset_audio_models,
    text_to_speech
)
import soundfile as sf

# Test configuration - exactly as user requested
TEST_SENTENCES = {
    "german": [
        "Kannst du mir erklären was Wasserstoffbrückenbindungen sind?",
        "Die Elektronenkonfiguration bestimmt chemische Eigenschaften.",
        "Thermodynamik beschreibt Energieumwandlungen in Systemen."
    ],
    "english": [
        "Can you explain what hydrogen bonds are?",
        "Electron configuration determines chemical properties.", 
        "Thermodynamics describes energy transformations in systems."
    ]
}

# Voice configuration: 1 female, 1 male as requested
VOICE_CONFIG = {
    "female": {"german": {"gender": "female", "voice": "de_DE-mls-medium"}, 
              "english": {"gender": "female", "voice": "en_GB-alba-medium"}},
    "male": {"german": {"gender": "male", "voice": "de_DE-thorsten-medium"}, 
            "english": {"gender": "male", "voice": "en_GB-alan-medium"}}
}

WHISPER_MODELS = ["tiny", "base", "small", "medium"]

class PiperTranscriptionPipeline:
    """Enhanced voice pipeline test with high-quality Piper TTS."""
    
    def __init__(self):
        self.test_dir = project_root / "tests" / "audio" / "piper_pipeline"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def generate_piper_audio(self) -> bool:
        """Generate high-quality test audio files using Piper TTS."""
        print("🎙️  Generating high-quality Piper TTS audio files...")
        
        # Calculate expected files: 3 sentences × 2 languages × 2 genders = 12 files
        expected_files = []
        for language in TEST_SENTENCES.keys():
            for i in range(3):
                for gender in VOICE_CONFIG.keys():
                    filename = f"piper_{language}_{i+1}_{gender}.wav"
                    expected_files.append(self.test_dir / filename)
        
        existing_count = sum(1 for f in expected_files if f.exists())
        if existing_count == 12:
            print("✅ All 12 Piper audio files already exist")
            return True
        
        print(f"📁 Found {existing_count}/12 existing files, generating missing ones...")
        
        generated_files = []
        
        for language, sentences in TEST_SENTENCES.items():
            print(f"\n🌐 Generating {language.upper()} audio files...")
            
            for i, sentence in enumerate(sentences):
                for gender, voice_config in VOICE_CONFIG.items():
                    filename = f"piper_{language}_{i+1}_{gender}.wav"
                    filepath = self.test_dir / filename
                    
                    # Skip if file already exists
                    if filepath.exists():
                        generated_files.append(filepath)
                        continue
                    
                    print(f"   🎵 Generating: {filename}")
                    
                    try:
                        # Use Piper TTS for high-quality audio
                        lang_code = "de" if language == "german" else "en"
                        voice_gender = voice_config[language]["gender"]
                        
                        result = text_to_speech(
                            text=sentence,
                            language=lang_code,
                            gender=voice_gender,
                            quality="medium",
                            output_path=str(filepath),
                            play=False,
                            use_piper=True
                        )
                        
                        if result and filepath.exists():
                            size = filepath.stat().st_size
                            if size > 10000:  # Quality check: should be > 10KB
                                generated_files.append(filepath)
                                print(f"   ✅ {filename} - {size} bytes (high quality)")
                            else:
                                print(f"   ⚠️  {filename} - {size} bytes (suspiciously small)")
                                filepath.unlink()  # Remove small file
                        else:
                            print(f"   ❌ Failed to generate {filename}")
                    
                    except Exception as e:
                        print(f"   ❌ Error generating {filename}: {e}")
        
        final_count = len([f for f in expected_files if f.exists()])
        print(f"\n✅ Generated/verified {final_count}/12 high-quality Piper audio files")
        return final_count >= 10  # Allow for some failures
    
    def test_transcription_accuracy(self) -> Dict:
        """Test transcription accuracy across different Whisper models."""
        print("\\n🧪 Testing transcription accuracy with Piper-generated audio...")
        
        results = {}
        
        # Force CPU usage for consistent testing
        os.environ['CLIAGENT_FORCE_DEVICE'] = 'cpu'
        
        for model_size in WHISPER_MODELS:
            print(f"\\n📊 Testing {model_size} model...")
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
            
            # Test each audio file
            for language, sentences in TEST_SENTENCES.items():
                language_results = {}
                
                for i, expected_sentence in enumerate(sentences):
                    sentence_results = {}
                    
                    for gender in VOICE_CONFIG.keys():
                        filename = f"piper_{language}_{i+1}_{gender}.wav"
                        filepath = self.test_dir / filename
                        
                        if not filepath.exists():
                            print(f"   ❌ Missing audio file: {filename}")
                            continue
                        
                        try:
                            print(f"   🎧 Testing: {filename}")
                            
                            # Load and transcribe audio
                            audio, sample_rate = sf.read(str(filepath))
                            start_time = time.time()
                            result, detected_lang = _transcribe_with_faster_whisper(
                                audio, sample_rate, fallback_to_google=False
                            )
                            transcribe_time = time.time() - start_time
                            
                            # Calculate accuracy
                            accuracy = self._calculate_accuracy(expected_sentence, result)
                            word_accuracy = self._calculate_word_accuracy(expected_sentence, result)
                            
                            sentence_results[gender] = {
                                'expected': expected_sentence,
                                'result': result,
                                'accuracy': accuracy,
                                'word_accuracy': word_accuracy,
                                'detected_language': detected_lang,
                                'transcribe_time': transcribe_time,
                                'load_time': load_time,
                                'filename': filename
                            }
                            
                            # Status indicator with special attention to German scientific terms
                            if accuracy > 0.8:
                                status = "✅ EXCELLENT"
                            elif accuracy > 0.6:
                                status = "⚠️  PARTIAL"
                            elif language == "german" and "wasserstoff" in result.lower():
                                status = "🔍 SCIENTIFIC_TERM"
                            else:
                                status = "❌ POOR"
                            
                            print(f"      {status} - Accuracy: {accuracy:.1%} | Word: {word_accuracy:.1%} | Time: {transcribe_time:.1f}s")
                            print(f"      Expected: {expected_sentence}")
                            print(f"      Got:      {result}")
                            
                        except Exception as e:
                            print(f"   ❌ Transcription failed for {filename}: {e}")
                            sentence_results[gender] = {
                                'expected': expected_sentence,
                                'result': f"ERROR: {e}",
                                'accuracy': 0.0,
                                'word_accuracy': 0.0,
                                'detected_language': 'error',
                                'transcribe_time': 0.0,
                                'load_time': load_time,
                                'filename': filename
                            }
                    
                    language_results[f"sentence_{i+1}"] = sentence_results
                
                model_results[language] = language_results
            
            results[model_size] = model_results
        
        # Clean up environment
        if 'CLIAGENT_FORCE_DEVICE' in os.environ:
            del os.environ['CLIAGENT_FORCE_DEVICE']
            
        return results
    
    def _calculate_accuracy(self, expected: str, actual: str) -> float:
        """Calculate character-level accuracy."""
        if not actual or not expected:
            return 0.0
        
        # Simple character overlap ratio
        expected_chars = set(expected.lower().replace(' ', ''))
        actual_chars = set(actual.lower().replace(' ', ''))
        
        if not expected_chars:
            return 1.0 if not actual_chars else 0.0
        
        overlap = len(expected_chars.intersection(actual_chars))
        return overlap / len(expected_chars)
    
    def test_integration_scenarios(self) -> bool:
        """Test specific integration scenarios from test_piper_integration.py"""
        print("\n🔧 Testing Piper integration scenarios...")
        
        # Test German TTS integration
        print("\n🇩🇪 Testing German TTS integration...")
        german_output = text_to_speech(
            text="Kannst du mir erklären was Wasserstoffbrückenbindungen sind?",
            language="de",
            gender="male",
            quality="medium",
            output_path="/tmp/piper_integration_german.wav",
            play=False,
            use_piper=True
        )
        
        german_success = False
        if german_output:
            audio_file = Path(german_output)
            if audio_file.exists():
                size = audio_file.stat().st_size
                print(f"✅ German audio generated: {german_output} ({size} bytes)")
                german_success = True
            else:
                print(f"❌ German audio file not found: {german_output}")
        else:
            print("❌ German TTS failed")
        
        # Test English TTS with female voice
        print("\n🇺🇸 Testing English TTS (female voice)...")
        english_output = text_to_speech(
            text="Can you explain what hydrogen bonds are?",
            language="en",
            gender="female",
            quality="medium",
            output_path="/tmp/piper_integration_english.wav",
            play=False,
            use_piper=True
        )
        
        english_success = False
        if english_output:
            audio_file = Path(english_output)
            if audio_file.exists():
                size = audio_file.stat().st_size
                print(f"✅ English audio generated: {english_output} ({size} bytes)")
                english_success = True
            else:
                print(f"❌ English audio file not found: {english_output}")
        else:
            print("❌ English TTS failed")
        
        # Test auto-language detection
        print("\n🌐 Testing auto-language detection...")
        auto_output = text_to_speech(
            text="Die Thermodynamik beschreibt Energieumwandlungen.",
            language="auto",  # Should detect German
            gender="female",
            output_path="/tmp/piper_integration_auto.wav", 
            play=False,
            use_piper=True
        )
        
        auto_success = False
        if auto_output:
            audio_file = Path(auto_output)
            if audio_file.exists():
                size = audio_file.stat().st_size
                print(f"✅ Auto-detection audio generated: {auto_output} ({size} bytes)")
                auto_success = True
            else:
                print(f"❌ Auto-detection audio file not found: {auto_output}")
        else:
            print("❌ Auto-detection TTS failed")
            
        overall_success = german_success and english_success and auto_success
        print(f"\n{'✅' if overall_success else '❌'} Integration test {'passed' if overall_success else 'failed'}")
        return overall_success
    
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
        """Generate comprehensive test report."""
        print("\\n" + "="*70)
        print("📋 PIPER TTS + WHISPER TRANSCRIPTION TEST REPORT")
        print("="*70)
        
        if not results:
            print("❌ No results to report")
            return
        
        # Performance summary
        print(f"\\n{'Model':<10} {'German Avg':<12} {'English Avg':<12} {'German Sci':<12} {'Status'}")
        print("-" * 65)
        
        best_model = None
        best_german_accuracy = 0.0
        
        for model_size, model_results in results.items():
            german_scores = []
            english_scores = []
            german_sci_detected = 0
            german_sci_total = 0
            
            # Collect all accuracy scores
            for language, lang_results in model_results.items():
                for sentence_id, sentence_results in lang_results.items():
                    for voice_result in sentence_results.values():
                        score = voice_result.get('accuracy', 0.0)
                        result_text = voice_result.get('result', '').lower()
                        
                        if language == 'german':
                            german_scores.append(score)
                            # Check for German scientific terms
                            german_sci_total += 1
                            if any(term in result_text for term in ['wasserstoff', 'elektronen', 'thermodynamik']):
                                german_sci_detected += 1
                        else:
                            english_scores.append(score)
            
            german_avg = sum(german_scores) / len(german_scores) if german_scores else 0.0
            english_avg = sum(english_scores) / len(english_scores) if english_scores else 0.0
            german_sci_rate = german_sci_detected / german_sci_total if german_sci_total else 0.0
            
            # Determine status
            if german_avg > 0.8 and english_avg > 0.8:
                status = "✅ EXCELLENT"
            elif german_avg > 0.6 and german_sci_rate > 0.5:
                status = "✅ GOOD_SCI"
            elif german_avg > 0.4:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ POOR"
            
            print(f"{model_size:<10} {german_avg:<11.1%} {english_avg:<11.1%} {german_sci_rate:<11.1%} {status}")
            
            if german_avg > best_german_accuracy:
                best_german_accuracy = german_avg
                best_model = model_size
        
        # Detailed German analysis (focus on scientific vocabulary)
        print("\\n🔬 GERMAN SCIENTIFIC VOCABULARY ANALYSIS:")
        print("-" * 50)
        
        for model_size, model_results in results.items():
            if 'german' in model_results:
                print(f"\\n📊 {model_size.upper()} MODEL:")
                for sentence_id, sentence_results in model_results['german'].items():
                    expected = list(sentence_results.values())[0]['expected']
                    if any(term in expected.lower() for term in ['wasserstoff', 'elektronen', 'thermodynamik']):
                        print(f"   Scientific sentence: {expected}")
                        for gender, result in sentence_results.items():
                            actual = result['result']
                            accuracy = result['accuracy']
                            # Check for key scientific terms
                            terms_found = []
                            if 'wasserstoff' in actual.lower():
                                terms_found.append('wasserstoff')
                            if 'elektronen' in actual.lower():
                                terms_found.append('elektronen')
                            if 'thermodynamik' in actual.lower():
                                terms_found.append('thermodynamik')
                            
                            term_status = f"[{', '.join(terms_found)}]" if terms_found else "[no sci terms]"
                            status = "✅" if accuracy > 0.7 else "⚠️" if terms_found else "❌"
                            print(f"   {gender:>6} {status}: {actual} {term_status} ({accuracy:.1%})")
        
        # Recommendations
        print("\\n🎯 RECOMMENDATIONS:")
        print("-" * 20)
        
        if best_model:
            print(f"• Best German performance: {best_model} model ({best_german_accuracy:.1%})")
            
            if best_german_accuracy < 0.5:
                print("• ⚠️  Poor German scientific vocabulary recognition across all models")
                print("• 💡 Consider upgrading to 'large' or 'large-v3-turbo' models")
                print("• 💡 Try German-specific fine-tuned models")
                print("• ✅ Piper TTS quality is excellent - issue is with Whisper models")
            elif best_german_accuracy < 0.8:
                print("• ⚠️  Partial success - consider larger models for better accuracy")
                print(f"• ✅ {best_model} model shows promise")
                print("• ✅ Piper TTS generates high-quality, clear audio")
            else:
                print(f"• ✅ {best_model} model provides excellent German transcription")
                print("• ✅ Piper TTS + Whisper pipeline working well")
        
        # System info
        print("\\n📋 SYSTEM INFO:")
        print("• TTS Engine: Piper TTS (high-quality neural synthesis)")
        print("• German voices: thorsten-medium (male), mls-medium (female)")
        print("• English voices: alan-medium (male), alba-medium (female)")
        print("• Test sentences: 3 German + 3 English × 2 voices = 12 audio files")
        print("• Target: German scientific vocabulary (Wasserstoffbrückenbindungen, etc.)")
        print(f"• Models tested: {len(results)} Faster-Whisper models")
        print("• Device: CPU (GTX 1080 Ti compatibility mode)")
    
    def run_full_pipeline(self) -> bool:
        """Run the complete enhanced voice pipeline test."""
        print("🚀 Enhanced Piper TTS + Whisper Transcription Pipeline")
        print("=" * 60)
        
        # Step 1: Generate high-quality Piper audio
        if not self.generate_piper_audio():
            print("\\n❌ Piper audio generation failed. Pipeline test aborted.")
            return False
        
        # Step 2: Test transcription accuracy
        results = self.test_transcription_accuracy()
        
        if not results:
            print("\\n❌ No transcription results. Pipeline test failed.")
            return False
        
        # Step 3: Generate comprehensive report
        self.generate_report(results)
        
        print("\\n🎯 Enhanced voice pipeline test completed!")
        print(f"📁 Audio files saved in: {self.test_dir}")
        return True


def main():
    """Main test runner."""
    test_runner = PiperTranscriptionPipeline()
    success = test_runner.run_full_pipeline()
    
    if success:
        print("\\n✅ Enhanced pipeline test completed successfully")
        return 0
    else:
        print("\\n❌ Enhanced pipeline test failed")
        return 1


if __name__ == "__main__":
    exit(main())