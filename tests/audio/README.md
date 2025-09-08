# Test Audio Files

This directory contains audio files generated during testing of the TTS and transcription pipeline.

## Directory Structure

- `german_transcription_failure.wav` - Known failure case for German scientific vocabulary testing
- `pipeline/` - Generated test audio files (12 files: 3 German + 3 English sentences × 2 voice types)  
- `piper_pipeline/` - Piper TTS generated test audio files for transcription accuracy testing

## File Generation

These audio files are **generated automatically** by the test suite and should not be committed to version control. They are ignored by `.gitignore` to keep repository size manageable.

To regenerate test audio files:

```bash
# Run voice pipeline tests
python test.py --unit

# Or run specific test files
python tests/test_voice_pipeline.py
python tests/test_piper_transcription_pipeline.py
```

## Total Size

Approximately 4.4MB (25 WAV files) when fully generated.