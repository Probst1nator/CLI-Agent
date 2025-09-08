# CLI-Agent Testing Framework

This directory contains the unified testing framework for CLI-Agent.

## Quick Start

From the project root, run:

```bash
# Essential tests (recommended)
python test.py

# Quick tests only (< 30s)
python test.py --quick

# Complete test suite
python test.py --full

# Autonomous end-to-end tests
python test.py --autonomous
```

## Test Structure

- **Integration Tests (Level 0)**: End-to-end autonomous testing where CLI-Agent actually creates files and runs code
- **Tool Tests (Level 1)**: Individual tool functionality validation
- **Core Tests (Level 2)**: Core infrastructure and imports (default level)
- **Shared Tests (Level 3)**: Shared utilities validation
- **Unit Tests (Level 4)**: Detailed component testing

## Files

- `test.py` - Main unified test runner (use this!)
- `run_tests.py` - Advanced test runner with hierarchical levels
- `archive/` - Old standalone test files (kept for reference)

## Legacy Note

The `archive/` directory contains the old standalone test files that have been consolidated into the unified framework:
- `test_autonomous.py` - Now integrated as Integration level tests
- `test_secure_autonomous.py` - Security features integrated
- `test_quick.py` - Now available via `--quick` flag
- `test_debug.py` - Debug capabilities built into verbose mode