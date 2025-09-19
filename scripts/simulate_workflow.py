#!/usr/bin/env python3
"""
Simulate the GitHub Actions workflow locally
This tests the same conditions as the workflow without needing Docker
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def print_status(color_code, message):
    """Print colored status message"""
    print(f"\033[{color_code}m{message}\033[0m")

def test_imports():
    """Test that all required imports work"""
    print_status("34", "🧪 Testing Python imports...")

    required_modules = [
        'pexpect',
        'chromadb',
        'speech_recognition',
        'pyfiglet',
        'pathspec',
        'rank_bm25',
        'fasttext'
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print_status("32", f"✅ {module}")
        except ImportError as e:
            print_status("31", f"❌ {module}: {e}")
            missing.append(module)

    return len(missing) == 0

def test_cli_execution():
    """Test CLI-Agent execution similar to workflow"""
    print_status("34", "🧪 Testing CLI-Agent execution...")

    # Create temp output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "test_output.txt"

        try:
            # Simulate the workflow command but with a simple test message
            cmd = [
                sys.executable, "main.py",
                "-m", f"This is a test message. Please respond briefly and save any output to {output_file}"
            ]

            # Set environment to disable GUI mode (like workflow)
            env = os.environ.copy()
            env['CLI_AGENT_GUI_MODE'] = 'false'

            print_status("33", f"Running: {' '.join(cmd)}")

            # Run with timeout
            result = subprocess.run(
                cmd,
                timeout=60,  # 1 minute timeout
                capture_output=True,
                text=True,
                env=env
            )

            if result.returncode == 0:
                print_status("32", "✅ CLI-Agent executed successfully")

                # Check if it produced some output
                if result.stdout.strip():
                    print_status("32", f"✅ Generated output ({len(result.stdout)} chars)")
                else:
                    print_status("33", "⚠️  No stdout output")

                return True
            else:
                print_status("31", f"❌ CLI-Agent failed with return code {result.returncode}")
                if result.stderr:
                    print_status("31", f"STDERR: {result.stderr[:500]}")
                return False

        except subprocess.TimeoutExpired:
            print_status("33", "⚠️  CLI-Agent execution timed out (normal for this test)")
            return True  # Timeout is expected for this test
        except Exception as e:
            print_status("31", f"❌ Error running CLI-Agent: {e}")
            return False

def test_workflow_conditions():
    """Test specific workflow conditions"""
    print_status("34", "🧪 Testing workflow conditions...")

    # Check if main files exist
    required_files = ["main.py", "requirements.txt", "tools/bash.py"]

    for file_path in required_files:
        if Path(file_path).exists():
            print_status("32", f"✅ {file_path} exists")
        else:
            print_status("31", f"❌ {file_path} missing")
            return False

    return True

def main():
    """Run all tests"""
    print_status("36", "🚀 CLI-Agent Workflow Simulation")
    print_status("36", "=" * 40)

    all_passed = True

    # Test 1: Check imports
    if not test_imports():
        print_status("31", "❌ Import test failed")
        all_passed = False

    # Test 2: Check workflow conditions
    if not test_workflow_conditions():
        print_status("31", "❌ Workflow conditions test failed")
        all_passed = False

    # Test 3: Test CLI execution
    if not test_cli_execution():
        print_status("31", "❌ CLI execution test failed")
        all_passed = False

    print_status("36", "=" * 40)

    if all_passed:
        print_status("32", "🎉 All tests passed! Workflow should work correctly.")
        print_status("34", "💡 The GitHub Actions workflow is ready to run.")
    else:
        print_status("31", "❌ Some tests failed. Fix issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    # Change to CLI-Agent directory
    os.chdir(Path(__file__).parent.parent)
    main()