#!/usr/bin/env python3
"""
Demo script to test HuggingFace Local models.

This script demonstrates how to use the HuggingFace Local Provider
with any supported model.

WARNING: First run will download large model files to ~/.cli-agent/models/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.chat import Chat, Role
from core.providers.cls_huggingface_local_interface import HuggingFaceLocalClient
from core.providers.huggingface_local_models import get_model_key_from_id

# ============================================================================
# Configuration
# ============================================================================
# Model to test (change this to test different models)
TEST_MODEL_ID = "moonshotai/Moonlight-16B-A3B-Instruct"
TEST_MODEL_KEY = get_model_key_from_id(TEST_MODEL_ID)


def check_requirements():
    """Check if required packages are installed."""
    missing = []

    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")

    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  → CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  → CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  → CUDA not available, will use CPU (slower)")
    except ImportError:
        missing.append("torch")

    try:
        import huggingface_hub
        print(f"✓ huggingface_hub {huggingface_hub.__version__}")
    except ImportError:
        missing.append("huggingface_hub")

    try:
        import accelerate
        print(f"✓ accelerate {accelerate.__version__}")
    except ImportError:
        missing.append("accelerate")

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install transformers torch huggingface_hub accelerate")
        return False

    return True


def check_disk_space():
    """Check if enough disk space is available."""
    import shutil
    models_dir = HuggingFaceLocalClient.get_models_dir()

    # Check if model already downloaded
    if HuggingFaceLocalClient.is_model_downloaded(TEST_MODEL_ID):
        print(f"\n✓ {TEST_MODEL_KEY} model already downloaded!")
        return True

    stat = shutil.disk_usage(models_dir.parent)
    free_gb = stat.free / (1024**3)

    print(f"\nDisk space available: {free_gb:.1f} GB")
    if free_gb < 35:
        print(f"⚠ Warning: {TEST_MODEL_KEY} model requires ~32GB. You may not have enough space.")
        return False

    return True


def run_simple_test():
    """Run a simple test with the configured model."""
    print("\n" + "="*80)
    print(f"Testing {TEST_MODEL_KEY}")
    print("="*80)

    # Create a test chat
    chat = Chat()
    chat.add_message(Role.SYSTEM, "You are a helpful assistant.")
    chat.add_message(Role.USER, "What is 2+2? Answer in one sentence.")

    print("\nUser: What is 2+2? Answer in one sentence.")
    print(f"\n{TEST_MODEL_KEY}: ", end="", flush=True)

    try:
        # Generate response
        stream = HuggingFaceLocalClient.generate_response(
            chat,
            model_key=TEST_MODEL_KEY,
            temperature=0.7
        )

        # Collect and print response
        response = ""
        for chunk in stream:
            if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                token = chunk.message.content
                response += token
                print(token, end="", flush=True)

        print("\n\n" + "="*80)
        print(f"✅ SUCCESS! {TEST_MODEL_KEY} model is working correctly!")
        print("="*80)

        return True

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return False

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_interactive_mode():
    """Run interactive chat with configured model."""
    print("\n" + "="*80)
    print(f"Interactive Chat with {TEST_MODEL_KEY}")
    print("="*80)
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-"*80)

    chat = Chat()
    chat.add_message(Role.SYSTEM, "You are a helpful assistant.")

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'clear':
                chat = Chat()
                chat.add_message(Role.SYSTEM, "You are a helpful assistant.")
                print("\n[Conversation cleared]")
                continue

            if not user_input:
                continue

            # Add user message
            chat.add_message(Role.USER, user_input)

            # Generate response
            print(f"\n{TEST_MODEL_KEY}: ", end="", flush=True)

            stream = HuggingFaceLocalClient.generate_response(
                chat,
                model_key=TEST_MODEL_KEY,
                temperature=0.7
            )

            # Collect response
            response = ""
            for chunk in stream:
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    token = chunk.message.content
                    response += token
                    print(token, end="", flush=True)

            print()  # New line after response

            # Add assistant response to chat
            chat.add_message(Role.ASSISTANT, response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except Exception as e:
            print(f"\n\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break


def main():
    """Main entry point."""
    print("="*80)
    print(f"{TEST_MODEL_KEY} Test Script")
    print("="*80)
    print(f"Model ID: {TEST_MODEL_ID}")
    print("="*80)

    # Check requirements
    print("\nChecking requirements...")
    if not check_requirements():
        sys.exit(1)

    # Check disk space
    if not check_disk_space():
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Choose mode
    print("\nSelect mode:")
    print("1. Simple test (single question)")
    print("2. Interactive chat")
    print("3. Exit")

    choice = input("\nYour choice (1-3): ").strip()

    if choice == "1":
        run_simple_test()
    elif choice == "2":
        run_interactive_mode()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
