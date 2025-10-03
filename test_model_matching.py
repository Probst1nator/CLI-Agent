#!/usr/bin/env python3
"""
Test script to reproduce model matching error from the logs.

Error message:
No models found matching criteria: preferred_models: ['gemini-flash-lite-latest', 'gemini-flash-latest', 'gemini-flash-lite-latest', 'gemma3n:e4b'].
Total models available: 26 (gemini-flash-latest, gemini-2.5-pro, gemini-flash-lite-latest, gemini-2.0-flash, ...)
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from core.llm_router import LlmRouter
from core.chat import Chat, Role
from core.llm import Llm

async def test_model_matching():
    """Test model matching with the problematic preferred_models list."""

    print("=" * 80)
    print("Testing Model Matching Issue")
    print("=" * 80)

    # Create a simple chat context
    chat = Chat()
    chat.add_message(Role.USER, "test message")

    # Get all available models
    all_models = Llm.get_available_llms(include_dynamic=True)
    print(f"\n📊 Total models available: {len(all_models)}")
    print(f"   First 10: {[m.model_key for m in all_models[:10]]}")

    # The problematic preferred models from the log
    preferred_models = [
        'gemini-flash-lite-latest',
        'gemini-flash-latest',
        'gemini-flash-lite-latest',  # duplicate
        'gemma3n:e4b'
    ]

    print(f"\n🎯 Requested preferred models: {preferred_models}")

    # Check which ones exist
    print("\n🔍 Checking existence of each preferred model:")
    for model_name in preferred_models:
        exists = any(model_name in m.model_key for m in all_models)
        exact_match = any(model_name == m.model_key for m in all_models)
        print(f"   {model_name}:")
        print(f"      - Substring match: {exists}")
        print(f"      - Exact match: {exact_match}")
        if exists:
            matches = [m.model_key for m in all_models if model_name in m.model_key]
            print(f"      - Matching models: {matches}")

    # Try to get models using LlmRouter
    print("\n🔧 Testing LlmRouter.get_models()...")
    router = LlmRouter()

    try:
        # Try with force_preferred_model=False (default behavior)
        print("\n   Test 1: force_preferred_model=False")
        models = router.get_models(preferred_models=preferred_models, force_preferred_model=False)
        print(f"   ✅ Got {len(models)} models: {[m.model_key for m in models[:5]]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    try:
        # Try with force_preferred_model=True (forces only preferred models)
        print("\n   Test 2: force_preferred_model=True")
        models = router.get_models(preferred_models=preferred_models, force_preferred_model=True)
        print(f"   ✅ Got {len(models)} models: {[m.model_key for m in models[:5]]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Try actual generation (SKIPPED - requires async context and model availability)
    print("\n🚀 Testing actual generate_completion()... SKIPPED")
    print("   (Would require async context and working model providers)")

    # Now test the scenario from the logs - what if all models in retry_models fail?
    print("\n🔥 Testing failure scenario: What if retry_models is empty?")
    router.retry_models = []
    try:
        models = router.get_models(preferred_models=preferred_models, force_preferred_model=True)
        print(f"   ✅ Got {len(models)} models: {models}")
    except Exception as e:
        print(f"   ❌ Error with empty retry_models: {e}")

    # Reset retry_models and test with failed_models populated
    print("\n🔥 Testing failure scenario: What if all preferred models are in failed_models?")
    router.retry_models = router.get_models(preferred_models=preferred_models, force_preferred_model=False)
    router.failed_models = set(preferred_models)
    try:
        models = router.get_models(preferred_models=preferred_models, force_preferred_model=True)
        print(f"   Models returned: {[m.model_key for m in models] if models else 'None'}")
        if models:
            print(f"   ✅ Got {len(models)} models (failed_models not filtering them out)")
        else:
            print("   ❌ Got 0 models (failed_models filtered them all out)")
    except Exception as e:
        print(f"   ❌ Error with failed_models: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_model_matching())
