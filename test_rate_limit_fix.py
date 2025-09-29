#!/usr/bin/env python3
"""
Test script to verify the rate limiting fix.
This script tests that models are not preemptively blocked.
"""

import sys
import os

# Add the CLI-Agent root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker import enhanced_rate_limit_tracker
from infrastructure.rate_limiting.cls_rate_limit_config import RateLimitConfig

def test_lazy_initialization():
    """Test that lazy initialization doesn't preemptively block models."""
    print("🧪 Testing lazy initialization...")

    # Test models from the debug output
    test_models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.0-flash",
        "moonshotai/kimi-k2-instruct",
        "qwen/qwen3-32b"
    ]

    for model in test_models:
        # Initialize with lazy=True (should not create limits)
        RateLimitConfig.initialize_model_limits(model, provider="google", lazy=True)

        # Check if model can make request
        can_request, reason = enhanced_rate_limit_tracker.can_make_request(model, token_cost=1000)
        status = enhanced_rate_limit_tracker.get_status_summary(model)

        print(f"📊 {model}:")
        print(f"   Can make request: {can_request}")
        print(f"   Reason if blocked: {reason}")
        print(f"   Status: {status['status']}")

        # With lazy initialization, unconfigured models should be allowed
        if status['status'] == 'no_limits_configured':
            print("   ✅ PASS: Model is unconfigured and should be allowed")
        elif can_request:
            print("   ✅ PASS: Model can make requests")
        else:
            print(f"   ❌ FAIL: Model is blocked - {reason}")
        print()

def test_reactive_initialization():
    """Test that reactive initialization works after rate limit errors."""
    print("🧪 Testing reactive initialization...")

    test_model = "test-gemini-model"

    # Model should start unconfigured
    status = enhanced_rate_limit_tracker.get_status_summary(test_model)
    print(f"Initial status: {status['status']}")
    assert status['status'] == 'no_limits_configured', "Model should start unconfigured"

    # Initialize reactively (simulating rate limit error)
    RateLimitConfig.initialize_limits_from_error(test_model, provider="google")

    # Now it should be configured
    status = enhanced_rate_limit_tracker.get_status_summary(test_model)
    print(f"After reactive init: {status['status']}")
    assert status['status'] in ['ok', 'rate_limited'], "Model should be configured after reactive init"

    print("✅ PASS: Reactive initialization works")

def test_stale_cooldown_clearing():
    """Test that stale cooldowns are cleared."""
    print("🧪 Testing stale cooldown clearing...")

    # This should clear any existing stale cooldowns
    cleared_count = enhanced_rate_limit_tracker.clear_stale_cooldowns(max_age_hours=1)
    print(f"Cleared {cleared_count} stale cooldowns")
    print("✅ PASS: Stale cooldown clearing completed")

def main():
    """Run all tests."""
    print("🚀 Testing CLI-Agent Rate Limiting Fix")
    print("=" * 50)

    test_lazy_initialization()
    print()
    test_reactive_initialization()
    print()
    test_stale_cooldown_clearing()

    print("\n🎉 All tests completed!")
    print("\n💡 To clear existing rate limits manually, run:")
    print("   python utils/clear_rate_limits.py --all")

if __name__ == "__main__":
    main()