#!/usr/bin/env python3
"""
Utility to clear rate limit cooldowns for CLI-Agent models.
This can be used when models appear to be stuck in rate limited state.
"""

import os
import sys

# Add the CLI-Agent root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker import (
    enhanced_rate_limit_tracker,
)


def clear_all_rate_limits():
    """Clear all rate limit cooldowns for all models."""
    try:
        # Get all models with rate limits
        models_cleared = []

        for model_key in list(enhanced_rate_limit_tracker.model_limits.keys()):
            enhanced_rate_limit_tracker.clear_all_cooldowns(model_key)
            models_cleared.append(model_key)

        print(f"✅ Cleared rate limit cooldowns for {len(models_cleared)} models:")
        for model in models_cleared:
            print(f"   - {model}")

        return True
    except Exception as e:
        print(f"❌ Error clearing rate limits: {e}")
        return False

def clear_specific_models(model_keys: list[str]):
    """Clear rate limit cooldowns for specific models."""
    try:
        cleared = []
        not_found = []

        for model_key in model_keys:
            if model_key in enhanced_rate_limit_tracker.model_limits:
                enhanced_rate_limit_tracker.clear_all_cooldowns(model_key)
                cleared.append(model_key)
            else:
                not_found.append(model_key)

        if cleared:
            print(f"✅ Cleared rate limit cooldowns for {len(cleared)} models:")
            for model in cleared:
                print(f"   - {model}")

        if not_found:
            print(f"⚠️  Models not found in rate limits: {', '.join(not_found)}")

        return len(cleared) > 0
    except Exception as e:
        print(f"❌ Error clearing specific rate limits: {e}")
        return False

def clear_stale_cooldowns(max_age_hours: int = 6):
    """Clear stale rate limit cooldowns."""
    try:
        cleared_count = enhanced_rate_limit_tracker.clear_stale_cooldowns(max_age_hours)
        print(f"✅ Cleared {cleared_count} stale rate limit cooldowns (older than {max_age_hours} hours)")
        return True
    except Exception as e:
        print(f"❌ Error clearing stale cooldowns: {e}")
        return False

def show_rate_limit_status():
    """Show current rate limit status for all models."""
    try:
        print("📊 Current Rate Limit Status:")
        print("=" * 60)

        if not enhanced_rate_limit_tracker.model_limits:
            print("No rate limits configured for any models.")
            return

        for model_key in enhanced_rate_limit_tracker.model_limits.keys():
            status = enhanced_rate_limit_tracker.get_status_summary(model_key)
            print(f"\n🤖 {model_key}:")
            print(f"   Status: {status['status']}")
            print(f"   Retry count: {status['retry_count']}")

            if 'limits' in status:
                for limit_type, limit_info in status['limits'].items():
                    if limit_info['is_rate_limited']:
                        print(f"   🚫 {limit_type}: RATE LIMITED ({limit_info['cooldown_remaining']:.1f}s remaining)")
                    else:
                        print(f"   ✅ {limit_type}: {limit_info['usage']}/{limit_info['limit']} (OK)")

    except Exception as e:
        print(f"❌ Error showing status: {e}")

def main():
    """Main CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clear rate limit cooldowns for CLI-Agent models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_rate_limits.py --all                    # Clear all rate limits
  python clear_rate_limits.py --stale                  # Clear stale cooldowns (6+ hours old)
  python clear_rate_limits.py --status                 # Show current status
  python clear_rate_limits.py --models gemini-2.5-flash gemini-2.5-pro  # Clear specific models
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Clear all rate limit cooldowns')
    parser.add_argument('--stale', action='store_true',
                       help='Clear stale cooldowns (default: 6+ hours old)')
    parser.add_argument('--stale-hours', type=int, default=6,
                       help='Hours threshold for stale cooldowns (default: 6)')
    parser.add_argument('--status', action='store_true',
                       help='Show current rate limit status')
    parser.add_argument('--models', nargs='+', metavar='MODEL',
                       help='Clear cooldowns for specific models')

    args = parser.parse_args()

    # If no arguments provided, show help and status
    if not any([args.all, args.stale, args.status, args.models]):
        parser.print_help()
        print("\n")
        show_rate_limit_status()
        return

    if args.status:
        show_rate_limit_status()

    if args.all:
        print("🧹 Clearing all rate limit cooldowns...")
        clear_all_rate_limits()

    if args.stale:
        print(f"🧹 Clearing stale rate limit cooldowns (older than {args.stale_hours} hours)...")
        clear_stale_cooldowns(args.stale_hours)

    if args.models:
        print("🧹 Clearing rate limit cooldowns for specific models...")
        clear_specific_models(args.models)

if __name__ == "__main__":
    main()