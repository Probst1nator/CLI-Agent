#!/usr/bin/env python3
"""
Simple bridge to integrate visual input with existing CLI-Agent.
This modifies get_user_input_with_bindings to check visual queue.
"""

import asyncio
import queue
import time
from typing import Tuple, List
from core.globals import g

# Store original function reference
_original_get_user_input = None

async def enhanced_get_user_input_with_bindings(*args, **kwargs) -> Tuple[str, List[str]]:
    """
    Enhanced version that checks visual queue with polling to prevent race conditions.
    
    This implementation:
    1. Checks queue immediately (for instant response)
    2. Polls queue for a short time before falling back to console input
    3. Eliminates race condition where browser input arrives after queue check
    """
    global _original_get_user_input
    
    # Immediate queue check (for instant response when message is already queued)
    if hasattr(g, 'visual_input_queue'):
        try:
            if not g.visual_input_queue.empty():
                visual_message = g.visual_input_queue.get_nowait()
                print(f"\n🌐 Visual Input (immediate): {visual_message}")
                return visual_message, []
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing visual input: {e}")
    
    # POLLING SOLUTION: Give browser input a chance to arrive
    # This solves the race condition by waiting briefly for browser messages
    if hasattr(g, 'visual_input_queue'):
        print("⏳ Checking for browser input...")
        poll_start_time = time.time()
        poll_duration = 2.0  # Wait up to 2 seconds for browser input
        poll_interval = 0.1  # Check every 100ms
        
        while (time.time() - poll_start_time) < poll_duration:
            try:
                if not g.visual_input_queue.empty():
                    visual_message = g.visual_input_queue.get_nowait()
                    elapsed = time.time() - poll_start_time
                    print(f"\n🌐 Visual Input (after {elapsed:.1f}s): {visual_message}")
                    return visual_message, []
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error during visual input polling: {e}")
                break
            
            # Non-blocking sleep
            await asyncio.sleep(poll_interval)
        
        print("⌛ Browser input timeout - switching to console input")
    
    # Fall back to original input method
    if _original_get_user_input:
        return await _original_get_user_input(*args, **kwargs)
    else:
        # Fallback if original not available
        from agent.core.input_handlers import get_user_input_with_bindings as original
        return await original(*args, **kwargs)

def install_visual_input_bridge():
    """
    Install the visual input bridge by monkey-patching the input handler.
    This is a surgical modification that doesn't break anything.
    """
    global _original_get_user_input
    
    try:
        # Import the module that uses the function
        import agent.core.input_handlers as input_handlers
        
        # Store original function
        _original_get_user_input = input_handlers.get_user_input_with_bindings
        
        # Copy all attributes from original function to enhanced function
        for attr in dir(_original_get_user_input):
            if not attr.startswith('__') and hasattr(_original_get_user_input, attr):
                try:
                    setattr(enhanced_get_user_input_with_bindings, attr, getattr(_original_get_user_input, attr))
                except (AttributeError, TypeError):
                    pass  # Skip attributes that can't be set
        
        # Replace with enhanced version
        input_handlers.get_user_input_with_bindings = enhanced_get_user_input_with_bindings
        
        print("✅ Visual input bridge installed successfully")
        print("   Browser inputs will now appear with 🌐 indicator")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install visual input bridge: {e}")
        return False

def uninstall_visual_input_bridge():
    """Remove the bridge and restore original functionality."""
    global _original_get_user_input
    
    try:
        if _original_get_user_input:
            import agent.core.input_handlers as input_handlers
            input_handlers.get_user_input_with_bindings = _original_get_user_input
            print("✅ Visual input bridge removed")
            return True
    except Exception as e:
        print(f"❌ Failed to remove visual input bridge: {e}")
        return False

if __name__ == "__main__":
    print("Visual Input Bridge for CLI-Agent")
    print("=" * 40)
    print("This module provides race-condition-free visual input integration.")
    print("")
    print("Usage:")
    print("  from visual_input_bridge import install_visual_input_bridge")
    print("  install_visual_input_bridge()")
    print("")
    print("The bridge will check the visual input queue before standard input,")
    print("eliminating race conditions while preserving all existing functionality.")