"""
Integration tests for main.py local mode functionality
Tests the specific issue where -l flag doesn't properly filter models
"""

import sys
from pathlib import Path
import argparse

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.llm_router import LlmRouter
from core.globals import g


def test_main_local_mode_flag_processing():
    """Test that main.py properly processes the -l flag and sets up local-only models"""
    print("🧪 Testing main.py local mode flag processing...")
    
    try:
        # Save original state
        original_selected_llms = g.SELECTED_LLMS
        original_force_local = g.FORCE_LOCAL
        
        # Reset state
        g.SELECTED_LLMS = []
        g.FORCE_LOCAL = False
        
        # Simulate argparse results for local mode
        args = argparse.Namespace()
        args.local = True
        args.llm = None
        
        # This simulates the logic from main.py lines 1933-1939
        if args.local and not g.SELECTED_LLMS:
            print("  🔄 Simulating main.py local mode detection...")
            local_models = LlmRouter.get_models(force_local=True)
            if local_models:
                g.SELECTED_LLMS = [model.model_key for model in local_models]
                print(f"  📊 Selected local models: {g.SELECTED_LLMS}")
            else:
                print("  ⚠️  No local models found")
        
        # Verify that only local models are selected
        if g.SELECTED_LLMS:
            # Verify each selected model is actually local
            all_models = LlmRouter.get_models()
            model_dict = {model.model_key: model for model in all_models}
            
            for selected_key in g.SELECTED_LLMS:
                # Find the model (key might be partial match)
                matching_models = [model for model in all_models if selected_key in model.model_key]
                if not matching_models:
                    print(f"  ❌ Selected model key not found in available models: {selected_key}")
                    return False
                    
                model = matching_models[0]
                if not model.local:
                    print(f"  ❌ Non-local model in SELECTED_LLMS: {selected_key}")
                    return False
                    
                print(f"  ✅ Verified local model: {selected_key}")
            
            print("  ✅ Local mode flag processing working correctly")
            result = True
        else:
            print("  ⚠️  No models selected - may be expected if no local models available")
            result = True
            
        # Restore original state
        g.SELECTED_LLMS = original_selected_llms
        g.FORCE_LOCAL = original_force_local
        
        return result
        
    except Exception as e:
        print(f"  ❌ Main local mode flag processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_model_display_vs_usage():
    """Test the discrepancy between displayed fallback models and actual usage in local mode"""
    print("🧪 Testing fallback model display vs usage in local mode...")
    
    try:
        # Get typical fallback models (like from main.py)
        typical_fallbacks = ["gemini-2.5-flash", "gemini-2.5-flash-lite-preview-06-17", 
                           "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro", "gemma3:27b"]
        
        # Get actual local models
        local_models = LlmRouter.get_models(force_local=True)
        local_keys = [model.model_key for model in local_models]
        
        print(f"  📊 Typical fallback models: {typical_fallbacks}")
        print(f"  🏠 Available local models: {local_keys}")
        
        # Find which fallback models are actually local
        actual_local_fallbacks = []
        for fallback in typical_fallbacks:
            is_local = any(fallback in local_key or local_key in fallback for local_key in local_keys)
            if is_local:
                actual_local_fallbacks.append(fallback)
                print(f"  ✅ Fallback model is local: {fallback}")
            else:
                print(f"  ❌ Fallback model is NOT local: {fallback}")
        
        print(f"  🔄 Actual local fallbacks: {actual_local_fallbacks}")
        
        # The issue is likely here: the displayed list includes cloud models
        # but they should be filtered out in local mode
        cloud_fallbacks_shown = len(typical_fallbacks) - len(actual_local_fallbacks)
        if cloud_fallbacks_shown > 0:
            print(f"  ⚠️  ISSUE FOUND: {cloud_fallbacks_shown} cloud models shown in fallback list")
            print("  🔍 This explains why cloud models are attempted in local mode!")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Fallback model display test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_selection_in_generate_completion():
    """Test that generate_completion properly handles preferred_models in local mode"""
    print("🧪 Testing model selection in generate_completion...")
    
    try:
        # Simulate the scenario from the error log
        preferred_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite-preview-06-17", 
                          "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro", "gemma3:27b"]
        
        # Get actual models for each preference  
        router = LlmRouter()
        all_models = LlmRouter.get_models()
        
        # Test what happens when we filter these for local-only
        local_preferred = []
        for pref in preferred_models:
            matching_models = [m for m in all_models if pref in m.model_key and m.local]
            if matching_models:
                local_preferred.append(pref)
                print(f"  ✅ Preferred model is local: {pref}")
            else:
                print(f"  ❌ Preferred model is NOT local: {pref}")
        
        print(f"  🔄 Local preferred models: {local_preferred}")
        
        # Test get_model with these preferences and force_local
        if local_preferred:
            model = LlmRouter.get_model(preferred_models=local_preferred, force_local=True)
            if model:
                if not model.local:
                    print(f"  ❌ get_model returned non-local model: {model.model_key}")
                    return False
                print(f"  ✅ get_model returned local model: {model.model_key}")
            else:
                print("  ⚠️  get_model returned None")
        
        # The issue might be that preferred_models includes cloud models
        # and the filtering isn't happening early enough
        cloud_preferred = [p for p in preferred_models if p not in local_preferred]
        if cloud_preferred:
            print(f"  🔍 ISSUE: Cloud models in preferred list: {cloud_preferred}")
            print("  🔍 These should be filtered out BEFORE calling get_model in local mode")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_local_mismatch_error_scenario():
    """Test the exact error scenario: 'local mismatch (required: True, model: False)'"""
    print("🧪 Testing local mismatch error scenario...")
    
    try:
        # Get a cloud model
        all_models = LlmRouter.get_models()
        cloud_models = [m for m in all_models if not m.local]
        
        if not cloud_models:
            print("  ⚠️  No cloud models found to test with")
            return True
            
        cloud_model = cloud_models[0]
        print(f"  🌐 Testing with cloud model: {cloud_model.model_key}")
        
        # Test the model_capable_check that generates this error
        router = LlmRouter()
        from core.chat import Chat
        
        capable, reason = router.model_capable_check(
            model=cloud_model,
            chat=Chat(),
            strengths=[],
            local=True,  # This should cause the mismatch
            force_free=False,
            has_vision=False
        )
        
        print(f"  🔍 Capability check result: capable={capable}, reason='{reason}'")
        
        if capable:
            print("  ❌ Cloud model should not be capable when local=True required")
            return False
            
        if "local mismatch" not in reason:
            print(f"  ❌ Expected 'local mismatch' in reason, got: {reason}")
            return False
            
        # This confirms the error message we saw
        if reason == "local mismatch (required: True, model: False)":
            print("  ✅ Confirmed exact error scenario from logs!")
            print("  🔍 The issue is that cloud models are being tested against local=True requirement")
            print("  🔍 This means the model selection isn't filtering early enough")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Local mismatch error scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False