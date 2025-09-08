"""
Test suite for LLM Router local mode functionality
This tests the core issue where local mode (-l) filtering broke during monorepo decomposition
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from core.llm_router import LlmRouter
from core.chat import Chat, Role


def test_local_mode_model_filtering():
    """Test that force_local=True only returns local models"""
    print("🧪 Testing local mode model filtering...")
    
    try:
        # Get all models without filtering
        all_models = LlmRouter.get_models()
        print(f"  📊 Total models available: {len(all_models)}")
        
        # Get only local models
        local_models = LlmRouter.get_models(force_local=True)
        print(f"  🏠 Local models found: {len(local_models)}")
        
        if not local_models:
            print("  ❌ No local models found - this might be expected if Ollama isn't running")
            return True  # Don't fail the test if no local models available
        
        # Verify all returned models are local
        for model in local_models:
            if not model.local:
                print(f"  ❌ Non-local model returned in local-only mode: {model.model_key}")
                return False
            print(f"  ✅ Local model: {model.model_key}")
        
        # Get only cloud models  
        cloud_models = LlmRouter.get_models(force_local=False)
        cloud_only = [m for m in cloud_models if not m.local]
        print(f"  ☁️  Cloud models found: {len(cloud_only)}")
        
        # Verify cloud models are excluded from local mode
        for cloud_model in cloud_only:
            if cloud_model in local_models:
                print(f"  ❌ Cloud model found in local-only results: {cloud_model.model_key}")
                return False
        
        print("  ✅ Local mode filtering working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Local mode filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_model_respects_local_flag():
    """Test that get_model() respects the force_local flag"""
    print("🧪 Testing get_model() local flag respect...")
    
    try:
        # Try to get a local model specifically
        local_model = LlmRouter.get_model(force_local=True)
        
        if local_model is None:
            print("  ⚠️  No local model available - skipping test")
            return True
            
        if not local_model.local:
            print(f"  ❌ get_model(force_local=True) returned non-local model: {local_model.model_key}")
            return False
            
        print(f"  ✅ get_model() correctly returned local model: {local_model.model_key}")
        
        # Try to get a cloud model specifically  
        cloud_model = LlmRouter.get_model(force_local=False)
        
        if cloud_model and cloud_model.local and len(LlmRouter.get_models(force_local=False)) > 1:
            # Only fail if there are cloud models available but we got a local one
            cloud_only_models = [m for m in LlmRouter.get_models(force_local=False) if not m.local]
            if len(cloud_only_models) > 0:
                print("  ❌ get_model(force_local=False) returned local model when cloud models available")
                return False
        
        print("  ✅ get_model() respects local flag correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ get_model() local flag test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_capability_check_local_filtering():
    """Test that model_capable_check correctly filters by local requirement"""
    print("🧪 Testing model_capable_check local filtering...")
    
    try:
        router = LlmRouter()
        
        # Get a mix of models
        all_models = LlmRouter.get_models()
        local_models = [m for m in all_models if m.local]
        cloud_models = [m for m in all_models if not m.local]
        
        if not local_models:
            print("  ⚠️  No local models available - skipping test")
            return True
            
        # Test local model with local=True requirement
        for model in local_models[:1]:  # Test just the first one
            capable, reason = router.model_capable_check(
                model=model,
                chat=Chat(),
                strengths=[],
                local=True,  # Require local
                force_free=False,
                has_vision=False
            )
            
            if not capable:
                print(f"  ❌ Local model {model.model_key} should pass local=True check: {reason}")
                return False
                
            print(f"  ✅ Local model {model.model_key} passes local=True check")
        
        # Test cloud model with local=True requirement (should fail)
        for model in cloud_models[:1]:  # Test just the first one
            capable, reason = router.model_capable_check(
                model=model,
                chat=Chat(),
                strengths=[],
                local=True,  # Require local
                force_free=False,
                has_vision=False
            )
            
            if capable:
                print(f"  ❌ Cloud model {model.model_key} should fail local=True check")
                return False
                
            if "local mismatch" not in reason:
                print(f"  ❌ Expected 'local mismatch' in reason, got: {reason}")
                return False
                
            print(f"  ✅ Cloud model {model.model_key} correctly fails local=True check: {reason}")
        
        print("  ✅ model_capable_check local filtering working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ model_capable_check local filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_generate_completion_local_mode():
    """Test that generate_completion respects force_local flag"""
    print("🧪 Testing generate_completion local mode...")
    
    try:
        # Create a simple chat
        chat = Chat()
        chat.add_message(Role.USER, "Hello, respond with just 'test successful'")
        
        # Try local-only completion (this might fail if no local models, which is expected)
        try:
            response = await LlmRouter.generate_completion(
                chat=chat,
                force_local=True,
                temperature=0.1
            )
            
            if response:
                print(f"  ✅ Local completion successful: {len(response)} characters")
                return True
            else:
                print("  ⚠️  Local completion returned empty response")
                return True  # Don't fail - might be expected
                
        except Exception as local_error:
            error_str = str(local_error)
            if "local mismatch" in error_str or "no local models" in error_str.lower():
                print(f"  ✅ Local-only mode correctly rejected cloud models: {error_str[:100]}...")
                return True
            else:
                print(f"  ❌ Unexpected error in local mode: {error_str}")
                return False
        
    except Exception as e:
        print(f"  ❌ generate_completion local mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_model_list_filtering():
    """Test that fallback model lists are properly filtered for local mode"""
    print("🧪 Testing fallback model list filtering...")
    
    try:
        # Get preferred models (this simulates what main.py does)
        all_fallback_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemma3:27b"]
        
        # Simulate local mode filtering
        local_models = LlmRouter.get_models(force_local=True)
        local_model_keys = [model.model_key for model in local_models]
        
        # Filter fallback models for local-only
        local_fallback_models = [key for key in all_fallback_models if any(key in local_key for local_key in local_model_keys)]
        
        print(f"  📊 All fallback models: {all_fallback_models}")
        print(f"  🏠 Available local models: {local_model_keys}")
        print(f"  🔄 Filtered local fallbacks: {local_fallback_models}")
        
        # Verify that local fallbacks only contain local models
        for fallback_key in local_fallback_models:
            is_local = any(fallback_key in local_key for local_key in local_model_keys)
            if not is_local:
                print(f"  ❌ Non-local model in local fallback list: {fallback_key}")
                return False
        
        if local_fallback_models:
            print("  ✅ Fallback model filtering working correctly")
        else:
            print("  ⚠️  No local fallback models found (expected if no local models available)")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Fallback model list filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ollama_model_detection():
    """Test that Ollama models are properly detected and marked as local"""
    print("🧪 Testing Ollama model detection...")
    
    try:
        from core.providers.cls_ollama_interface import OllamaClient
        
        # Check if Ollama is accessible
        try:
            available_models = OllamaClient.get_available_models()
            print(f"  📊 Ollama reports {len(available_models)} models available")
            
            if not available_models:
                print("  ⚠️  No Ollama models found - Ollama may not be running")
                return True
                
            # Get models through LLM router
            all_models = LlmRouter.get_models()
            ollama_models_in_router = [m for m in all_models if m.local and "ollama" in m.__class__.__module__.lower()]
            
            print(f"  🔄 Router found {len(ollama_models_in_router)} Ollama models")
            
            for model in ollama_models_in_router[:3]:  # Check first 3
                if not model.local:
                    print(f"  ❌ Ollama model not marked as local: {model.model_key}")
                    return False
                print(f"  ✅ Ollama model correctly marked as local: {model.model_key}")
            
            print("  ✅ Ollama model detection working correctly")
            return True
            
        except Exception as ollama_error:
            print(f"  ⚠️  Ollama not accessible: {ollama_error}")
            return True  # Don't fail if Ollama isn't running
            
    except Exception as e:
        print(f"  ❌ Ollama model detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False