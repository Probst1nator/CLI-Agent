#!/usr/bin/env python3
"""
Preload Whisper models script.

This script preloads Faster Whisper models to avoid timeout issues when they're requested
for the first time through the API. Run this script during container startup.
"""

import logging
import sys
import time
import os
from typing import Dict, List, Optional
import gc
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("preload-models")

try:
    import torch
    from py_classes.remote_host.services.whisper_service import WhisperService, optimize_torch_for_jetson
except ImportError:
    logger.error("Failed to import WhisperService. Please ensure you're running this script from the project root.")
    sys.exit(1)

def check_disk_space(path: str = "/root/.cache/whisper") -> float:
    """Check available disk space in GB at the specified path."""
    try:
        stats = shutil.disk_usage(path)
        free_gb = stats.free / (1024 * 1024 * 1024)
        logger.info(f"Available disk space: {free_gb:.2f}GB")
        return free_gb
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return 1000.0  # Return a large number to avoid blocking logic

def check_is_jetson() -> bool:
    """Check if running on a Jetson device."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            is_jetson = 'Jetson' in model
            if is_jetson:
                logger.info(f"Running on Jetson device: {model.strip()}")
            return is_jetson
    except:
        # Check using environment variables or other hints
        return os.environ.get('JETSON_OVERRIDE_DEVICES', '') != ''

def optimize_for_gpu():
    """Apply optimizations for running on GPU if available."""
    if not torch.cuda.is_available():
        return False

    try:
        # Set environment variables for better GPU memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
        
        # Set memory-efficient PyTorch options
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Apply Jetson-specific optimizations
        is_jetson = check_is_jetson()
        if is_jetson:
            logger.info("Applying Jetson-specific optimizations")
            optimize_torch_for_jetson()
        
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {gpu_name} with {gpu_mem:.2f}GB memory")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up GPU optimizations: {e}")
        return False

def preload_whisper_models(model_names: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Preload specific Whisper models or all models.
    
    Args:
        model_names: List of model names to preload, or None to preload all
        
    Returns:
        Dict: Mapping of model names to load success status
    """
    try:
        # Check disk space (models need ~4-6GB)
        free_space = check_disk_space()
        if free_space < 8.0:
            logger.warning(f"Low disk space ({free_space:.2f}GB). Model download may fail.")
        
        logger.info("Initializing WhisperService...")
        
        # Apply GPU optimizations first
        is_gpu = optimize_for_gpu()
        if is_gpu:
            logger.info("GPU optimizations applied")
            
            # Set lower memory limits on Jetson
            is_jetson = check_is_jetson()
            if is_jetson:
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.6)
                    logger.info("Set lower memory limits for Jetson")
        
        # Try creating service with more aggressive GC
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create service without preloading
        service = WhisperService(preload_models=False)
        
        if model_names is None:
            # Load all models
            logger.info("Preloading all Whisper models...")
            return service.download_all_models()
        else:
            # Load specific models with memory cleanup between each
            results = {}
            for model_name in model_names:
                logger.info(f"Preloading Whisper model: {model_name}")
                
                # Clear memory before loading each model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Use Jetson-specific loading for medium/large models if available
                success = False
                is_jetson = check_is_jetson() and torch.cuda.is_available()
                if is_jetson and model_name in ["medium", "large", "large-v2"]:
                    # Try Jetson-specific loading first
                    logger.info(f"Using Jetson-specific loading for {model_name}")
                    try:
                        if hasattr(service, '_load_model_jetson'):
                            success = service._load_model_jetson(model_name)
                        else:
                            success = service._load_model(model_name)
                    except Exception as e:
                        logger.error(f"Jetson-specific loading failed: {e}")
                        success = False
                else:
                    # Use standard loading
                    success = service._load_model(model_name)
                
                results[model_name] = success
                
                # If on GPU and this is not the last model, unload it to save memory
                if is_gpu and model_name != model_names[-1]:
                    if model_name in service.models:
                        logger.info(f"Temporarily unloading {model_name} after preloading to save memory")
                        del service.models[model_name]
                        gc.collect()
                        torch.cuda.empty_cache()
                
            return results
    except Exception as e:
        logger.error(f"Error preloading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def main():
    logger.info("Starting Whisper model preloading...")
    
    # For stability, especially in memory-constrained environments,
    # only preload tiny model, which is extremely lightweight
    models_to_preload = ["tiny"]
    
    # Only attempt to add a more capable model if we're confident we have the memory
    is_jetson = check_is_jetson()
    
    if torch.cuda.is_available():
        # Check available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Total GPU memory: {total_memory:.2f}GB")
        
        # Very conservative model selection based on available memory
        if not is_jetson and total_memory >= 16.0:
            # Only on large memory systems, preload small model too
            models_to_preload.append("small")
        elif not is_jetson and total_memory >= 10.0:
            # Only on large memory systems, preload base model too
            models_to_preload.append("base")
    else:
        # On CPU, add small model preloading if we're not on a Jetson
        if not is_jetson:
            models_to_preload.append("base")
    
    logger.info(f"Will preload the following models: {', '.join(models_to_preload)}")
    
    start_time = time.time()
    results = preload_whisper_models(models_to_preload)
    
    # Print results
    logger.info("Preload results:")
    for model_name, success in results.items():
        status = "Success" if success else "Failed"
        logger.info(f"  - {model_name}: {status}")
    
    logger.info(f"Whisper model preloading completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main() 