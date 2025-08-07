"""
Whisper Transcription Service.

This module provides a service for transcribing audio using Faster Whisper, 
an optimized implementation of OpenAI's Whisper model.
"""

import logging
import os
import base64
import time
from typing import Dict, Any
import numpy as np
import warnings
import torch
import subprocess
import gc
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Optimize PyTorch for Jetson
def optimize_torch_for_jetson():
    """Apply optimizations specific to Jetson devices."""
    try:
        # Only run on Jetson
        is_jetson = False
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                is_jetson = 'Jetson' in model
        except:
            pass
            
        if not is_jetson:
            return False
            
        # Set environment variables for optimal performance
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        
        # Run garbage collection
        gc.collect()
        
        # More aggressive memory settings
        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Set TF32 on for tensor cores
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            
            # Defer CUDA initialization where possible
            torch.utils.cuda._lazy_init()
        
        return True
    except Exception as e:
        logger.warning(f"Error optimizing for Jetson: {e}")
        return False

# Try to optimize for Jetson
optimize_torch_for_jetson()

class WhisperService:
    """
    Service for transcribing audio using Faster Whisper models.
    
    This service provides methods for:
    - Loading Whisper models
    - Transcribing audio data
    - Handling different model sizes
    """
    
    def __init__(self, preload_models: bool = False):
        """
        Initialize the WhisperService and optionally preload all models.
        
        Args:
            preload_models: If True, download and load all available models at initialization
        """
        self.models = {}
        
        # Check for GPU availability
        self.has_nvidia_gpu = self._check_nvidia_gpu()
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        
        # Check CTranslate2 CUDA support explicitly
        self.ctranslate2_cuda_available = False
        try:
            import ctranslate2
            cuda_count = ctranslate2.get_cuda_device_count()
            self.ctranslate2_cuda_available = cuda_count > 0
            logger.info(f"CTranslate2 CUDA device count: {cuda_count}")
            
            # Print CUDA environment variables for debugging
            relevant_vars = ["CUDA_HOME", "CUDA_VISIBLE_DEVICES", "CUDA_PATH", "CT2_USE_CUDA", "LD_LIBRARY_PATH"]
            env_info = {var: os.environ.get(var, "Not set") for var in relevant_vars}
            logger.info(f"CUDA environment variables: {env_info}")
        except Exception as e:
            logger.warning(f"Error checking CTranslate2 CUDA support: {e}")
            self.ctranslate2_cuda_available = False
        
        # Check if we're on a Jetson device
        self.is_jetson = self._check_is_jetson()
        
        # Set memory-efficient options for PyTorch if using CUDA
        if self.device == "cuda":
            # Get available GPU memory and log it
            self._log_gpu_memory("Before initialization")
            
            # Set PyTorch to maximize compatibility over speed
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Set memory allocation to be more conservative on Jetson
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # For Jetson, reserve 40% memory for system
                memory_fraction = 0.6 if self.is_jetson else 0.8
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"Set GPU memory fraction to {memory_fraction}")
            
            # Empty cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log GPU name and memory
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"WhisperService initialized with GPU: {gpu_name}")
            self._log_gpu_memory("After initialization")
        elif self.has_nvidia_gpu and not self.cuda_available:
            logger.warning("NVIDIA GPU detected but CUDA is not available to PyTorch. "
                          "Check CUDA installation and PyTorch CUDA compatibility.")
        else:
            logger.info(f"WhisperService initialized with device: {self.device}")
        
        # List of all available Whisper models
        self.available_model_names = ["tiny", "base", "small", "medium", "large", "large-v2"]
        
        # Download and load all models if requested
        if preload_models:
            self.download_all_models()
        else:
            # At least preload the medium model by default to avoid timeouts
            # Only do this if on CPU or if enough memory is available
            if self.device == "cpu" or self._has_enough_memory_for("medium"):
                self._load_model("medium")
            else:
                logger.warning("Not enough GPU memory for preloading medium model. Will load on demand.")
                # Try with small model instead
                if self._has_enough_memory_for("small"):
                    logger.info("Loading small model instead as a fallback")
                    self._load_model("small")
    
    def _check_is_jetson(self) -> bool:
        """Check if we're running on a Jetson device."""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                is_jetson = 'Jetson' in model
                if is_jetson:
                    logger.info(f"Running on Jetson device: {model.strip()}")
                return is_jetson
        except:
            # Check using environment variable
            return 'JETSON_OVERRIDE_DEVICES' in os.environ
    
    def _log_gpu_memory(self, label: str = ""):
        """Log GPU memory usage with a label."""
        if not self.cuda_available:
            return
            
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            free = total_memory - reserved
            
            logger.info(f"GPU Memory ({label}): "
                       f"Total: {total_memory:.2f}GB, "
                       f"Reserved: {reserved:.2f}GB, "
                       f"Allocated: {allocated:.2f}GB, "
                       f"Free: {free:.2f}GB")
        except Exception as e:
            logger.warning(f"Failed to log GPU memory: {e}")
    
    def _has_enough_memory_for(self, model_name: str) -> bool:
        """
        Check if enough GPU memory is available for a specific model.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if there's likely enough memory
        """
        if not self.cuda_available:
            return True
            
        try:
            # Memory requirements in GB (approximate)
            # Increased for Jetson devices which need more headroom
            model_memory_gb = {
                "tiny": 1.0 if not self.is_jetson else 1.5,
                "base": 1.5 if not self.is_jetson else 2.0,
                "small": 3.0 if not self.is_jetson else 3.5,
                "medium": 5.5 if not self.is_jetson else 6.5,
                "large": 10.0 if not self.is_jetson else 12.0,
                "large-v2": 10.0 if not self.is_jetson else 12.0
            }
            
            # Get free memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            free_memory = total_memory - reserved
            
            # Add buffer for runtime operations (20%)
            required_memory = model_memory_gb.get(model_name, 5) * 1.2
            
            # Check if enough memory is available
            has_enough = free_memory >= required_memory
            if not has_enough:
                logger.warning(f"Insufficient GPU memory for model '{model_name}'. "
                              f"Required: {required_memory:.2f}GB, Available: {free_memory:.2f}GB")
            
            return has_enough
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}")
            return True  # Default to True if can't check
    
    def _check_nvidia_gpu(self) -> bool:
        """
        Check if an NVIDIA GPU is present on the system using nvidia-smi.
        
        Returns:
            bool: True if NVIDIA GPU is detected
        """
        try:
            subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def download_all_models(self) -> Dict[str, bool]:
        """
        Download all Whisper models that aren't already downloaded.
        
        Returns:
            Dict: Mapping of model names to download success status
        """
        results = {}
        for model_name in self.available_model_names:
            logger.info(f"Checking/downloading Whisper model: {model_name}")
            success = self._download_model(model_name)
            results[model_name] = success
            
        return results
    
    def _download_model(self, model_name: str) -> bool:
        """
        Download a Whisper model if not already downloaded.
        
        Args:
            model_name: Name of the Whisper model to download
            
        Returns:
            bool: True if model was successfully downloaded or already exists
        """
        try:
            # This will download the model if it doesn't exist
            # WhisperModel automatically downloads models if they don't exist
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Download to CPU to avoid OOM issues
                download_device = "cpu"
                compute_type = "int8" if self.is_jetson else "float16" 
                WhisperModel(model_name, device=download_device, compute_type=compute_type, download_root=None)
            logger.info(f"Successfully downloaded/verified Whisper model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to download Whisper model {model_name}: {e}")
            return False
            
    def _load_model(self, model_name: str = "large-v2") -> bool:
        """
        Load a Whisper model if not already loaded.
        
        Args:
            model_name: Name of the Whisper model to load (tiny, base, small, medium, large, large-v2)
            
        Returns:
            bool: True if model was successfully loaded or already exists
        """
        if model_name in self.models:
            return True
        
        # First check if we have enough memory if using CUDA
        if self.device == "cuda" and not self._has_enough_memory_for(model_name):
            logger.warning(f"Not enough GPU memory for model {model_name}, falling back to CPU")
            loading_device = "cpu"
        else:
            loading_device = self.device
        
        # Special handling for Jetson
        if self.is_jetson and loading_device == "cuda":
            logger.info("Using Jetson-specific loading optimizations")
            
            # Always clean before loading
            torch.cuda.empty_cache()
            gc.collect()
            
            # Try loading model in chunks if it's medium or larger
            if model_name in ["medium", "large", "large-v2"]:
                return self._load_model_jetson(model_name)
            
        try:
            logger.info(f"Loading Whisper model: {model_name} on {loading_device}")
            
            # Clean up memory before loading
            if self.device == "cuda":
                self._log_gpu_memory("Before model load")
                torch.cuda.empty_cache()
                gc.collect()
            
            compute_type = "int8" if self.is_jetson else "float16" if loading_device == "cuda" else "float32"
            
            # Check if CTranslate2 can use CUDA - use our pre-computed flag for consistency
            # If CUDA is not available for CTranslate2 but we're trying to use it, force CPU
            if loading_device == "cuda" and not self.ctranslate2_cuda_available:
                logger.warning("CTranslate2 was not compiled with CUDA support, falling back to CPU")
                logger.warning("Please rebuild the Docker image to ensure proper CUDA support")
                loading_device = "cpu"
                compute_type = "float32"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Try to load with memory-efficient options
                self.models[model_name] = WhisperModel(
                    model_name,
                    device=loading_device,
                    compute_type=compute_type,
                    download_root=None
                )
            
            if self.device == "cuda":
                self._log_gpu_memory("After model load")
            
            # Verify the device the model is actually using
            model_device = "unknown"
            if hasattr(self.models[model_name], 'device'):
                model_device = self.models[model_name].device
            logger.info(f"Successfully loaded Whisper model: {model_name} on {loading_device} (actual: {model_device})")
            
            # Double-check that CTranslate2 CUDA is working if we requested CUDA
            if loading_device == "cuda":
                try:
                    import ctranslate2
                    cuda_count = ctranslate2.get_cuda_device_count()
                    if cuda_count > 0:
                        logger.info(f"CTranslate2 confirms CUDA is available with {cuda_count} devices")
                    else:
                        logger.warning("CTranslate2 still reports no CUDA devices available even though we requested CUDA")
                except Exception as e:
                    logger.warning(f"Could not verify CTranslate2 CUDA devices: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model {model_name}: {e}")
            # If CUDA failed, try CPU as fallback
            if loading_device == "cuda":
                logger.warning(f"Attempting to load model {model_name} on CPU as fallback")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.models[model_name] = WhisperModel(
                            model_name,
                            device="cpu",
                            compute_type="float32"
                        )
                    logger.info(f"Successfully loaded Whisper model: {model_name} on CPU")
                    return True
                except Exception as cpu_e:
                    logger.error(f"CPU fallback also failed for model {model_name}: {cpu_e}")
            return False
    
    def _load_model_jetson(self, model_name: str) -> bool:
        """
        Specialized model loading routine for Jetson devices.
        Uses int8 quantization to reduce memory pressure.
        
        Args:
            model_name: Name of the Whisper model to load
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Using Jetson-optimized loading for {model_name}")
            
            # Force garbage collection before loading
            gc.collect()
            torch.cuda.empty_cache()
            self._log_gpu_memory("Before model load")
            
            # Determine device based on CUDA support - use class-level detection
            device = "cuda" if self.ctranslate2_cuda_available else "cpu"
            compute_type = "int8" if device == "cuda" else "float32"
            
            if device != "cuda":
                logger.warning("CTranslate2 was not compiled with CUDA support, using CPU instead")
                logger.warning("Performance will be significantly slower. Please rebuild the Docker image.")
            else:
                logger.info("CTranslate2 CUDA support confirmed, using GPU acceleration")
            
            # Use int8 quantization on Jetson for improved memory usage
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models[model_name] = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=None
                )
            
            # Verify the device the model is actually using
            model_device = "unknown"
            if hasattr(self.models[model_name], 'device'):
                model_device = self.models[model_name].device
            logger.info(f"Successfully loaded Whisper model: {model_name} on {device} (actual: {model_device})")
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
            self._log_gpu_memory("After model load")
            
            return True
        except Exception as e:
            logger.error(f"Error in Jetson-optimized loading: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available Whisper models.
        
        Returns:
            Dict: Information about available models
        """
        model_info = {
            "tiny": {"params": "39M", "description": "Fastest, lowest accuracy"},
            "base": {"params": "74M", "description": "Fast with basic accuracy"},
            "small": {"params": "244M", "description": "Good balance of speed and accuracy"},
            "medium": {"params": "769M", "description": "High accuracy, moderate speed"},
            "large": {"params": "1550M", "description": "Highest accuracy, slowest"},
            "large-v2": {"params": "1550M", "description": "Improved version of large"}
        }
        
        # Add loading status to each model
        for model_name in model_info:
            model_info[model_name]["loaded"] = model_name in self.models
            if model_name in self.models and hasattr(self.models[model_name], 'device'):
                model_info[model_name]["device"] = str(self.models[model_name].device)
            
        return model_info
        
    def transcribe_audio(self, audio_data_b64: str, sample_rate: int, 
                        model_name: str = "large-v2") -> Dict[str, Any]:
        """
        Transcribe audio data using the specified Whisper model.
        
        Args:
            audio_data_b64: Base64 encoded audio data
            sample_rate: Sample rate of the audio data
            model_name: Name of the Whisper model to use (default: large-v2)
            
        Returns:
            Dict: Transcription results
        """
        start_time = time.time()
        
        try:
            # Free up memory before transcription - more aggressive
            if self.device == "cuda":
                self._log_gpu_memory("Before transcription")
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Validate model name
            valid_models = ["tiny", "base", "small", "medium", "large", "large-v2"]
            if model_name not in valid_models:
                return {
                    "error": f"Invalid model name. Choose from: {', '.join(valid_models)}"
                }
            
            # Load model if not already loaded
            if not self._load_model(model_name):
                return {
                    "error": f"Failed to load model: {model_name}"
                }
            
            # Decode audio data
            audio_bytes = base64.b64decode(audio_data_b64)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Save audio to temporary file for Whisper
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_np, sample_rate)
            
            # Clear numpy array to free memory
            del audio_np
            gc.collect()
            
            # Perform transcription with memory-efficient settings
            logger.info(f"Transcribing audio with model: {model_name}")
            model = self.models[model_name]
            
            # Check model device
            model_device = model.device if hasattr(model, 'device') else "unknown"
            logger.info(f"Model {model_name} is on device: {model_device}")
            
            # Transcribe with memory-efficient options
            beam_size = 3 if self.device == "cuda" else 5
            
            # For smaller models (tiny, base), use a smaller beam size
            if model_name in ["tiny", "base"]:
                beam_size = 2
                
            # Check if onnxruntime is available for VAD filter
            use_vad = True
            try:
                import onnxruntime
                logger.info("ONNX Runtime is available, using VAD filter")
                
                # Disable VAD for larger models on memory-constrained systems
                # This helps prevent OOM errors especially on smaller devices
                total_memory = 0
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if (model_name in ["medium", "large", "large-v2"] and total_memory < 8) or \
                   (model_name in ["small"] and total_memory < 4):
                    logger.info(f"Limited memory detected ({total_memory:.2f}GB), disabling VAD filter for {model_name} model")
                    use_vad = False
            except ImportError:
                use_vad = False
                logger.warning("ONNX Runtime is not available, disabling VAD filter")
            
            # Additional memory cleanup before transcription
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Starting transcription with beam_size={beam_size}, vad_filter={use_vad}")
            
            # Transcribe using faster-whisper
            segments, info = model.transcribe(
                temp_path,
                beam_size=beam_size,
                vad_filter=use_vad
            )
            
            # Process segments
            segment_list = []
            transcription = ""
            
            # Convert segments iterator to list
            for segment in segments:
                segment_data = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                segment_list.append(segment_data)
                transcription += segment.text + " "
            
            transcription = transcription.strip()
            language = info.language if hasattr(info, 'language') else ""
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")
                
            # Free memory after transcription
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            self._log_gpu_memory("After transcription")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Format the response
            response = {
                "success": True,
                "transcription": transcription,
                "language": language,
                "processing_time": processing_time,
                "model": model_name,
                "device": str(model_device),
                "segments": segment_list
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }


if __name__ == "__main__":
    # Configure basic logging when script is run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Whisper model download...")
    
    # Initialize the service without preloading models
    service = WhisperService()
    download_results = service.download_all_models()
    
    # Print download results
    logger.info("Download results:")
    for model_name, success in download_results.items():
        status = "Success" if success else "Failed"
        logger.info(f"  - {model_name}: {status}")
    
    logger.info("Whisper model download completed.")


