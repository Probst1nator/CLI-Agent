"""
CLI-Agent Remote Host Client.

This module provides client functionality for interacting with the CLI-Agent Remote Host server.
"""

import logging
import os
import base64
import requests
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
from dotenv import load_dotenv
from termcolor import colored

logger = logging.getLogger(__name__)

class RemoteHostClient:
    """
    Client for interacting with the CLI-Agent Remote Host server.
    
    This client provides methods for:
    - Checking server health and available services
    - Wake word detection
    - Whisper transcription
    - (Future services will be added here)
    """
    
    def __init__(self, server_url: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            server_url: URL of the remote host server.
                       If None, will try to get from CLI_AGENT_REMOTE_HOST env var,
                       then fall back to http://localhost:5000.
        """
        # Load environment variables
        load_dotenv()
        
        # Use provided URL or get from environment with fallback
        self.server_url = server_url or os.environ.get("CLI_AGENT_REMOTE_HOST", "http://localhost:5000")
        
        # Strip trailing slash if present
        if self.server_url.endswith("/"):
            self.server_url = self.server_url[:-1]
            
        # Set up endpoints
        self.health_endpoint = f"{self.server_url}/health"
        self.services_endpoint = f"{self.server_url}/services"
        self.wake_word_endpoint = f"{self.server_url}/wake_word/detect"
        self.whisper_transcribe_endpoint = f"{self.server_url}/whisper/transcribe"
        self.whisper_models_endpoint = f"{self.server_url}/whisper/models"
        
        # Cache of available services
        self._available_services: Optional[Dict[str, bool]] = None
        
        print(f"Remote: <{colored('Client', 'green')}> initialized with server URL: {colored(self.server_url, 'blue')}")
        
    def check_server_health(self) -> Dict[str, Any]:
        """
        Check if the remote server is healthy and get available services.
        
        Returns:
            Dict with server health information or empty dict if server is unreachable
        """
        try:
            response = requests.get(
                self.health_endpoint, 
                timeout=2.0
            )
            
            if response.status_code == 200:
                result = response.json()
                # Cache available services
                if "services" in result:
                    self._available_services = result["services"]
                return result
            else:
                logger.error(f"Health check failed with status {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error checking server health: {e}")
            return {}
            
    def is_service_available(self, service_name: str) -> bool:
        """
        Check if a specific service is available.
        
        Args:
            service_name: The name of the service to check
            
        Returns:
            bool: True if the service is available, False otherwise
        """
        # Use cached services if available
        if self._available_services is not None:
            return self._available_services.get(service_name, False)
            
        # Otherwise, check health to get services
        health_info = self.check_server_health()
        services = health_info.get("services", {})
        
        return services.get(service_name, False)
    
    def list_services(self) -> Dict[str, Any]:
        """
        Get a list of available services and their documentation.
        
        Returns:
            Dict with services information or empty dict if server is unreachable
        """
        try:
            response = requests.get(
                self.services_endpoint, 
                timeout=2.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Services request failed with status {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching services information: {e}")
            return {}

    # Wake word detection methods
    def detect_wake_word(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Send audio data to the remote server for wake word detection.
        
        Args:
            audio_data: Audio data as a numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            Dict with detection results or error information
        """
        if not self.is_service_available("wake_word"):
            return {"detected": False, "error": "Wake word service not available on the remote host"}
            
        try:
            # Convert audio to correct format (int16, mono)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # If not mono, convert to mono
            if len(audio_int16.shape) > 1 and audio_int16.shape[1] > 1:
                audio_int16 = audio_int16.mean(axis=1).astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_int16.tobytes()
            
            # Base64 encode
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Send to server
            print(f"Remote: <{colored('API', 'green')}> sending request to {colored(self.wake_word_endpoint, 'blue')}")
            response = requests.post(
                self.wake_word_endpoint,
                json={
                    "audio_data": audio_b64,
                    "sample_rate": sample_rate
                },
                timeout=5.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Wake word detection request failed with status {response.status_code}"
                logger.error(error_msg)
                print(f"Remote: <{colored('API', 'red')}> {error_msg}")
                return {"detected": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Error in remote wake word detection: {e}")
            print(f"Remote: <{colored('API', 'red')}> error in detection: {e}")
            return {"detected": False, "error": str(e)}
    
    # Whisper transcription methods
    def list_whisper_models(self) -> Dict[str, Any]:
        """
        Get information about available Whisper models.
        
        Returns:
            Dict with models information or error information
        """
        if not self.is_service_available("whisper"):
            return {"error": "Whisper service not available on the remote host"}
            
        try:
            print(f"Remote: <{colored('API', 'green')}> getting Whisper models from {colored(self.whisper_models_endpoint, 'blue')}")
            response = requests.get(
                self.whisper_models_endpoint,
                timeout=5.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Whisper models request failed with status {response.status_code}"
                logger.error(error_msg)
                print(f"Remote: <{colored('API', 'red')}> {error_msg}")
                return {"error": error_msg}
                
        except Exception as e:
            logger.error(f"Error getting Whisper models: {e}")
            print(f"Remote: <{colored('API', 'red')}> error getting Whisper models: {e}")
            return {"error": str(e)}
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, model: str = "large-v2") -> Dict[str, Any]:
        """
        Send audio data to the remote server for transcription using Whisper.
        
        Args:
            audio_data: Audio data as a numpy array
            sample_rate: Sample rate of the audio data
            model: Whisper model to use (default: "large-v2")
            
        Returns:
            Dict with transcription results or error information
        """
        if not self.is_service_available("whisper"):
            return {"success": False, "error": "Whisper service not available on the remote host"}
            
        try:
            # Convert audio to correct format (int16, mono)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # If not mono, convert to mono
            if len(audio_int16.shape) > 1 and audio_int16.shape[1] > 1:
                audio_int16 = audio_int16.mean(axis=1).astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_int16.tobytes()
            
            # Base64 encode
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Send to server
            print(f"Remote: <{colored('API', 'green')}> transcribing with {model} model via {colored(self.whisper_transcribe_endpoint, 'blue')}")
            response = requests.post(
                self.whisper_transcribe_endpoint,
                json={
                    "audio_data": audio_b64,
                    "sample_rate": sample_rate,
                    "model": model
                },
                timeout=30.0  # Longer timeout for transcription
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    print(f"Remote: <{colored('API', 'green')}> transcription successful in {result.get('processing_time', 0):.2f}s")
                return result
            else:
                error_msg = f"Transcription request failed with status {response.status_code}"
                logger.error(error_msg)
                print(f"Remote: <{colored('API', 'red')}> {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Error in remote transcription: {e}")
            print(f"Remote: <{colored('API', 'red')}> error in transcription: {e}")
            return {"success": False, "error": str(e)}

    def wait_for_wake_word(self, chunk_duration: float = 3.0) -> Optional[str]:
        """
        Listen for wake word using the remote server indefinitely.
        
        Args:
            chunk_duration: Duration of each audio chunk in seconds
            
        Returns:
            Optional[str]: Detected wake word or None if an error occurred
        """
        # Check if service is available
        if not self.is_service_available("wake_word"):
            logger.error("Wake word service not available on the remote host")
            return None
            
        try:
            # Import required libraries
            import sounddevice as sd
            from scipy import signal
            import time
            
            # Use 16kHz for wake word detection (Vosk requirement)
            target_sample_rate: int = 16000
            
            print(f"Remote: <{colored('Wake Word Service', 'green')}> is listening for wake word...")
            
            # Common sample rates to try
            sample_rates_to_try: List[int] = [16000, 44100, 48000]
            
            # Keep track of devices we've already tried and failed with
            failed_devices: Dict[int, List[int]] = {}  # {device_id: [failed_sample_rates]}
            last_working_device: Optional[int] = None
            last_working_rate: Optional[int] = None
            device_failures_count: Dict[int, int] = {}  # Count consecutive failures per device
            
            # Function to find a working device
            def find_working_device() -> Tuple[Optional[int], int]:
                nonlocal failed_devices
                
                # First try the default device
                try:
                    default_device_info = sd.query_devices(kind='input')
                    print(f"Remote: <{colored('Audio', 'green')}> found default input device: {default_device_info['name']}")
                    
                    # Test if default device works with any of our sample rates
                    for rate in sample_rates_to_try:
                        if None in failed_devices and rate in failed_devices[None]:
                            print(f"Remote: <{colored('Audio', 'yellow')}> skipping default device with {rate}Hz (previously failed)")
                            continue
                            
                        try:
                            print(f"Remote: <{colored('Audio', 'green')}> testing default device with sample rate {rate}Hz")
                            # Just query the device to see if it works with this sample rate
                            sd.check_input_settings(device=None, samplerate=rate, channels=1)
                            
                            # Try a short test recording to verify device is truly available
                            test_duration = 0.1  # Very short test recording
                            test_audio = sd.rec(
                                int(test_duration * rate),
                                samplerate=rate,
                                channels=1,
                                dtype='float32',
                                device=None,
                                blocking=True
                            )
                            
                            if test_audio.size > 0:
                                print(f"Remote: <{colored('Audio', 'green')}> default device works with {rate}Hz")
                                return None, rate
                            else:
                                print(f"Remote: <{colored('Audio', 'yellow')}> default device test recording failed with {rate}Hz")
                                if None not in failed_devices:
                                    failed_devices[None] = []
                                failed_devices[None].append(rate)
                        except Exception as e:
                            print(f"Remote: <{colored('Audio', 'yellow')}> default device doesn't work with {rate}Hz: {e}")
                            if None not in failed_devices:
                                failed_devices[None] = []
                            failed_devices[None].append(rate)
                except Exception as e:
                    print(f"Remote: <{colored('Audio', 'yellow')}> could not query default device: {e}")
                
                # Try all available devices
                devices = sd.query_devices()
                print(f"Remote: <{colored('Audio', 'green')}> found {len(devices)} audio devices")
                
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        print(f"Remote: <{colored('Audio', 'green')}> testing input device {i}: {device['name']}")
                        
                        if i in failed_devices and len(failed_devices[i]) >= len(sample_rates_to_try):
                            print(f"Remote: <{colored('Audio', 'yellow')}> skipping device {i} (all rates previously failed)")
                            continue
                            
                        for rate in sample_rates_to_try:
                            if i in failed_devices and rate in failed_devices[i]:
                                print(f"Remote: <{colored('Audio', 'yellow')}> skipping device {i} with {rate}Hz (previously failed)")
                                continue
                                
                            try:
                                print(f"Remote: <{colored('Audio', 'green')}> testing device {i} with sample rate {rate}Hz")
                                # Test device with this sample rate
                                sd.check_input_settings(device=i, samplerate=rate, channels=1)
                                
                                # Try a short test recording to verify device is truly available
                                test_duration = 0.1  # Very short test recording
                                test_audio = sd.rec(
                                    int(test_duration * rate),
                                    samplerate=rate,
                                    channels=1,
                                    dtype='float32',
                                    device=i,
                                    blocking=True
                                )
                                
                                if test_audio.size > 0:
                                    print(f"Remote: <{colored('Audio', 'green')}> device {i} works with {rate}Hz")
                                    return i, rate
                                else:
                                    print(f"Remote: <{colored('Audio', 'yellow')}> device {i} test recording failed with {rate}Hz")
                                    if i not in failed_devices:
                                        failed_devices[i] = []
                                    failed_devices[i].append(rate)
                            except Exception as e:
                                print(f"Remote: <{colored('Audio', 'yellow')}> device {i} doesn't work with {rate}Hz: {e}")
                                if i not in failed_devices:
                                    failed_devices[i] = []
                                failed_devices[i].append(rate)
                
                # If no device worked, try again with the default device as a last resort
                # Sometimes devices become available after a short wait
                try:
                    print(f"Remote: <{colored('Audio', 'yellow')}> no working device found, trying default device as last resort")
                    default_rate = 44100  # Most widely supported rate
                    test_audio = sd.rec(
                        int(0.1 * default_rate),
                        samplerate=default_rate,
                        channels=1,
                        dtype='float32',
                        blocking=True
                    )
                    if test_audio.size > 0:
                        print(f"Remote: <{colored('Audio', 'green')}> default device works as last resort with {default_rate}Hz")
                        return None, default_rate
                except Exception as e:
                    print(f"Remote: <{colored('Audio', 'red')}> last resort recording also failed: {e}")
                
                print(f"Remote: <{colored('Audio', 'red')}> no working input device found")
                return None, 0  # No working device found
            
            while True:
                # Check if we need to find a new device
                if last_working_device is None or device_failures_count.get(last_working_device, 0) > 2:
                    # Clear failure counts when looking for a new device
                    device_failures_count = {}
                    
                    # Find a working device
                    working_device, device_sample_rate = find_working_device()
                    
                    if working_device is None and device_sample_rate == 0:
                        logger.error("No working input device found")
                        # Wait and retry device detection
                        logger.info("Waiting 2 seconds before retrying device detection...")
                        print(f"Remote: <{colored('Audio', 'yellow')}> no working device found, waiting 2 seconds before retry")
                        time.sleep(2)
                        # Clear failed devices to allow retrying all devices
                        failed_devices = {}
                        continue
                else:
                    # Use the last working device and rate
                    working_device = last_working_device
                    device_sample_rate = last_working_rate
                    
                # Design a bandpass filter to focus on speech frequencies (300-3000 Hz)
                nyquist = 0.5 * device_sample_rate
                low = 300 / nyquist
                high = 3000 / nyquist
                b, a = signal.butter(4, [low, high], btype='band')
                
                # Record audio
                audio_data = None
                try:
                    device_info = "default" if working_device is None else f"{working_device}"
                    print(f"Remote: <{colored('Audio', 'green')}> recording with device {device_info} at {device_sample_rate}Hz")
                    
                    # Actual recording
                    audio_data = sd.rec(
                        int(chunk_duration * device_sample_rate),
                        samplerate=device_sample_rate,
                        channels=1,
                        dtype='float32',
                        device=working_device,
                        blocking=True
                    )
                    
                    # Update tracker of last working device/rate
                    last_working_device = working_device
                    last_working_rate = device_sample_rate
                    
                    # Reset failure count for this device on success
                    if working_device in device_failures_count:
                        device_failures_count[working_device] = 0
                    
                except Exception as rec_error:
                    print(f"Remote: <{colored('Audio', 'red')}> recording error: {rec_error}")
                    
                    # Increment failure count for this device
                    if working_device not in device_failures_count:
                        device_failures_count[working_device] = 0
                    device_failures_count[working_device] += 1
                    
                    # If this device/rate combo failed, add to failed_devices
                    if working_device not in failed_devices:
                        failed_devices[working_device] = []
                    if device_sample_rate not in failed_devices[working_device]:
                        failed_devices[working_device].append(device_sample_rate)
                    
                    # Try another device next time
                    last_working_device = None
                    
                    # Continue to next attempt
                    continue
                
                # If we don't have valid audio data, continue to next attempt
                if audio_data is None or audio_data.size == 0:
                    print(f"Remote: <{colored('Audio', 'red')}> no audio data recorded")
                    last_working_device = None  # Force finding a new device
                    continue
                
                # Process audio to improve quality
                try:
                    # Resample to target sample rate if needed
                    if device_sample_rate != target_sample_rate:
                        print(f"Remote: <{colored('Audio', 'green')}> resampling from {device_sample_rate}Hz to {target_sample_rate}Hz")
                        num_samples = int(len(audio_data) * target_sample_rate / device_sample_rate)
                        audio_data = signal.resample(audio_data, num_samples)
                except Exception as proc_error:
                    print(f"Remote: <{colored('Audio', 'red')}> error processing audio: {proc_error}")
                    # Continue with unprocessed audio if processing fails
                
                print(f"Remote: <{colored('Wake Word Service', 'green')}> sending audio data to server...")
                # Save audio data to temporary file for debugging
                try:
                    import soundfile as sf
                    sf.write('./temp.wav', audio_data, target_sample_rate)
                except Exception as e:
                    logger.error(f"Failed to save debug audio file: {e}")
                
                # Send to server for detection
                result = self.detect_wake_word(audio_data, target_sample_rate)
                
                if result.get("detected", False):
                    wake_word = result.get("wake_word")
                    print(f"Remote: <{colored('Wake Word Service', 'green')}> detected wake word: {colored(wake_word, 'yellow')}")
                    return wake_word
                    
                # Brief pause to prevent CPU overload
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Remote: <{colored('Wake Word Service', 'red')}> error in detection: {e}")
            return None 