# Building Docker Images with CUDA Support

This document provides instructions for building Docker images with proper CUDA support for the Whisper transcription service.

## Prerequisites

- CUDA-capable GPU (e.g., NVIDIA Jetson or other NVIDIA GPU)
- Docker installed
- NVIDIA Container Toolkit installed
- Git

## Building for Jetson Devices

To build the Docker image for Jetson devices:

```bash
./shell_scripts/remote_host_jetson.sh
```

This script will:
1. Build the Docker image using `Dockerfile.jetson`
2. Compile CTranslate2 from source with CUDA support
3. Install all required dependencies
4. Set up the CUDA verification and startup scripts

## Building for Other ARM Devices with NVIDIA GPUs

To build the Docker image for other ARM devices with NVIDIA GPUs:

```bash
./shell_scripts/remote_host_arm.sh
```

## Verification

When the container starts, it will run the CUDA verification script to check:
- Environment variables (CUDA_HOME, CT2_USE_CUDA, etc.)
- CUDA installation (binaries, nvcc compiler)
- PyTorch CUDA configuration
- CTranslate2 CUDA support

You should see detailed output in the container logs, including successful detection of CUDA devices for CTranslate2.

## Troubleshooting

If CTranslate2 doesn't detect CUDA devices:

1. Verify that NVIDIA drivers are properly installed on the host:
   ```bash
   nvidia-smi
   ```

2. Check that the NVIDIA Container Toolkit is working:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. Ensure the container has access to the GPU:
   ```bash
   docker run --rm --gpus all your-image python3 -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
   ```

4. Check that the container has all required CUDA libraries:
   ```bash
   docker run --rm --gpus all your-image ls -la /usr/local/cuda/lib64/
   ```

## Re-Builds

When rebuilding after code changes:

1. Make sure to rebuild the image from scratch to ensure proper compilation:
   ```bash
   docker build --no-cache -t your-image -f Dockerfile.jetson .
   ```

2. For quick testing, you can run the container with:
   ```bash
   docker run --rm --gpus all -p 5000:5000 your-image
   ``` 