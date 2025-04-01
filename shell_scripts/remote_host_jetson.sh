#!/bin/bash
# Script to build and run CLI-Agent Remote Host with GPU support for Jetson devices

set -e  # Exit on any error

IMAGE_NAME="cli-agent-remote-host-jetson"

# Try different Dockerfiles until one succeeds
echo "Building Docker image for Jetson with GPU support..."
echo "Note: This will take some time as it needs to install dependencies."
echo "The Rust compiler and CTranslate2 components may need to compile from source which could take 20+ minutes."

if docker build --no-cache --progress=plain -t $IMAGE_NAME -f py_classes/remote_host/Dockerfile.arm .; then
    echo "Successfully built using Dockerfile.arm"
elif docker build --no-cache --progress=plain -t $IMAGE_NAME -f py_classes/remote_host/Dockerfile.jetson .; then
    echo "Successfully built using Dockerfile.jetson"
else
    echo "ERROR: Failed to build with any provided Dockerfile"
    exit 1
fi

# Stop any running container
docker stop cli-agent-remote-host 2>/dev/null || true
docker rm cli-agent-remote-host 2>/dev/null || true

echo "Running container with optimized Jetson GPU settings..."
# Enhanced settings specifically for Jetson devices
docker run -d \
  --name cli-agent-remote-host \
  -p 5000:5000 \
  --runtime nvidia \
  --privileged \
  --memory=0 \
  --memory-swap=-1 \
  --shm-size=2g \
  --ipc=host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e CT2_USE_CUDA=1 \
  -e PYTHONMALLOC=malloc \
  -e MALLOC_TRIM_THRESHOLD_=100000 \
  -e MALLOC_MMAP_THRESHOLD_=100000 \
  -e PYTHONMALLOCSTATS=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6 \
  -e OPENBLAS_CORETYPE=ARMV8 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e NVIDIA_PERSISTENCED_MODE=1 \
  -v cli-agent-whisper-cache:/root/.cache/whisper \
  -v cli-agent-vosk-cache:/root/.cache/vosk \
  -v cli-agent-huggingface-cache:/root/.cache/huggingface \
  $IMAGE_NAME

echo "Container started with optimized Jetson GPU settings. Access the service at http://localhost:5000"
echo "To check logs: docker logs -f cli-agent-remote-host" 