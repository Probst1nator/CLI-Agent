#!/bin/bash
# Script to build and run the CLI-Agent Remote Host Docker container

# Print colorful messages
print_green() {
    echo -e "\e[32m$1\e[0m"
}

print_yellow() {
    echo -e "\e[33m$1\e[0m"
}

print_red() {
    echo -e "\e[31m$1\e[0m"
}

# Detect if we're on ARM architecture (like Jetson Orin)
ARCH=$(uname -m)
IS_ARM=0
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm"* ]]; then
    IS_ARM=1
    print_yellow "Detected ARM architecture ($ARCH). Using specialized Dockerfile for ARM."
fi

# Check if NVIDIA GPU is available
HAS_NVIDIA_GPU=0
if command -v nvidia-smi &> /dev/null; then
    HAS_NVIDIA_GPU=1
    print_yellow "NVIDIA GPU detected. Configuring for GPU acceleration."
fi

# Set variables
IMAGE_NAME="cli-agent-remote-host"
CONTAINER_NAME="cli-agent-remote-host"
HOST_PORT=5000
CONTAINER_PORT=5000

# Function to build Docker image
build_image() {
    if [ $IS_ARM -eq 1 ]; then
        # For ARM architecture
        print_green "Building Docker image for ARM using Dockerfile.arm..."
        print_yellow "Note: This will take some time as it needs to install dependencies."
        print_yellow "The Rust compiler and CTranslate2 components may need to compile from source which could take 20+ minutes."
        if ! docker build --no-cache --progress=plain -f py_classes/remote_host/Dockerfile.arm -t $IMAGE_NAME .; then
            print_yellow "First build attempt failed. Trying alternative Dockerfile.jetson..."
            if ! docker build --no-cache --progress=plain -f py_classes/remote_host/Dockerfile.jetson -t $IMAGE_NAME .; then
                print_red "Failed to build Docker image!"
                exit 1
            fi
        fi
    else
        # For other architectures
        print_green "Building Docker image using standard Dockerfile..."
        print_yellow "Note: This will take some time as it needs to install dependencies."
        print_yellow "The Rust compiler and CTranslate2 components may need to compile from source which could take several minutes."
        docker build --no-cache --progress=plain -f py_classes/remote_host/Dockerfile -t $IMAGE_NAME .
        
        if [ $? -ne 0 ]; then
            print_red "Failed to build Docker image!"
            exit 1
        fi
    fi
}

# Function to run Docker container
run_container() {
    # Check if container already exists
    if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
        print_yellow "Container '$CONTAINER_NAME' already exists. Stopping and removing it..."
        docker stop $CONTAINER_NAME > /dev/null 2>&1
        docker rm $CONTAINER_NAME > /dev/null 2>&1
    fi
    
    print_green "Starting container..."
    
    # Run with GPU support if available
    if [ $HAS_NVIDIA_GPU -eq 1 ] && [ $IS_ARM -eq 1 ]; then
        # For Jetson devices with optimized settings
        print_green "Starting container with optimized Jetson GPU settings..."
        docker run -d \
            --name $CONTAINER_NAME \
            -p $HOST_PORT:$CONTAINER_PORT \
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
    elif [ $HAS_NVIDIA_GPU -eq 1 ]; then
        # For other NVIDIA GPUs
        print_green "Starting container with NVIDIA GPU support..."
        docker run -d \
            --name $CONTAINER_NAME \
            -p $HOST_PORT:$CONTAINER_PORT \
            --gpus all \
            --shm-size=2g \
            --ipc=host \
            -e NVIDIA_VISIBLE_DEVICES=0 \
            -e CUDA_VISIBLE_DEVICES=0 \
            -e CT2_USE_CUDA=1 \
            -e PYTHONMALLOC=malloc \
            -e MALLOC_TRIM_THRESHOLD_=100000 \
            -e MALLOC_MMAP_THRESHOLD_=100000 \
            -e PYTHONMALLOCSTATS=0 \
            -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
            -v cli-agent-whisper-cache:/root/.cache/whisper \
            -v cli-agent-vosk-cache:/root/.cache/vosk \
            -v cli-agent-huggingface-cache:/root/.cache/huggingface \
            $IMAGE_NAME
    else
        # No GPU
        print_yellow "Starting container without GPU support..."
        docker run -d \
            --name $CONTAINER_NAME \
            -p $HOST_PORT:$CONTAINER_PORT \
            -e PYTHONMALLOC=malloc \
            -e MALLOC_TRIM_THRESHOLD_=100000 \
            -e MALLOC_MMAP_THRESHOLD_=100000 \
            -v cli-agent-whisper-cache:/root/.cache/whisper \
            -v cli-agent-vosk-cache:/root/.cache/vosk \
            -v cli-agent-huggingface-cache:/root/.cache/huggingface \
            $IMAGE_NAME
    fi
    
    if [ $? -ne 0 ]; then
        print_red "Failed to start container!"
        exit 1
    fi
    
    print_green "Container started successfully!"
    print_green "Remote host service is now available at: http://localhost:$HOST_PORT"
    print_yellow "To check container logs: docker logs -f $CONTAINER_NAME"
    print_yellow "To stop the container: docker stop $CONTAINER_NAME"
}

# Main execution
print_green "=== CLI-Agent Remote Host Docker Builder ==="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_red "Docker not found! Please install Docker first."
    exit 1
fi

# Parse command line arguments
case "$1" in
    --build-only)
        build_image
        ;;
    --run-only)
        run_container
        ;;
    *)
        build_image
        run_container
        ;;
esac

print_green "Done!" 