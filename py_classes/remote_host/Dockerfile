FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing and build tools
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libsndfile1 \
    libgomp1 \
    build-essential \
    python3-dev \
    gcc \
    git \
    ffmpeg \
    libavformat-dev \
    libavfilter-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavcodec-dev \
    pkg-config \
    libssl-dev \
    rustc \
    cargo \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with improved error handling
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Failed to install dependencies, trying alternative approach" && \
     pip install --no-cache-dir flask==2.3.3 requests==2.31.0 gunicorn==21.2.0 && \
     pip install --no-cache-dir vosk==0.3.45 && \
     pip install --no-cache-dir numpy scipy sounddevice soundfile python-dotenv && \
     pip install --no-cache-dir termcolor psutil && \
     # Install faster-whisper with specified versions
     pip install --no-cache-dir "huggingface-hub>=0.13.0" "numpy>=1.17.0" && \
     pip install --no-cache-dir av && \
     RUSTFLAGS="-C target-feature=-crt-static" pip install --no-cache-dir tokenizers==0.13.3)

# Install CTranslate2 dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CT2_USE_CUDA=1
ENV CUDA_DYNAMIC_LOADING=ON

# Clone and build CTranslate2 from source with CUDA support
WORKDIR /tmp
RUN git clone https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && \
    git checkout v3.20.0 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DWITH_CUDA_RUNTIME=STATIC -DCUDA_DYNAMIC_LOADING=ON -DOPENMP_RUNTIME=COMP && \
    make -j$(nproc) && \
    cd ../python && \
    pip install -e .

# Return to app directory
WORKDIR /app

# Verify CTranslate2 installation with CUDA
RUN python3 -c "import ctranslate2; print(f'CTranslate2 version: {ctranslate2.__version__}'); print(f'CUDA device count: {ctranslate2.get_cuda_device_count()}')" || \
    echo "WARNING: CTranslate2 CUDA detection failed - will use CPU fallback"

# Install onnxruntime
RUN pip install --no-cache-dir onnxruntime-gpu || pip install --no-cache-dir onnxruntime

# Install faster-whisper
RUN pip install --no-cache-dir --no-deps faster-whisper==1.0.0

# Copy the project files
COPY . .

# Make the preload script executable
RUN chmod +x /app/py_classes/remote_host/preload_models.py

# Copy the CUDA check and startup scripts
COPY py_classes/remote_host/scripts/check_cuda.py /app/check_cuda.py
COPY py_classes/remote_host/scripts/start.sh /app/start.sh
RUN chmod +x /app/check_cuda.py /app/start.sh

# Set environment variables
ENV PORT=5000
ENV HOST=0.0.0.0
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Create volume for model cache
VOLUME /root/.cache/vosk
VOLUME /root/.cache/whisper
VOLUME /root/.cache/huggingface

# Create empty .env file
RUN touch /app/.env

# Use our startup script as entrypoint
ENTRYPOINT ["/app/start.sh"]

# For development, use this instead:
# CMD ["python", "py_classes/remote_host/run_server.py"] 