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
     echo "Note: Some optional dependencies might be missing" && \
     # Install faster-whisper with specified versions
     pip install --no-cache-dir "huggingface-hub>=0.13.0" "numpy>=1.17.0" && \
     pip install --no-cache-dir av && \
     RUSTFLAGS="-C target-feature=-crt-static" pip install --no-cache-dir tokenizers==0.13.3 && \
     CT2_USE_CUDA=1 pip install --no-cache-dir ctranslate2==3.20.0 && \
     pip install --no-cache-dir onnxruntime-gpu || pip install --no-cache-dir onnxruntime && \
     pip install --no-cache-dir --no-deps faster-whisper==1.0.0)

# Copy the project files
COPY . .

# Make the preload script executable
RUN chmod +x /app/py_classes/remote_host/preload_models.py

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "Preloading Whisper models..."\n\
python /app/py_classes/remote_host/preload_models.py\n\
echo "Starting Gunicorn server..."\n\
# Set Python memory management variables\n\
export MALLOC_TRIM_THRESHOLD_=100000\n\
export MALLOC_MMAP_THRESHOLD_=100000\n\
export PYTHONMALLOC=malloc\n\
export PYTHONMALLOCSTATS=0\n\
# Set reasonable timeouts for transcription\n\
exec gunicorn -b 0.0.0.0:5000 --timeout 600 py_classes.remote_host.cls_remote_host_server:app\n\
' > /app/start.sh && chmod +x /app/start.sh

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

# Run the startup script
CMD ["/app/start.sh"]

# For development, use this instead:
# CMD ["python", "py_classes/remote_host/run_server.py"] 