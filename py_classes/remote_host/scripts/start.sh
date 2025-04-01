#!/bin/bash
echo "Checking CUDA installation..."
python3 /app/check_cuda.py
echo "Preloading Whisper models..."
python3 /app/py_classes/remote_host/preload_models.py
echo "Starting Gunicorn server..."
# Set Python memory management variables
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000
export PYTHONMALLOC=malloc
export PYTHONMALLOCSTATS=0
# Use only 1 worker to avoid memory competition and set a longer timeout
exec gunicorn -b 0.0.0.0:5000 --timeout 600 --workers 1 --threads 1 --worker-class sync py_classes.remote_host.cls_remote_host_server:app 