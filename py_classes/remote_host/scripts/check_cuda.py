#!/usr/bin/env python3
import os
import subprocess

print("======= CUDA Environment Check =======\n")

# Check environment variables
print("Environment Variables:")
for var in ["CUDA_HOME", "CUDA_VISIBLE_DEVICES", "CT2_USE_CUDA", "LD_LIBRARY_PATH", "PATH"]:
    print(f"  {var}={os.environ.get(var, 'Not set')}")

# Check CUDA installation
print("\nCUDA Installation:")
try:
    out = subprocess.check_output(["ls", "-la", "/usr/local/cuda/bin"]).decode("utf-8")
    print("  CUDA binaries: Found")
except Exception as e:
    print(f"  CUDA binaries: Not found - {e}")

try:
    out = subprocess.check_output(["which", "nvcc"]).decode("utf-8").strip()
    print(f"  nvcc: {out}")
except Exception as e:
    print(f"  nvcc: Not found - {e}")

# Check PyTorch CUDA
print("\nPyTorch CUDA:")
try:
    import torch
    print(f"  torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  Error loading PyTorch: {e}")

# Check CTranslate2 CUDA
print("\nCTranslate2 CUDA:")
try:
    import ctranslate2
    print(f"  ctranslate2 version: {ctranslate2.__version__}")
    cuda_count = ctranslate2.get_cuda_device_count()
    print(f"  CUDA device count: {cuda_count}")
    if cuda_count > 0:
        print("  CUDA is properly detected!")
    else:
        print("  WARNING: CTranslate2 does not detect any CUDA devices")
except Exception as e:
    print(f"  Error loading CTranslate2: {e}")

print("\n======= End of CUDA Check =======\n") 