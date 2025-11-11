# HuggingFace Local Provider

A provider for running HuggingFace transformer models locally with automatic downloading and memory management.

## Overview

The HuggingFace Local Provider enables CLI-Agent to run large language models directly on your machine using the HuggingFace transformers library. It provides:

- **Automatic Model Downloads**: Models are downloaded from HuggingFace Hub on first use
- **Memory Efficient**: Only one model loaded at a time, automatic unloading
- **Streaming Generation**: Real-time token-by-token generation
- **GPU/CPU Support**: Automatic device detection with fallback
- **Cancellation Support**: Stop generation mid-stream
- **Custom Models**: Easy to add any HuggingFace model

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `transformers>=4.48.0` - Core model loading and inference
- `torch>=2.0.0` - PyTorch backend
- `huggingface_hub>=0.20.0` - Model downloading
- `accelerate>=0.20.0` - Efficient loading and multi-GPU support
- `safetensors>=0.4.0` - Safe tensor format
- `sentencepiece>=0.1.99` - Tokenizer support
- `protobuf>=3.20.0` - Tokenizer protocols

### 2. Configure (Optional)

Create/edit `.env` file:

```bash
# HuggingFace token (required for gated models like Llama)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=your_token_here

# Device selection (auto, cuda, cpu, mps)
HF_LOCAL_DEVICE=auto

# Storage location for models
HF_LOCAL_MODELS_DIR=~/.cli-agent/models/

# Max VRAM to use (in GB)
HF_LOCAL_MAX_MEMORY_GB=16
```

## Supported Models

### Pre-configured Models

| Model | Size | Context | Features |
|-------|------|---------|----------|
| `moonlight-16b-instruct` | 16B MoE (3B active) | 8K | High quality, efficient |
| `qwen2.5-7b-instruct` | 7B | 128K | Long context |
| `qwen2.5-3b-instruct` | 3B | 128K | Smaller, fast |
| `qwen2.5-1.5b-instruct` | 1.5B | 128K | Very lightweight |
| `gemma2-9b-instruct` | 9B | 8K | Google's model |
| `gemma2-2b-instruct` | 2B | 8K | Compact version |
| `llama3.2-3b-instruct` | 3B | 128K | Requires HF_TOKEN |
| `llama3.2-1b-instruct` | 1B | 128K | Requires HF_TOKEN |
| `qwen2-vl-7b-instruct` | 7B | 32K | Vision capable |
| `qwen2-vl-2b-instruct` | 2B | 32K | Vision capable |

### Model Sizes (Approximate Disk Space)

- **Moonlight-16B**: ~32GB
- **7B models**: ~14GB
- **3B models**: ~6GB
- **2B models**: ~4GB
- **1B models**: ~2GB

## Usage

### Through CLI-Agent

Models are automatically available in CLI-Agent:

```bash
# Auto-select (router picks appropriate model)
python main.py

# Explicit model selection
python main.py -m moonlight-16b-instruct

# With GUI
python main.py --gui
```

### Direct Python Usage

```python
from core.chat import Chat, Role
from core.providers.cls_huggingface_local_interface import HuggingFaceLocalClient

# Create a chat
chat = Chat()
chat.add_message(Role.SYSTEM, "You are a helpful assistant.")
chat.add_message(Role.USER, "What is 2+2?")

# Generate response
stream = HuggingFaceLocalClient.generate_response(
    chat,
    model_key="moonlight-16b-instruct",
    temperature=0.7
)

# Stream tokens
for chunk in stream:
    if hasattr(chunk, 'message'):
        print(chunk.message.content, end="", flush=True)
```

### Through LLM Router

```python
from core.llm_router import LlmRouter

router = LlmRouter()

# Simple generation
response = router.generate_completion(
    "What is 2+2?",
    model_key="moonlight-16b-instruct"
)
print(response)
```

## Testing

### Unit Tests

```bash
# Run all tests
python tests/providers/test_huggingface_local.py

# Run with verbose output
python tests/providers/test_huggingface_local.py -v
```

### Integration Test

⚠️ **WARNING**: Downloads ~32GB for Moonlight model

```bash
python tests/providers/test_huggingface_local.py --integration
```

### Interactive Demo

```bash
python test_moonlight_demo.py
```

This provides:
1. Dependency check
2. Disk space verification
3. Simple test mode
4. Interactive chat mode

## Moonlight-16B-A3B-Instruct

### Overview

Moonlight is a 16B parameter MoE (Mixture of Experts) model with only 3B active parameters per forward pass, making it efficient while maintaining high quality.

**Key Features**:
- Architecture: DeepSeek V3 (MoE)
- Total params: 16B (3B active)
- Context: 8K tokens
- Training: 5.7T tokens with Muon optimizer
- Performance: Outperforms Llama-3.2-3B and Qwen2.5-3B
- License: MIT

### Performance Benchmarks

| Benchmark | Moonlight | Llama3.2-3B | Qwen2.5-3B |
|-----------|-----------|-------------|------------|
| MMLU | **70.0** | 54.75 | 65.6 |
| HumanEval | **48.1** | 28.0 | 42.1 |
| GSM8K | 77.4 | 34.0 | **79.1** |
| MATH | **45.3** | 8.5 | 42.6 |

### Why Moonlight?

1. **Efficient**: Only 3B active params (fast inference)
2. **High Quality**: Beats larger models on many benchmarks
3. **Open**: MIT license, no restrictions
4. **Modern**: Uses latest DeepSeek V3 architecture
5. **Well-supported**: Active development, good documentation

### System Requirements

**Minimum** (CPU):
- RAM: 32GB+
- Disk: 40GB free
- CPU: Modern multi-core

**Recommended** (GPU):
- VRAM: 16GB+ (e.g., RTX 4080, A100)
- RAM: 16GB+
- Disk: 40GB free SSD

**Optimal** (GPU):
- VRAM: 24GB+ (e.g., RTX 4090, A6000)
- RAM: 32GB+
- Disk: 100GB+ NVMe SSD

## Adding Custom Models

### At Runtime

```python
from core.providers.huggingface_local_models import add_custom_model
from core.lm_capabilities import LMCapabilities

add_custom_model(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    model_key="llama3.1-8b-instruct",
    context_window=128000,
    capabilities=[LMCapabilities.LOCAL],
    requires_auth=True,
    trust_remote_code=False,
    description="Llama 3.1 8B instruction-tuned"
)
```

### In Registry

Edit `core/providers/huggingface_local_models.py`:

```python
SUPPORTED_MODELS = [
    # ... existing models ...
    {
        "model_id": "your-org/your-model",
        "model_key": "your-model",
        "context_window": 8192,
        "capabilities": [LMCapabilities.LOCAL],
        "requires_auth": False,
        "trust_remote_code": True,
        "description": "Your custom model",
    },
]
```

Then update `core/llm.py` (automatic via registry iteration).

## Architecture

### Components

```
core/providers/
├── cls_huggingface_local_interface.py  # Main provider
├── huggingface_local_models.py         # Model registry
└── ...

core/
├── llm.py                               # LLM integration
└── llm_router.py                        # Router (uses provider)

~/.cli-agent/
└── models/
    ├── moonshotai--Moonlight-16B-A3B-Instruct/
    ├── Qwen--Qwen2.5-7B-Instruct/
    └── ...
```

### How It Works

1. **Model Request**: User requests a model (e.g., "moonlight-16b-instruct")
2. **Registry Lookup**: Provider finds model_id from model_key
3. **Check Download**: Check if model exists in `~/.cli-agent/models/`
4. **Download**: If not present, download from HuggingFace Hub
5. **Load Model**: Load into memory (unload previous if needed)
6. **Generate**: Create streaming generator
7. **Stream**: Yield tokens as they're generated

### Memory Management

- **LRU**: Last model stays in memory
- **Auto-unload**: Previous model unloaded when switching
- **GPU Cache**: CUDA cache cleared after unload
- **Single Instance**: Only one model in memory at a time

### Streaming

Uses `TextIteratorStreamer` from transformers:
1. Generation runs in background thread
2. Tokens yielded to queue as generated
3. Main thread reads from queue
4. Cancellation via threading.Event

## Troubleshooting

### Out of Memory

**Problem**: Model too large for GPU

**Solution**:
```bash
# Use CPU (slower but works)
export HF_LOCAL_DEVICE=cpu

# Or use smaller model
python main.py -m qwen2.5-3b-instruct
```

### Slow Download

**Problem**: Model download is slow/stuck

**Solution**:
- Check internet connection
- Download manually: `huggingface-cli download moonshotai/Moonlight-16B-A3B-Instruct`
- Use mirror: `export HF_ENDPOINT=https://hf-mirror.com`

### Gated Model Access

**Problem**: "Repository not found" for Llama models

**Solution**:
1. Go to model page (e.g., https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
2. Accept license terms
3. Create token: https://huggingface.co/settings/tokens
4. Add to `.env`: `HF_TOKEN=your_token_here`

### Custom Code Trust

**Problem**: "trust_remote_code" error

**Solution**:
Already handled automatically. If you see this, the model config may have changed. Set manually:
```python
model_info["trust_remote_code"] = True
```

### Import Errors

**Problem**: "No module named 'transformers'"

**Solution**:
```bash
pip install transformers torch huggingface_hub accelerate
# or
pip install -r requirements.txt
```

## Performance Tips

### GPU Optimization

1. **Use GPU**: Set `HF_LOCAL_DEVICE=cuda`
2. **Batch Size**: For multiple requests, use batch generation
3. **Quantization**: Use 8-bit/4-bit loading (future feature)
4. **Flash Attention**: Install `flash-attn` for faster attention

### CPU Optimization

1. **Use smaller models**: 1.5B-3B range
2. **Reduce context**: Shorter conversations = faster
3. **Lower temperature**: Greedy decoding (temp=0) is faster
4. **Thread count**: Set `OMP_NUM_THREADS` to CPU count

### Storage Optimization

1. **Clean cache**: Remove unused models from `~/.cli-agent/models/`
2. **Shared cache**: Use `HF_LOCAL_MODELS_DIR` pointing to shared location
3. **Symlinks**: Link to existing HF cache: `~/.cache/huggingface/`

## Comparison with Ollama

| Feature | HuggingFace Local | Ollama |
|---------|-------------------|--------|
| Models | Any HF model | Curated list |
| Setup | Python deps | Binary install |
| Custom Code | ✅ Supported | ❌ Limited |
| Vision | ✅ Via models | ✅ Via models |
| Quantization | Future | ✅ Built-in |
| Speed | Good | Better (GGUF) |
| Memory | Higher | Lower |
| Flexibility | Very high | Medium |

**Use HuggingFace Local when**:
- Need latest/specific models
- Model uses custom code
- Want full Python control
- Need research features

**Use Ollama when**:
- Want simple setup
- Need quantized models
- Want lower memory usage
- Production deployment

## References

- [Moonlight Model Card](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct)
- [Moonlight Technical Report](https://arxiv.org/abs/2502.16982)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [DeepSeek V3 Architecture](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## Contributing

To add support for new models:

1. Test model works with transformers
2. Add to `SUPPORTED_MODELS` in `huggingface_local_models.py`
3. Test with `test_huggingface_local.py`
4. Update this documentation
5. Submit PR

## License

Provider code: Same as CLI-Agent (check LICENSE)

Models: Each model has its own license:
- Moonlight: MIT
- Qwen: Apache 2.0
- Gemma: Gemma License
- Llama: Llama License

Always check model cards for current license terms.
