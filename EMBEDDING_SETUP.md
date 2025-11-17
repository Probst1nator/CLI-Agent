# EmbeddingGemma Notebook Setup - Complete

## What Was Done

### 1. Isolated Environment Created ✓

Location: `~/.cli-agent/embeddinggemma_venv`

**Installed Packages:**
```
sentence-transformers: 5.1.2
transformers: 4.57.1  
torch: 2.6.0+cu118 (CUDA 11.8 for GTX 1080 Ti)
datasets
python-dotenv
huggingface-hub
ipykernel
```

### 2. Jupyter Kernel Registered ✓

**Kernel Name:** `embeddinggemma`
**Display Name:** `EmbeddingGemma (Python 3.12)`

List available kernels:
```bash
jupyter kernelspec list
```

### 3. Notebook Configured ✓

The notebook `emeddinggemma_finetuning.ipynb` is pre-configured to use the isolated kernel automatically.

### 4. HuggingFace Authentication ✓

Automatically loads `HUGGINGFACE_TOKEN` from `.env` file in Cell 7.

## How To Use (NOW WITH AUTOMATIC SETUP!)

### Quick Start - VSCodium/VS Code (RECOMMENDED)

```bash
cd ~/repos/CLI-Agent
code emeddinggemma_finetuning.ipynb  # or: codium emeddinggemma_finetuning.ipynb
```

**In VSCodium:**
1. **First time:** Close and reopen VSCodium to load workspace settings
2. Click "Select Kernel" (top-right)
3. Select "EmbeddingGemma (Python 3.12)"
   - If not visible: Choose "Enter interpreter path..." → paste: `~/.cli-agent/embeddinggemma_venv/bin/python`
4. Run Cell 6 to verify (should print "✓ Already running in isolated environment")
5. Run All Cells

### Alternative - Classic Jupyter

```bash
cd ~/repos/CLI-Agent
jupyter notebook emeddinggemma_finetuning.ipynb
```

**In Jupyter:**
1. Run **Cell 6** (Environment Setup) - it handles everything automatically!
2. **First time only:** Follow prompts to switch kernel (one-time action)
3. Run All Cells

### What Cell 6 Does Automatically

**First run (no environment):**
- Creates `~/.cli-agent/embeddinggemma_venv`
- Installs PyTorch 2.6.0, sentence-transformers, transformers, etc.
- Registers Jupyter kernel "EmbeddingGemma (Python 3.12)"
- Shows instructions to switch kernel
- Takes ~2-3 minutes

**After first run:**
- You switch kernel: `Kernel` → `Change Kernel` → "EmbeddingGemma (Python 3.12)"
- Restart kernel: `Kernel` → `Restart Kernel`
- Re-run Cell 6 to verify

**Every subsequent run:**
- Cell 6 detects correct environment
- Prints "✓ Already running in isolated environment"
- Continues automatically

### Expected Behavior

All cells should execute without import errors. The model will:
1. Download (first run only, ~300MB)
2. Load successfully
3. Train on permission classification examples
4. Show before/after similarity scores

## Verification

Test the environment manually:

```bash
~/.cli-agent/embeddinggemma_venv/bin/python -c "
from sentence_transformers import SentenceTransformer
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
model = SentenceTransformer('google/embeddinggemma-300M', device='cuda', trust_remote_code=True)
print('✓ Model loads successfully')
"
```

## Troubleshooting

### "KeyError: 'gemma3_text'"

**Cause:** Jupyter kernel is using old shared environment instead of isolated venv  
**Fix:** 
1. Restart Jupyter server completely
2. Reopen notebook
3. Verify kernel is "EmbeddingGemma (Python 3.12)"

### "Module not found" errors

**Cause:** Wrong Python environment
**Fix:**
```bash
~/.cli-agent/embeddinggemma_venv/bin/pip install [missing-package]
```

### Remove & Recreate

```bash
# Remove everything
rm -rf ~/.cli-agent/embeddinggemma_venv
jupyter kernelspec remove embeddinggemma

# Then just open notebook and run Cell 6 - it recreates everything automatically!
```

## Existing ML Permission System

The repository already contains a complete ML-based autonomous permission system:

### Core Files

- `core/permissions/embedding_classifier.py` - Uses Ollama for embeddings
- `core/permissions/autonomous_permission.py` - Orchestration layer
- `core/permissions/settings_manager.py` - Configuration management  
- `tests/core/test_autonomous_permissions.py` - Comprehensive tests

### Architecture

```
Tool Call Request
    ↓
Layer 1: Static Deny Rules (block immediately)
    ↓
Layer 2: Static Allow Rules (approve immediately)
    ↓
Layer 3: ML Classifier (if enabled)
    ├─ Auto-allow (high confidence safe)
    ├─ Ask with warning (high confidence unsafe)
    └─ Neutral (uncertain)
    ↓
Layer 4: User Prompt (fallback)
```

### Note on Implementation

The existing system uses **Ollama** with `embeddinggemma:latest`, not sentence-transformers.  
The notebook demonstrates the fine-tuning concept but uses a different embedding backend.

## Next Steps

The notebook is ready for experimentation and fine-tuning exploration.  
The production system already has a working implementation using Ollama.

