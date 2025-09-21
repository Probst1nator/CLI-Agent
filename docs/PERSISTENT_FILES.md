# CLI-Agent Persistent Files

All files CLI-Agent saves to disk, organized by XDG Base Directory specification.

## Storage Locations

**Configuration**: `~/.config/cli-agent/` (XDG_CONFIG_HOME)
**Data**: `~/.local/share/cli-agent/` (XDG_DATA_HOME)
**State**: `~/.local/state/cli-agent/` (XDG_STATE_HOME)
**Cache**: `~/.cache/cli-agent/` (XDG_CACHE_HOME)
**Audio Models**: `~/.cache/piper-tts/` (external dependency)

> ⚡ **Migration**: Files automatically migrate from `~/.cli-agent/` to XDG paths on first run.

## Configuration Files

| File | Purpose | Location | Cleanup |
|------|---------|----------|---------|
| `llm_config.json` | LLM selections & fallback order | `~/.config/cli-agent/` | Manual |
| `user_config.json` | User preferences | `~/.config/cli-agent/` | Manual |
| `permissions.json` | Tool permissions | `~/.config/cli-agent/` | Manual |

## Data Files (Persistent)

| File/Directory | Purpose | Location | Cleanup |
|----------------|---------|----------|---------|
| `finetuning_data/` | Fine-tuning datasets | `~/.local/share/cli-agent/` | Manual |
| `sandbox/` | Agent execution space | `~/.local/share/cli-agent/` | Manual |
| `vector_db/` | All vector databases | `~/.local/share/cli-agent/` | 30d TTL |

## State Files (Ephemeral)

| File/Directory | Purpose | Location | Cleanup |
|----------------|---------|----------|---------|
| `model_limits/{date}_*` | Daily rate/token limits | `~/.local/state/cli-agent/` | Daily rotation |

## Cache & Temporary Files

| Category | TTL | Purpose | Location |
|----------|-----|---------|----------|
| `python_temp` | 24h | Python execution | `~/.cache/cli-agent/temp/` |
| `audio_temp` | 1-2h | Audio processing/debug | `~/.cache/cli-agent/audio_temp/` |
| `screenshots` | 7d | Screenshot files | `~/.cache/cli-agent/screenshots/` |
| `model_cache.json` | Auto | Model discovery cache | `~/.cache/cli-agent/` |
| `llm_cache.db` | Permanent | LLM response cache (SQLite) | `~/.cache/cli-agent/` |
| `logs/` | 1-7d | Application logs | `~/.cache/cli-agent/logs/` |

**System temp files** (1 day TTL, being migrated to cache):
- `/tmp/faster_whisper_debug_*.wav` → `~/.cache/cli-agent/audio_temp/`
- `/tmp/boosted_*.wav` → `~/.cache/cli-agent/audio_temp/`
- `/tmp/sr_test.wav` → `~/.cache/cli-agent/audio_temp/`

## Vector Databases

| Database | Purpose | Location |
|----------|---------|----------|
| SearchWeb Vector DB | Web search semantic cache | `~/.local/share/cli-agent/vector_db/searchweb_vectordb/` |
| Command History | Shell command embeddings | `~/.local/share/cli-agent/vector_db/` |

**Management:** `python scripts/manage_search_cache.py [stats|prune|list]`

## Audio Models

| Component | Location | Size | Management |
|-----------|----------|------|------------|
| Piper TTS voices | `~/.cache/piper-tts/voices/` | Varies | Manual |
| Language detection | `~/.cache/piper-tts/lid.176.bin` | ~135MB | Auto download |
| Whisper models | System-dependent | Varies | Auto download |

## Log Management

| Type | Size Limit | Backups | Max Age | Auto Cleanup |
|------|------------|---------|---------|--------------|
| Application | 10MB | 5 | 7d | ✓ |
| Debug | 5MB | 3 | 3d | ✓ |
| TUI | 2MB | 2 | 1d | ✓ |

## Quick Management

**Check storage usage:**
```bash
du -sh ~/.config/cli-agent/ ~/.local/share/cli-agent/ ~/.local/state/cli-agent/ ~/.cache/cli-agent/
```

**Manual cleanup:**
```python
from core.file_manager import get_file_manager
get_file_manager().cleanup_expired_files()
```

**SQLite cache management:**
```python
from core.cache.llm_cache_manager import LLMCacheManager
cache = LLMCacheManager("~/.cache/cli-agent/llm_cache.db")
print(cache.get_cache_statistics())
cache.cleanup_expired_entries(max_age_days=30)
cache.vacuum_database()
```

**Reset configs:**
```bash
# Reset user preferences
rm ~/.config/cli-agent/user_config.json

# Clear LLM cache
rm ~/.cache/cli-agent/llm_cache.db

# Rebuild search cache
rm -rf ~/.local/share/cli-agent/vector_db/searchweb_vectordb/

# Reset daily limits (next day)
rm -rf ~/.local/state/cli-agent/model_limits/
```

## Directory Structure Overview

```
~/.config/cli-agent/           # XDG_CONFIG_HOME
├── user_config.json
├── llm_config.json
└── permissions.json

~/.local/share/cli-agent/      # XDG_DATA_HOME
├── finetuning_data/
├── sandbox/
└── vector_db/
    └── searchweb_vectordb/

~/.local/state/cli-agent/      # XDG_STATE_HOME
└── model_limits/
    ├── 2024-01-15_model_token_limits.json
    └── 2024-01-15_model_rate_limits.json

~/.cache/cli-agent/            # XDG_CACHE_HOME
├── temp/
├── audio_temp/
├── screenshots/
├── logs/
├── llm_cache.db               # SQLite database
└── model_cache.json
```

## Backup Essentials

**Critical (backup required):**
- `~/.config/cli-agent/user_config.json`
- `~/.config/cli-agent/llm_config.json`
- `~/.config/cli-agent/permissions.json`

**Important (backup recommended):**
- `~/.local/share/cli-agent/finetuning_data/`
- `~/.local/share/cli-agent/vector_db/` (for search history)

**Ephemeral (auto-managed, no backup needed):**
- `~/.local/state/cli-agent/` (daily rotation)
- `~/.cache/cli-agent/` (all cache files, logs, temp files)