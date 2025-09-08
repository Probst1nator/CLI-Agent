# DEPRECATED: py_methods/

⚠️ **This directory is deprecated and will be removed in a future version.**

## Migration Status

All utilities have been migrated to the modern shared structure:

### ✅ Migrated Components

| Old Location | New Location | Purpose |
|-------------|--------------|---------|
| `utils.py` | `shared/audio/audio_utils.py` | Audio processing utilities |

## How to Update Your Code

Replace old imports:
```python
# OLD - Deprecated
from py_methods.utils import listen_microphone

# NEW - Modern Architecture  
from shared.audio.audio_utils import listen_microphone
```

## Timeline

- **Phase 1**: ✅ Migration completed
- **Phase 2**: Legacy imports still work via compatibility layers
- **Phase 3**: ⚠️ Deprecation warnings will be added  
- **Phase 4**: 🔴 Legacy directories will be removed

Please update your imports to use the modern shared structure.