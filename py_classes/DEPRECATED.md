# DEPRECATED: py_classes/

⚠️ **This directory is deprecated and will be removed in a future version.**

## Migration Status

All components have been migrated to the modern architecture:

### ✅ Migrated Components

| Old Location | New Location | Purpose |
|-------------|--------------|---------|
| `cls_computational_notebook.py` | `agent/notebook/` | Notebook execution |
| `cls_playbook_manager.py` | `agent/playbook/` | Strategic workflows |
| `cls_util_manager.py` | `agent/utils_manager/` | Tool orchestration |
| `cls_text_stream_painter.py` | `agent/text_painter/` | Output formatting |
| `cls_llm_selection.py` | `agent/llm_selection/` | Model selection |
| `cls_util_base.py` | `agent/utils_manager/` | Base utility class |
| `cls_rate_limit_*.py` | `infrastructure/rate_limiting/` | Rate limiting |
| `cls_vector_db.py` | `infrastructure/vector_db/` | Vector database |
| `utils/cls_utils_web_server.py` | `shared/utils/web/` | Web server utilities |
| `utils/cls_utils_rag.py` | `shared/utils/rag/` | RAG utilities |
| `utils/cls_utils_python.py` | `shared/utils/python/` | Python utilities |
| `utils/cls_utils_youtube.py` | `shared/utils/youtube/` | YouTube utilities |
| `utils/BraveSearchAPI.py` | `shared/utils/search/` | Search API |
| `remote_host/` | `infrastructure/remote_host/` | Remote services |

## How to Update Your Code

Replace old imports:
```python
# OLD - Deprecated
from py_classes.cls_computational_notebook import ComputationalNotebook
from py_classes.cls_util_manager import UtilsManager

# NEW - Modern Architecture  
from agent.notebook import ComputationalNotebook
from agent.utils_manager import UtilsManager
```

## Timeline

- **Phase 1**: ✅ Migration completed
- **Phase 2**: Legacy imports still work via compatibility layers
- **Phase 3**: ⚠️ Deprecation warnings will be added
- **Phase 4**: 🔴 Legacy directories will be removed

Please update your imports to use the modern architecture.