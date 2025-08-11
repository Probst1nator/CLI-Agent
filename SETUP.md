# CLI-Agent Setup Guide

## 🚀 **Complete Installation (Recommended)**

For full CLI-Agent with all tools:

```bash
# 1. Clone repository
git clone https://github.com/Probst1nator/CLI-Agent.git
cd CLI-Agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# 3. Install shared infrastructure
pip install -e ./shared
pip install -e ./core

# 4. Install root dependencies
pip install -r requirements.txt

# 5. Run tests to verify
python run_tests.py --level 2
```

## 🔧 **Tool-Specific Installation**

For individual tools only:

### **File Copier Tool**
```bash
cd tools/file_copier
pip install -r requirements.txt  # Installs shared+core+tool deps
python main.py
```

### **Podcast Generator Tool**
```bash
cd tools/podcast_generator  
pip install -r requirements.txt  # Installs shared+core+tool deps
python main.py
```

## 🔄 **Development Setup**

For contributors and development:

```bash
# 1. Standard installation (above)

# 2. Install development dependencies
pip install pytest black flake8 mypy

# 3. Set up git hooks (optional)
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running tests..."
python run_tests.py --level 2
EOF
chmod +x .git/hooks/pre-commit

# 4. Verify development setup
python run_tests.py --level 4 --verbose
```

## 📦 **Package Structure**

This project uses a **formal monorepo** structure with editable packages:

- **`shared/`**: Common utilities (installable package)
- **`core/`**: AI infrastructure (installable package)  
- **`tools/`**: Standalone applications (depend on shared+core)
- **`utils/`**: Plugin system for main CLI agent

Each tool's `requirements.txt` installs the shared infrastructure using:
```
-e ../../shared
-e ../../core
```

## 🔍 **Troubleshooting**

### **Import Errors**
```bash
# If you see import errors, ensure packages are installed:
pip install -e ./shared -e ./core

# Or for a specific tool:
cd tools/file_copier
pip install -r requirements.txt
```

### **Legacy Import Warnings**
The codebase maintains backward compatibility. If you see warnings about legacy imports, they're safe to ignore during the transition period.

### **Path Issues**
If tools can't find shared components:
```bash
# Verify installation
pip list | grep cli-agent

# Should show:
# cli-agent-shared    1.0.0    /path/to/CLI-Agent/shared
# cli-agent-core      1.0.0    /path/to/CLI-Agent/core
```

## ⚡ **Quick Start Commands**

```bash
# Run main CLI agent
python main.py

# Run file copier tool
python tools/file_copier/main.py

# Run podcast generator
python tools/podcast_generator/main.py

# Run comprehensive tests
python run_tests.py --level 4

# Run tool-specific tests
python run_tests.py --tools file_copier --verbose
```

## 🏗️ **Architecture Benefits**

- **✅ No more sys.path hacks**: Clean, standard imports
- **✅ Explicit dependencies**: Clear tool requirements  
- **✅ Backward compatibility**: Legacy imports still work
- **✅ Easy tool extraction**: Tools can be distributed independently
- **✅ Development efficiency**: Shared infrastructure automatically available