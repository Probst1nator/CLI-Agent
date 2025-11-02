# Linter & Formatter Setup Guide

This document explains the linting and formatting setup for this multi-language project (Python, JavaScript/TypeScript, HTML, CSS).

## 🎯 Philosophy

**Minimal tools, maximum coverage, zero overlap.**

- Only use linters/formatters when explicitly needed
- Avoid redundant tools doing the same job
- Prioritize functionality over perfect static analysis scores
- See `CLAUDE.md` for the policy on when to fix linter warnings

## 🛠️ Tools Overview

| Language | Linting | Formatting | Type Checking |
|----------|---------|------------|---------------|
| **Python** | Ruff | Ruff | Pyright |
| **JavaScript/TypeScript** | ESLint | Prettier | TypeScript (built-in) |
| **HTML/CSS/JSON/Markdown** | - | Prettier | - |

### Why These Tools?

#### Python
- **Ruff**: Extremely fast linter + formatter written in Rust. Replaces Black, Flake8, isort, pyupgrade, and more.
- **Pyright**: Fast, accurate type checker with excellent VSCodium integration (via Pylance).

#### Web Languages
- **Prettier**: Industry standard formatter for JS/TS/HTML/CSS/JSON/Markdown. Opinionated, zero-config.
- **ESLint**: Catches actual bugs and problematic patterns in JavaScript/TypeScript.

### What We Removed (Redundant)
- ❌ **Black**: Ruff's formatter does the same job, faster
- ❌ **Flake8**: Ruff includes all Flake8 rules
- ❌ **isort**: Ruff handles import sorting
- ❌ **mypy**: Pyright is faster and better integrated with VSCodium
- ❌ **pyupgrade**: Ruff includes pyupgrade rules

## 📋 Installation

### 1. Python Tools

```bash
# Install Ruff (linting + formatting)
pip install ruff

# Pyright is included in Pylance extension for VSCodium
```

### 2. JavaScript/Web Tools

```bash
# Install Prettier and ESLint
npm install --save-dev prettier eslint

# Or globally
npm install -g prettier eslint
```

### 3. VSCodium Extensions

**Required:**
```bash
# Python
code --install-extension charliermarsh.ruff        # Ruff
code --install-extension ms-python.vscode-pylance  # Pyright/Pylance

# Web
code --install-extension esbenp.prettier-vscode    # Prettier
code --install-extension dbaeumer.vscode-eslint    # ESLint
```

**Optional but recommended:**
```bash
code --install-extension ms-python.python          # Python extension pack
code --install-extension tamasfe.even-better-toml  # TOML syntax
```

## ⚙️ Configuration Files

All configuration files are in the project root:

| File | Purpose |
|------|---------|
| `pyproject.toml` | Ruff configuration for Python |
| `pyrightconfig.json` | Pyright type checking configuration |
| `.prettierrc` | Prettier formatting rules |
| `.prettierignore` | Files to exclude from Prettier |
| `.eslintrc.json` | ESLint linting rules |
| `.vscode/settings.json` | VSCodium editor settings (format on save, etc.) |

### Key Settings

#### Python (`pyproject.toml`)
- Line length: 88 characters (Black-compatible)
- Target: Python 3.11
- Enabled rules: E (pycodestyle), F (pyflakes), I (isort), UP (pyupgrade), B (bugbear), C4 (comprehensions), SIM (simplify)
- Format on save: Yes (via Ruff)

#### JavaScript/TypeScript (`.prettierrc`)
- Single quotes: Yes
- Semicolons: Yes
- Tab width: 2 spaces
- Print width: 100 characters
- Trailing commas: ES5 style

#### HTML (`.prettierrc`)
- Tab width: 4 spaces
- Print width: 120 characters

## 🚀 Usage

### Command Line

```bash
# Python - Ruff
ruff check .                 # Lint all Python files
ruff check --fix .           # Lint and auto-fix
ruff format .                # Format all Python files
ruff check --select I --fix  # Sort imports only

# JavaScript/TypeScript - ESLint
eslint .                     # Lint JS/TS files
eslint --fix .               # Lint and auto-fix

# Web - Prettier
prettier --check .           # Check formatting
prettier --write .           # Format all files
prettier --write "**/*.js"   # Format JS files only
```

### VSCodium (Automatic)

With `.vscode/settings.json` configured:
- **Save any file** → Auto-format with the appropriate tool
- **Python**: Ruff formats + organizes imports
- **JS/TS**: Prettier formats, ESLint fixes issues
- **HTML/CSS/JSON**: Prettier formats

### Pre-commit Hook (Optional)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Auto-format staged files before commit

# Python files
git diff --cached --name-only --diff-filter=ACM | grep '\.py$' | xargs -r ruff format
git diff --cached --name-only --diff-filter=ACM | grep '\.py$' | xargs -r ruff check --fix

# Web files
git diff --cached --name-only --diff-filter=ACM | grep -E '\.(js|ts|html|css|json)$' | xargs -r prettier --write

# Re-add formatted files
git diff --cached --name-only --diff-filter=ACM | xargs -r git add
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## 🔧 Common Tasks

### Format entire project
```bash
# Python
ruff format .

# Web languages
prettier --write .
```

### Check for issues without fixing
```bash
# Python
ruff check .

# JavaScript/TypeScript
eslint .

# All web files
prettier --check .
```

### Fix all auto-fixable issues
```bash
# Python
ruff check --fix .

# JavaScript/TypeScript
eslint --fix .
prettier --write .
```

### Ignore specific warnings

**Python**: Add `# noqa: <rule>` or configure in `pyproject.toml`
```python
# Example: Ignore line length for this line
very_long_variable_name = "some very long string that exceeds 88 characters"  # noqa: E501
```

**JavaScript**: Add `// eslint-disable-next-line <rule>`
```javascript
// eslint-disable-next-line no-console
console.log('Debug message');
```

**Prettier**: Add to `.prettierignore`
```
# Ignore minified files
**/*.min.js
```

## 🎨 VSCodium Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Format document | `Shift + Alt + F` |
| Format selection | `Ctrl + K, Ctrl + F` |
| Organize imports (Python) | `Shift + Alt + O` |
| Show all problems | `Ctrl + Shift + M` |
| Quick fix | `Ctrl + .` |

## 🐛 Troubleshooting

### Ruff not formatting on save
1. Check extension is installed: `code --list-extensions | grep ruff`
2. Check `.vscode/settings.json` has `"editor.defaultFormatter": "charliermarsh.ruff"` for Python
3. Reload VSCodium: `Ctrl + Shift + P` → "Developer: Reload Window"

### Prettier not working
1. Check extension: `code --list-extensions | grep prettier`
2. Verify `.prettierrc` exists in project root
3. Check file isn't in `.prettierignore`
4. Setting: `"prettier.requireConfig": true` requires `.prettierrc` file

### ESLint errors not showing
1. Check extension: `code --list-extensions | grep eslint`
2. Check `.eslintrc.json` is valid JSON
3. Restart ESLint server: `Ctrl + Shift + P` → "ESLint: Restart ESLint Server"

### Type checking not working
1. Check Pylance extension is installed
2. Verify `pyrightconfig.json` exists
3. Check Python interpreter is selected: `Ctrl + Shift + P` → "Python: Select Interpreter"

## 📚 Further Reading

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pyright Documentation](https://github.com/microsoft/pyright)
- [Prettier Documentation](https://prettier.io/docs/en/)
- [ESLint Documentation](https://eslint.org/docs/latest/)

## 🔄 Migration from Old Setup

If you were using the old setup (Black, Flake8, mypy):

1. **Uninstall old tools** (optional):
   ```bash
   pip uninstall black flake8 isort mypy pyupgrade
   ```

2. **Install Ruff**:
   ```bash
   pip install ruff
   ```

3. **Format codebase once**:
   ```bash
   ruff format .
   ruff check --fix .
   ```

4. **Update CI/CD** (if applicable):
   Replace:
   ```yaml
   - black --check .
   - flake8 .
   - mypy .
   ```
   With:
   ```yaml
   - ruff check .
   - ruff format --check .
   ```

## ⚠️ Important Notes

1. **Don't mix formatters**: Only use ONE formatter per language (Ruff for Python, Prettier for web)
2. **Commit `.vscode/settings.json`**: This ensures consistent formatting across the team
3. **Update regularly**: Run `pip install --upgrade ruff` and `npm update -g prettier eslint`
4. **CI/CD**: Add linting checks to your pipeline to catch issues before merge

## 🤝 Contributing

When contributing to this project:
1. Install the required VSCodium extensions
2. Let format-on-save handle formatting (don't format manually)
3. Fix linter errors before committing
4. Don't add `# noqa` or `// eslint-disable` without good reason

---

**Last Updated**: 2025-01-07
**Maintained By**: CLI-Agent Team
