# User Scripts

User-facing utility scripts for CLI-Agent setup, configuration, and management.

## 📋 Scripts Overview

### `setup_cli_agent.sh` ⭐ (Recommended)
**Purpose**: Complete CLI-Agent command line integration setup

**What it does**:
- Creates `cli-agent` command accessible from anywhere
- Adds CLI-Agent to system PATH
- Sets up shell aliases (`ca` as shortcut for `cli-agent`)
- Configures environment variables
- Supports both user-local and system-wide installation

**Usage**:
```bash
# User installation (recommended)
./user_scripts/setup_cli_agent.sh --user

# System-wide installation (requires sudo)
./user_scripts/setup_cli_agent.sh --system

# Check installation status
./user_scripts/setup_cli_agent.sh --check

# Uninstall
./user_scripts/setup_cli_agent.sh --uninstall
```

**After installation**:
```bash
# Use from anywhere
cli-agent -m "Hello world"
cli-agent --gui
cli-agent --isolated

# Or use the alias
ca -docker 3
ca --help
```

**Options**:
- `--user` - Install for current user only (`~/.local/bin`)
- `--system` - Install system-wide (`/usr/local/bin`, requires sudo)
- `--path` - Add CLI-Agent directory to PATH
- `--check` - Verify installation status
- `--uninstall` - Remove all CLI-Agent setup
- `--help` - Show help message

**What gets configured**:
- Symlink: `~/.local/bin/cli-agent` → `~/repos/CLI-Agent/cli-agent`
- PATH: Adds `~/.local/bin` to your shell PATH
- Alias: `alias ca='cli-agent'` in shell profile
- Environment: `CLI_AGENT_DIR` variable

---

### `setup_bash_alias.sh`
**Purpose**: Simple symlink creation for system-wide access

**What it does**:
- Creates symlink in `/usr/local/bin/cli-agent` pointing to `main.py`
- Requires sudo permissions
- Minimal setup without shell profile modifications

**Usage**:
```bash
sudo ./user_scripts/setup_bash_alias.sh
```

**Note**: Use `setup_cli_agent.sh --system` instead for a more complete setup.

---

### `setup_desktop_entry.sh`
**Purpose**: Create desktop application entry for GUI launch

**What it does**:
- Creates `.desktop` file in `~/.local/share/applications/`
- Adds CLI-Agent to system application menu
- Pre-configures with voice mode and local LLM
- Auto-greets user as "Nova" on launch

**Usage**:
```bash
./user_scripts/setup_desktop_entry.sh
```

**After installation**:
- Find "CLI-Agent" in your application menu
- Click to launch with voice mode enabled
- Opens in new terminal window

**Configuration**:
- Default command: `python3 main.py --voice --local -m "Hi, can you introduce yourself as Nova..."`
- Edit the script to customize startup behavior

---

### `run_screen_agent.sh`
**Purpose**: Automated screen capture + AI analysis workflow

**What it does**:
- Captures screenshot using `-img` flag
- Analyzes image content with AI
- Generates detailed blog post about the image
- Creates styled HTML output with embedded image
- Uses online LLM with auto-execution (`--yolo`)

**Usage**:
```bash
./user_scripts/run_screen_agent.sh
```

**Workflow**:
1. Takes screenshot (using Spectacle or fallback)
2. AI observes and explains image contents
3. AI researches key aspects using web search
4. AI creates styled HTML blog post with embedded image
5. Saves to `blogposts/` folder
6. Opens in Firefox

**Requirements**:
- `PYTHON_ENV_PATH` set in `.env` file
- Online API keys configured
- Screenshot tool available (Spectacle or fallback)

**Customization**:
Edit the `-m` message arguments to change the analysis workflow.

---

### `manage_search_cache.py`
**Purpose**: Manage SearchWeb vector database cache

**What it does**:
- Views cache statistics (total entries, size, etc.)
- Prunes old search results by date
- Prunes by domain (remove specific websites)
- Inspects cached entries with metadata
- Clears entire cache if needed

**Usage**:
```bash
# View cache statistics
./user_scripts/manage_search_cache.py --stats

# Remove entries older than 30 days
./user_scripts/manage_search_cache.py --prune-days 30

# Remove all cached results from a domain
./user_scripts/manage_search_cache.py --prune-domain example.com

# Inspect specific entries (limit to 10)
./user_scripts/manage_search_cache.py --inspect --limit 10

# Clear entire cache
./user_scripts/manage_search_cache.py --clear-all
```

**Cache Location**: `~/.cli-agent/search_cache/`

**Benefits**:
- Reduces redundant web searches
- Faster search result retrieval
- Semantic similarity matching
- Configurable retention policies

---

## 🚀 Quick Start

### Recommended Setup (First-Time Users)

```bash
cd ~/repos/CLI-Agent

# 1. Complete CLI integration (recommended)
./user_scripts/setup_cli_agent.sh --user

# 2. Restart your shell
source ~/.bashrc  # or ~/.zshrc

# 3. Test the installation
cli-agent --help
ca --help  # Test alias

# 4. Optional: Desktop application entry
./user_scripts/setup_desktop_entry.sh

# 5. Start using CLI-Agent
cli-agent -m "Hello, world!"
ca --gui  # Launch web interface
```

### System-Wide Installation (All Users)

```bash
# Install for all users (requires sudo)
sudo ./user_scripts/setup_cli_agent.sh --system

# Now all users can run
cli-agent -m "Hello"
```

---

## 📂 Script Comparison

| Feature | `setup_cli_agent.sh` | `setup_bash_alias.sh` | `setup_desktop_entry.sh` |
|---------|---------------------|----------------------|-------------------------|
| **Scope** | Full integration | Minimal symlink | GUI launcher |
| **PATH setup** | ✅ Yes | ❌ No | ❌ No |
| **Shell alias** | ✅ Yes (`ca`) | ❌ No | ❌ No |
| **Environment vars** | ✅ Yes | ❌ No | ❌ No |
| **User/System** | Both options | System only | User only |
| **Sudo required** | Only for `--system` | ✅ Yes | ❌ No |
| **Uninstall** | ✅ Yes | ❌ Manual | ❌ Manual |
| **Check status** | ✅ Yes | ❌ No | ❌ No |

**Recommendation**: Use `setup_cli_agent.sh` for comprehensive setup.

---

## 🔧 Troubleshooting

### "cli-agent command not found"

**Solution 1**: Check installation
```bash
./user_scripts/setup_cli_agent.sh --check
```

**Solution 2**: Reload shell configuration
```bash
source ~/.bashrc  # or ~/.zshrc for zsh
```

**Solution 3**: Verify PATH
```bash
echo $PATH | grep .local/bin
which cli-agent
```

**Solution 4**: Reinstall
```bash
./user_scripts/setup_cli_agent.sh --user
```

---

### Desktop entry not appearing

**Solution 1**: Update desktop database
```bash
update-desktop-database ~/.local/share/applications/
```

**Solution 2**: Check file exists
```bash
ls -la ~/.local/share/applications/cli-agent.desktop
```

**Solution 3**: Verify file permissions
```bash
chmod 755 ~/.local/share/applications/cli-agent.desktop
```

**Solution 4**: Log out and back in
Desktop environments sometimes cache application lists.

---

### run_screen_agent.sh fails

**Check 1**: Verify environment variable
```bash
grep PYTHON_ENV_PATH .env
```

**Check 2**: Test Python environment
```bash
source /path/to/your/venv/bin/activate
python3 --version
```

**Check 3**: Verify API keys
```bash
grep -E "OPENAI|ANTHROPIC|GOOGLE" .env
```

**Check 4**: Test screenshot tool
```bash
spectacle --help  # or whatever tool you're using
```

---

### manage_search_cache.py errors

**Error: "python-dateutil is required"**
```bash
pip install python-dateutil
```

**Error: "Vector database is not ready"**
```bash
# Check ChromaDB installation
pip install chromadb

# Check Ollama (if using local embeddings)
ollama list
```

**Error: "Permission denied"**
```bash
# Fix cache directory permissions
chmod -R u+rw ~/.cli-agent/search_cache/
```

---

## 📝 Customization Examples

### Custom Desktop Entry Behavior

Edit `setup_desktop_entry.sh` to customize startup:

```bash
# Change the BASE_CMD line (line 16)
BASE_CMD="echo \"Starting CLI-Agent...\" && python3 ${MAIN_SCRIPT} --gui"

# Or for different greeting
BASE_CMD="python3 ${MAIN_SCRIPT} -v -m \"Hello, I'm ready to help!\""
```

### Custom Screen Analysis Workflow

Edit `run_screen_agent.sh` message arguments:

```bash
# Change analysis focus
python3 /path/to/main.py --online -img --yolo -m \
  "Analyze the UI/UX design elements in the image", \
  "Suggest improvements for accessibility and usability", \
  "Create a detailed report in HTML format"
```

### Custom Shell Alias

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Quick shortcuts
alias cag='cli-agent --gui'        # GUI mode
alias cai='cli-agent --isolated'   # Isolated mode
alias cav='cli-agent -v'           # Voice mode
alias cad='cli-agent -docker 3'    # 3 containers

# Advanced workflows
alias ca-screenshot='cli-agent -img -m "Analyze this screenshot"'
alias ca-code='cli-agent -m "Help me with coding: "'
```

---

## 🧹 Uninstallation

### Complete Removal

```bash
# Uninstall CLI-Agent integration
./user_scripts/setup_cli_agent.sh --uninstall

# Remove desktop entry
rm ~/.local/share/applications/cli-agent.desktop
update-desktop-database ~/.local/share/applications/

# Remove cache (optional)
rm -rf ~/.cli-agent/

# Remove environment from shell profiles (manual)
nano ~/.bashrc  # Remove CLI-Agent related lines
```

### Keep Config, Remove Commands

```bash
# Just remove the cli-agent command
rm ~/.local/bin/cli-agent  # or sudo rm /usr/local/bin/cli-agent

# Keep config and cache
# (located in ~/.cli-agent/)
```

---

## 🔗 Related Documentation

- **Main Setup**: `../README.md` - General installation and usage
- **Isolated Mode**: `../docs/ISOLATED_MODE.md` - Docker isolation
- **Multi-User**: `../docs/MULTI_USER_ARCHITECTURE.md` - SaaS deployment
- **Architecture**: `../docs/` - Technical documentation
- **SaaS Scripts**: `../saas_scripts/README.md` - Infrastructure deployment

---

## 💡 Tips & Best Practices

### For Daily Use

1. **Use the alias**: `ca` is much faster to type than `cli-agent`
2. **Check status regularly**: Run `ca --check` to verify setup
3. **Update PATH**: After installation, always reload your shell
4. **Isolated mode**: Use `ca --isolated` for safe testing

### For Automation

1. **Screen workflows**: Customize `run_screen_agent.sh` for your needs
2. **Cache management**: Schedule weekly cache pruning with cron
3. **Desktop entries**: Create multiple entries for different workflows
4. **Environment management**: Keep API keys in `.env`, never commit

### For Multiple Users

1. **System-wide install**: Use `--system` flag for shared machines
2. **Per-user config**: Each user gets their own `~/.cli-agent/` directory
3. **Shared cache**: Optional shared cache setup (advanced)
4. **Resource limits**: Monitor usage with `docker stats` for containerized mode

---

## 📊 Script Maintenance

### When to Use Each Script

| Scenario | Use This Script |
|----------|----------------|
| First-time setup | `setup_cli_agent.sh --user` |
| Add to app menu | `setup_desktop_entry.sh` |
| Automated screenshots | `run_screen_agent.sh` |
| Clean old searches | `manage_search_cache.py --prune-days 30` |
| System-wide access | `setup_cli_agent.sh --system` |
| Quick minimal setup | `setup_bash_alias.sh` |

### Keeping Scripts Updated

Scripts are maintained in this repository. To update:

```bash
cd ~/repos/CLI-Agent
git pull origin main

# Reinstall if needed
./user_scripts/setup_cli_agent.sh --user
```

---

## 🐛 Known Issues

### Issue: PATH not updating immediately
**Workaround**: Restart shell or run `source ~/.bashrc`

### Issue: Desktop entry shows wrong icon
**Workaround**: Edit `~/.local/share/applications/cli-agent.desktop` and change `Icon=` line

### Issue: run_screen_agent.sh hangs on screenshot
**Workaround**: Ensure screenshot tool (Spectacle) is installed and configured

### Issue: Cache grows too large
**Solution**: Run `./user_scripts/manage_search_cache.py --prune-days 7` regularly

---

## 📞 Support

- **Issues**: https://github.com/anthropics/CLI-Agent/issues
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `docs/` folder for detailed guides
- **Examples**: See examples in main README

---

**These scripts make CLI-Agent easily accessible and integrate it seamlessly into your daily workflow.**
