# CLI-Agent Isolated Mode

Run CLI-Agent in a fully isolated Docker container with a single flag: `--isolated`

## Quick Start

```bash
# First run (builds Docker image, ~2-3 min)
python3 main.py --isolated

# Subsequent runs (instant start, ~2-5 sec)
python3 main.py --isolated -m "Hello, world!"
```

## Features

- **🐳 Transparent Containerization**: Feels identical to local execution
- **⚡ Cached Image**: First build takes ~2-3 min, all subsequent runs start instantly
- **🔄 Auto-Sync**: Latest source code automatically copied into container
- **💾 Persistent Data**: Configuration and chat history persist across runs
- **🔒 Isolated Filesystem**: Container cannot access host files (except working directory)
- **🌐 Full Feature Support**: All CLI-Agent features work in isolated mode

## How It Works

### Architecture

```
User runs: python3 main.py --isolated [args]
     ↓
Early detection in main.py (line ~89)
     ↓
Docker runner orchestrates:
  1. Check if cached image exists
  2. Build image if needed (first run only)
  3. Sync latest source code into container
  4. Execute: docker run [mounts] python3 main.py [args]
  5. Stream output back to terminal
     ↓
User sees: Normal CLI-Agent output
```

### Cached Image Strategy

**First Run**:
- Builds `cli-agent:isolated-cache` image with all dependencies
- Base image: `python:3.10-slim`
- Installs all packages from `requirements.txt`
- Time: ~2-3 minutes (one-time cost)

**Subsequent Runs**:
- Reuses cached image
- Copies latest source code (fast sync)
- Starts container immediately
- Time: ~2-5 seconds

**Rebuild Triggers**:
- Manually delete image: `docker rmi cli-agent:isolated-cache`
- Requirements change: Image rebuild recommended (automatic detection coming soon)

## Usage Examples

### Basic Usage

```bash
# Interactive mode
python3 main.py --isolated

# With message
python3 main.py --isolated -m "Explain Docker"

# Continue conversation
python3 main.py --isolated -c

# Voice mode
python3 main.py --isolated -v
```

### GUI/API Mode

```bash
# Web interface (auto-opens browser)
python3 main.py --isolated --gui

# API server only
python3 main.py --isolated --api

# Access: http://localhost:7860
```

### Advanced Usage

```bash
# Debug mode (verbose logging)
python3 main.py --isolated --debug

# Multiple messages
python3 main.py --isolated -m "First" "Second" "Third"

# Specific LLM model
python3 main.py --isolated --llm gemini-2.5-flash

# Auto-execute safe commands
python3 main.py --isolated --yolo

# Combined flags
python3 main.py --isolated --gui -v --debug
```

## What Gets Isolated

### ✅ Isolated (Cannot Access)

- Host filesystem (except mounted directories)
- Host's local network services
- Other running processes
- Host's environment variables (except forwarded ones)

### 🔓 Accessible

- **Working Directory**: Current directory mounted at `/workspace`
- **Config Directory**: `~/.cli-agent` mounted for persistence
- **Internet**: Full internet access for AI API calls
- **API Keys**: Automatically forwarded from host environment

## File Persistence

### Persistent Data

The following are automatically persisted across container runs:

```
~/.cli-agent/               # Mounted into container
├── chat_history/          # All conversations
├── logs/                  # Session logs
├── cache/                 # LLM cache
├── config/                # Configuration files
└── profiles/              # Saved profiles
```

### Working Directory Access

```bash
# Files in current directory are accessible
cd ~/projects/my-app
python3 main.py --isolated -m "Analyze this codebase"

# Container sees files at /workspace
# Can read/write files in current directory
```

### Source Code Updates

Latest CLI-Agent source code is automatically synced into the container on every run:

```bash
# Edit CLI-Agent source code
vim ~/repos/CLI-Agent/tools/bash.py

# Changes immediately available in next isolated run
python3 main.py --isolated  # Uses updated code
```

## Environment Variables

### Automatically Forwarded

The following environment variables are automatically passed to the container:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `GROQ_API_KEY`
- `BRAVE_API_KEY`
- `NVIDIA_API_KEY`

### Using .env File

If `.env` exists in the project root, it's automatically loaded:

```bash
# Create .env file
cat > .env <<EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
EOF

# Automatically loaded in isolated mode
python3 main.py --isolated
```

## Special Features

### GUI/API Mode

Automatically forwards port 7860 for web interface:

```bash
python3 main.py --isolated --gui
# Web UI: http://localhost:7860
# Auto-opens browser
```

### Voice Mode

Attempts to mount audio devices for voice input/output:

```bash
python3 main.py --isolated -v
# Mounts /dev/snd if available
# Shows warning if audio devices unavailable
```

### Debug Mode

Full debug logging with container introspection:

```bash
python3 main.py --isolated --debug
# All debug output streamed to terminal
# Container name: cli-agent-isolated-<pid>
```

## Troubleshooting

### Docker Not Found

```
❌ Docker is not installed or not in PATH
```

**Solution**: Install Docker
```bash
# Ubuntu/Debian
sudo apt-get install docker.io
sudo usermod -aG docker $USER  # Add user to docker group
newgrp docker                  # Reload group membership

# macOS
brew install --cask docker

# Verify
docker --version
```

### Permission Denied

```
docker: Got permission denied while trying to connect...
```

**Solution**: Add user to docker group
```bash
sudo usermod -aG docker $USER
newgrp docker  # Or logout/login
```

### Image Build Fails

```
❌ Failed to build Docker image
```

**Solution**: Check build logs and retry
```bash
# Manual build with verbose output
cd ~/repos/CLI-Agent
docker build -f docker_files/Dockerfile.isolated-exec -t cli-agent:isolated-cache .

# Common fixes
docker system prune -a  # Clean up Docker cache
sudo service docker restart  # Restart Docker daemon
```

### Container Won't Start

```
❌ Failed to start container
```

**Solution**: Check container logs
```bash
# List all containers
docker ps -a

# Check logs
docker logs cli-agent-isolated-<pid>

# Remove stuck containers
docker rm -f $(docker ps -aq --filter name=cli-agent-isolated)
```

### Audio Not Working

```
⚠️ Warning: Audio devices not available in container
```

**Solution**: This is normal if `/dev/snd` doesn't exist
```bash
# Check if audio devices exist
ls -la /dev/snd

# If missing, use text-only mode or run locally
python3 main.py -v  # Local execution with voice
```

### Source Code Not Updating

Source code should auto-sync on every run. If not:

```bash
# Force rebuild image
docker rmi cli-agent:isolated-cache
python3 main.py --isolated  # Rebuilds image
```

### Port Already in Use (GUI Mode)

```
Error: Port 7860 already in use
```

**Solution**: Stop existing container
```bash
# Find container using port
docker ps | grep 7860

# Stop it
docker stop <container-id>

# Or force remove all isolated containers
docker rm -f $(docker ps -aq --filter name=cli-agent-isolated)
```

## Performance

### Startup Time

| Run Type | Time |
|----------|------|
| First run (build) | ~2-3 minutes |
| Subsequent runs | ~2-5 seconds |
| Source sync | ~1-2 seconds |
| Container cleanup | Instant (auto) |

### Resource Usage

| Resource | Usage |
|----------|-------|
| Image size | ~2-3 GB |
| Memory | ~500MB - 2GB |
| CPU | Minimal overhead |
| Disk I/O | Negligible |

### Optimization Tips

1. **Keep image cached**: Don't delete `cli-agent:isolated-cache` unless necessary
2. **Use bind mounts**: Working directory is bind-mounted (instant access)
3. **Persistent config**: Keeps chat history and cache across runs
4. **Clean up regularly**:
   ```bash
   docker system prune -a  # Remove unused images/containers
   ```

## Comparison with Other Modes

### Isolated vs Local

| Feature | `--isolated` | Local |
|---------|-------------|-------|
| **Dependencies** | In container | Host system |
| **Isolation** | Full | None |
| **Startup** | ~2-5 sec | Instant |
| **Updates** | Auto-sync | Direct |
| **Safety** | High | Medium |

### Isolated vs Docker Multi-Container

| Feature | `--isolated` | `-docker N` |
|---------|-------------|-------------|
| **Use Case** | Single isolated run | Multi-user SaaS |
| **Complexity** | Simple | Complex |
| **Setup** | Instant | Requires orchestration |
| **Web UI** | Optional | Always |

### Isolated vs Production Container

| Feature | `--isolated` | Production Deploy |
|---------|-------------|-------------------|
| **Source Code** | Synced at runtime | Baked into image |
| **Use Case** | Development/Testing | Production deployment |
| **Rebuild** | Auto-sync | Manual rebuild |
| **Persistence** | Host mounts | Docker volumes |

## Advanced Configuration

### Custom Image Name

Edit `shared/utils/docker_runner.py`:

```python
IMAGE_NAME = "my-custom-cli-agent:latest"
```

### Additional Volume Mounts

Modify `docker_runner.py` → `prepare_docker_run_command()`:

```python
# Add custom mount
cmd.extend(["-v", "/path/on/host:/path/in/container"])
```

### Resource Limits

Add resource constraints in `docker_runner.py`:

```python
cmd.extend([
    "--memory", "2g",
    "--cpus", "2.0"
])
```

### Network Configuration

Add network settings:

```python
# Isolate from network
cmd.extend(["--network", "none"])

# Custom DNS
cmd.extend(["--dns", "8.8.8.8"])
```

## Development Workflow

### Recommended Workflow

```bash
# 1. Build image once
python3 main.py --isolated --help

# 2. Develop CLI-Agent locally
vim tools/my_new_tool.py

# 3. Test in isolated mode (uses latest code)
python3 main.py --isolated -m "Test my new tool"

# 4. Iterate quickly (no rebuild needed)
# Edit code → run --isolated → repeat
```

### Debugging Container

```bash
# Run with debug mode
python3 main.py --isolated --debug

# Inspect running container
docker ps | grep cli-agent-isolated
docker exec -it cli-agent-isolated-<pid> /bin/bash

# Check container logs
docker logs cli-agent-isolated-<pid>
```

### Rebuilding Image

```bash
# When requirements.txt changes
docker rmi cli-agent:isolated-cache
python3 main.py --isolated  # Auto-rebuilds

# Or manually
cd ~/repos/CLI-Agent
docker build -f docker_files/Dockerfile.isolated-exec -t cli-agent:isolated-cache .
```

## Security Considerations

### What's Protected

✅ **Host Filesystem**: Container can't access files outside working directory
✅ **Process Isolation**: Container processes separated from host
✅ **Network Isolation**: Container uses separate network namespace
✅ **User Isolation**: Runs as non-root user inside container

### What's Exposed

⚠️ **Working Directory**: Current directory is mounted (read/write)
⚠️ **Config Directory**: `~/.cli-agent` is mounted (read/write)
⚠️ **Environment Variables**: API keys are forwarded
⚠️ **Internet Access**: Full internet access (for AI APIs)

### Best Practices

1. **Don't run in sensitive directories**: Container can modify working directory
2. **Keep API keys secure**: Use `.env` file with proper permissions
3. **Review mounted directories**: Check what's accessible
4. **Regular updates**: Rebuild image periodically for security patches

## Uninstallation

To completely remove isolated mode components:

```bash
# Remove Docker image
docker rmi cli-agent:isolated-cache

# Remove any running containers
docker rm -f $(docker ps -aq --filter name=cli-agent-isolated)

# Clean up Docker system
docker system prune -a

# Remove source files (optional)
rm docker_files/Dockerfile.isolated-exec
rm shared/utils/docker_runner.py
# Edit main.py to remove --isolated flag and early detection logic
```

## Quick Reference

### Common Commands

```bash
# Basic usage
python3 main.py --isolated
python3 main.py --isolated -m "Your message"
python3 main.py --isolated -c         # Continue conversation

# GUI mode
python3 main.py --isolated --gui      # Web UI at localhost:7860

# Voice mode
python3 main.py --isolated -v         # Voice input/output
python3 main.py --isolated -s         # TTS only

# Debug mode
python3 main.py --isolated --debug    # Verbose logging

# Specific LLM
python3 main.py --isolated --llm gemini-2.5-flash
```

### Timing

- **First run:** ~2-3 min (builds image)
- **Next runs:** ~2-5 sec (instant)

### What's Mounted

- `~/.cli-agent` → `/home/cliagent/.cli-agent` (config, chat history, cache)
- Current working directory → `/workspace` (access your files)
- `.env` file automatically loaded (API keys)

### Quick Troubleshooting

```bash
# Docker not found
sudo apt install docker.io
sudo usermod -aG docker $USER
newgrp docker

# Rebuild image
docker rmi cli-agent:isolated-cache
python3 main.py --isolated

# Clean up containers
docker rm -f $(docker ps -aq --filter name=cli-agent-isolated)

# Check running containers
docker ps | grep cli-agent-isolated
```

### Files Created

- `docker_files/Dockerfile.isolated-exec` - Docker image definition
- `shared/utils/docker_runner.py` - Container orchestration
- `main.py` - Added `--isolated` flag and early detection

## FAQ

**Q: Do I need Docker installed?**
A: Yes, Docker is required for `--isolated` mode. Install from [docker.com](https://docs.docker.com/get-docker/)

**Q: How much disk space does it use?**
A: ~2-3 GB for the cached image (one-time)

**Q: Can I use this in production?**
A: Yes, but consider using dedicated production Dockerfile for better optimization

**Q: Does it work offline?**
A: Container works offline after initial build, but AI features require internet

**Q: Can I run multiple instances?**
A: Yes, each run gets a unique container (isolated from each other)

**Q: How do I update CLI-Agent source code?**
A: Just edit files normally - changes are auto-synced on next run

**Q: Can I use custom Python packages?**
A: Add them to `requirements.txt` and rebuild image

**Q: Does it support GPU?**
A: Not by default, but can be added with `--gpus all` flag in docker_runner.py

**Q: How do I pass custom environment variables?**
A: Edit `docker_runner.py` → `prepare_docker_run_command()` and add to `api_key_vars` list

**Q: Is this slower than local execution?**
A: Container startup adds ~2-5 seconds, negligible for most workflows

## Support

- **Documentation**: This file (docs/ISOLATED_MODE.md)
- **Issues**: https://github.com/anthropics/CLI-Agent/issues
- **Docker Docs**: https://docs.docker.com/

---

**Isolated mode provides a secure, convenient way to run CLI-Agent in a fully containerized environment with minimal setup and maximum flexibility.**
