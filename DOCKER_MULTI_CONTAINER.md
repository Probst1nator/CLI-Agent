# CLI-Agent Multi-Container System

A complete solution for running multiple containerized CLI-Agent instances with full directory synchronization.

## Overview

The multi-container system allows you to spawn N CLI-Agent instances in Docker containers while maintaining full access to your local codebase including ignored files (`.git`, `__pycache__`, etc.).

## Quick Start

### 1. Setup CLI-Agent Command

```bash
# Install the cli-agent command for your user
./scripts/setup_cli_agent.sh --user

# Restart your shell or source the profile
source ~/.bashrc
```

### 2. Start Multiple Containers

```bash
# Start 3 containerized CLI-Agent instances (browsers auto-open)
cli-agent -docker 3

# Start 5 containers with voice mode enabled
cli-agent -docker 5 -v

# Start 2 containers without opening browsers
cli-agent -docker 2 --no-browser

# Start containers with specific arguments
cli-agent -docker 2 -m "Hello from container"
```

### 3. Access the Instances

Each container gets 3 consecutive ports starting from 7860:
- Container 1: http://localhost:7860, 7861, 7862
- Container 2: http://localhost:7863, 7864, 7865
- Container 3: http://localhost:7866, 7867, 7868
- Container N: http://localhost:7860+(N-1)*3, ...

**Browser Auto-Opening**: By default, browser windows automatically open for each container when services are ready. Use `--no-browser` to disable this behavior.

## Commands

### Docker Operations
```bash
cli-agent -docker N               # Start N containers (auto-open browsers)
cli-agent -docker N --no-browser  # Start N containers without opening browsers
cli-agent --docker-status         # Show container status
cli-agent --docker-stop           # Stop all containers
cli-agent --docker-logs [N]       # Show logs (all or specific container)
cli-agent --docker-clean          # Clean up all resources
```

### Local Operations (unchanged)
```bash
cli-agent -m "Hello world"        # Local execution
cli-agent -v                      # Local with voice
cli-agent -c                      # Continue conversation
```

### Convenience Alias
```bash
ca -docker 5                      # Short alias for cli-agent
```

## Architecture

### Components

1. **CLI Wrapper (`cli-agent`)**
   - Main entry point that detects Docker vs local execution
   - Delegates Docker operations to orchestrator
   - Passes through local arguments to `main.py`

2. **Docker Orchestrator (`scripts/docker_orchestrator.py`)**
   - Generates dynamic docker-compose files
   - Manages container lifecycle
   - Handles port allocation and networking

3. **Sync Dockerfile (`docker_files/Dockerfile.sync`)**
   - Optimized for full directory synchronization
   - Includes all necessary dependencies
   - Proper user permissions and security

4. **Setup Script (`scripts/setup_cli_agent.sh`)**
   - Installs CLI command system-wide or per-user
   - Configures shell integration
   - Manages PATH and aliases

### Directory Synchronization

The system uses Docker bind mounts to provide real-time synchronization:

```yaml
volumes:
  - "~/repos/CLI-Agent:/workspace"  # Full directory sync
  - "cli-agent-data-N:/workspace/data"       # Persistent data
  - "cli-agent-logs-N:/workspace/logs"       # Persistent logs
```

**What gets synchronized:**
- ✅ All source code files
- ✅ Configuration files (`.env`, etc.)
- ✅ Git repository (`.git/`)
- ✅ Python cache (`__pycache__/`)
- ✅ Temporary files
- ✅ All ignored files

### Port Allocation Strategy

Each container gets 3 consecutive ports:
- **Base formula**: `7860 + (container_number - 1) * 3`
- **Container 1**: 7860, 7861, 7862
- **Container 2**: 7863, 7864, 7865
- **Container N**: 7860+(N-1)*3, 7861+(N-1)*3, 7862+(N-1)*3

## Browser Auto-Opening

### How It Works
When containers start, the system:
1. **Waits for services** - Monitors each container until web services are ready
2. **Opens browsers automatically** - Launches browser windows for each container
3. **Cross-platform support** - Works on Linux, macOS, and Windows
4. **Fallback handling** - Gracefully handles cases where browsers can't open

### Browser Detection (Linux)
The system tries multiple browser commands in order:
- `xdg-open` (default system browser)
- `firefox`
- `google-chrome`
- `chromium`

### Controlling Browser Behavior
```bash
# Auto-open browsers (default)
cli-agent -docker 3

# Disable browser auto-opening
cli-agent -docker 3 --no-browser

# Manual browser opening after startup
firefox http://localhost:7860 http://localhost:7863 http://localhost:7866
```

### Timing
- **Service wait time**: Up to 2 minutes per container
- **Browser delay**: 1 second between browser opens
- **Background operation**: Browser opening runs in background thread

## Container Management

### Viewing Status
```bash
cli-agent --docker-status
```
Output shows:
- Container names and status
- Port mappings
- Running time
- Health status

### Viewing Logs
```bash
# All containers
cli-agent --docker-logs

# Specific container
cli-agent --docker-logs 2
```

### Stopping Containers
```bash
# Graceful stop
cli-agent --docker-stop

# Force cleanup (removes volumes)
cli-agent --docker-clean
```

## Environment Variables

The system automatically passes through these environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `GROQ_API_KEY`

Additional variables can be set in `.env` file or shell environment.

## File Structure

```
CLI-Agent/
├── cli-agent                          # Main CLI wrapper
├── scripts/
│   ├── docker_orchestrator.py         # Container orchestration
│   └── setup_cli_agent.sh            # Installation script
├── docker_files/
│   ├── Dockerfile.sync                # Multi-container Dockerfile
│   └── docker-compose.sync.yml       # Generated compose file
└── main.py                           # Original CLI-Agent entry point
```

## Installation Options

### User Installation (Recommended)
```bash
./scripts/setup_cli_agent.sh --user
```
- Installs to `~/.local/bin/cli-agent`
- No sudo required
- User-specific installation

### System Installation
```bash
./scripts/setup_cli_agent.sh --system
```
- Installs to `/usr/local/bin/cli-agent`
- Requires sudo
- Available for all users

### Check Installation
```bash
./scripts/setup_cli_agent.sh --check
```

### Uninstall
```bash
./scripts/setup_cli_agent.sh --uninstall
```

## Troubleshooting

### Docker Image Build Issues
```bash
# Manual build
python3 scripts/docker_orchestrator.py --build

# Check Docker status
docker images | grep cli-agent
```

### Container Startup Issues
```bash
# Check logs
cli-agent --docker-logs

# Check container status
docker ps -a | grep cli-agent

# Manual cleanup
cli-agent --docker-clean
```

### Permission Issues
```bash
# Ensure proper permissions
chmod +x cli-agent
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

### Port Conflicts
- Default ports: 7860-7890+
- Modify base port in orchestrator if needed
- Check for existing services: `netstat -tlnp | grep 786`

## Examples

### Basic Usage
```bash
# Start 3 containers
cli-agent -docker 3

# Check status
cli-agent --docker-status

# Access via browser
firefox http://localhost:7860  # Container 1
firefox http://localhost:7863  # Container 2
firefox http://localhost:7866  # Container 3

# Stop all
cli-agent --docker-stop
```

### Development Workflow
```bash
# Start containers for parallel testing
cli-agent -docker 5

# Make code changes in your editor
# Changes are immediately available in all containers

# Test different scenarios in parallel
# Container 1: Test feature A
# Container 2: Test feature B
# etc.

# View logs
cli-agent --docker-logs 1

# Clean up when done
cli-agent --docker-clean
```

### Resource Management
Each container defaults to:
- **Memory**: 2GB limit, 512MB reservation
- **CPU**: 1.0 limit, 0.25 reservation
- **Networking**: Custom bridge network
- **Storage**: Persistent volumes for data/logs

## Security Notes

- Containers run as non-root user (UID 1000)
- Full directory access (including `.git`) for development
- API keys passed through environment variables
- Network isolation via custom bridge network
- No external network access restrictions

## Performance

- **Startup time**: ~30-60 seconds per container
- **Resource usage**: ~2GB RAM per container
- **Build time**: ~5-10 minutes first time
- **Sync latency**: Real-time (bind mount)

## Integration

This system integrates seamlessly with:
- Existing CLI-Agent workflows
- VS Code development
- Git operations
- Python development tools
- Audio/TTS functionality
- All existing CLI-Agent features

The multi-container system provides the foundation for:
- Parallel AI model testing
- Load balancing
- Development/staging environments
- Multi-user scenarios
- Distributed processing