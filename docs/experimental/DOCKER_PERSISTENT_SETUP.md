# CLI-Agent Isolated Container Setup

A complete solution for running CLI-Agent in a secure, isolated Docker container with browser UI exposed to your local network.

## Features

- **🌐 Network Exposed**: Browser UI accessible from any device on your local network
- **🔒 Fully Isolated**: Container has no access to host's local network (only internet)
- **🔄 Auto-Restart**: Automatically starts on system reboot via systemd
- **💾 Persistent Data**: All data, logs, and configuration persist across restarts
- **🛡️ Security**: Runs as non-root user with minimal privileges

## Quick Setup

### 1. Run the Setup Script

```bash
cd /home/prob/repos/CLI-Agent
./scripts/setup_isolated_container.sh
```

This script will:
- Build the isolated Docker image
- Create necessary configuration files
- Install and enable the systemd service
- Start the container
- Display access information

### 2. Configure Firewall (if UFW is active)

If you have UFW firewall enabled, allow access to port 7861:

```bash
sudo ufw allow 7861/tcp
```

### 3. Configure API Keys

Edit the `.env` file with your actual API keys:

```bash
nano .env
```

Add your keys:
```env
OPENAI_API_KEY=your_actual_openai_key
ANTHROPIC_API_KEY=your_actual_anthropic_key
GOOGLE_API_KEY=your_actual_google_key
GROQ_API_KEY=your_actual_groq_key
```

### 4. Restart to Apply API Keys

```bash
sudo systemctl restart cli-agent-isolated
```

## Access the Interface

Once running, access the CLI-Agent web interface at:
- **From host machine**: http://localhost:7861
- **From other devices**: http://YOUR_HOST_IP:7861

Replace `YOUR_HOST_IP` with your actual host machine's IP address.

## Network Isolation Details

The container is configured with:
- **✅ Internet Access**: Can reach external APIs and services
- **❌ Local Network Access**: Cannot access other devices on your network
- **❌ Host Access**: Cannot access host machine's services
- **✅ Exposed Port**: Only port 7861 is exposed for the web UI

This provides security while maintaining functionality for AI API calls.

## Management Commands

### Service Management
```bash
# Check status
sudo systemctl status cli-agent-isolated

# View live logs
sudo journalctl -u cli-agent-isolated -f

# Stop the service
sudo systemctl stop cli-agent-isolated

# Start the service
sudo systemctl start cli-agent-isolated

# Restart the service
sudo systemctl restart cli-agent-isolated

# Disable auto-start on boot
sudo systemctl disable cli-agent-isolated

# Re-enable auto-start on boot
sudo systemctl enable cli-agent-isolated
```

### Docker Management
```bash
# View running container
docker ps | grep cli-agent-isolated

# View container logs
docker logs cli-agent-isolated -f

# Enter the container
docker exec -it cli-agent-isolated /bin/bash

# Manual container management (if service is stopped)
cd /home/prob/repos/CLI-Agent
docker compose -f docker_files/docker-compose.isolated.yml up -d
docker compose -f docker_files/docker-compose.isolated.yml down
```

## File Structure

```
CLI-Agent/
├── docker_files/
│   ├── Dockerfile.isolated           # Isolated container image
│   └── docker-compose.isolated.yml   # Container orchestration
├── scripts/
│   ├── setup_isolated_container.sh   # Automated setup script
│   └── cli-agent-isolated.service    # Systemd service definition
└── .env                              # API keys and configuration
```

## Configuration Files

### docker-compose.isolated.yml
- Defines the isolated container configuration
- Sets up network isolation and port mapping
- Configures persistent volumes and restart policy

### Dockerfile.isolated
- Creates the container image with CLI-Agent
- Installs dependencies including aiohttp for web server
- Security hardening with non-root user
- Direct Python execution (no shell wrapper)

### cli-agent-isolated.service
- Systemd service for auto-start on boot
- Manages container lifecycle
- Provides logging and restart capabilities

## Security Features

### Container Security
- **Non-root user**: Runs as UID 1000 (cliagent)
- **Read-only where possible**: Minimal write permissions
- **No privilege escalation**: Security hardened
- **Resource limits**: CPU and memory constraints

### Network Security
- **Isolated bridge network**: Separate from host network
- **No local network access**: Cannot reach 192.168.x.x, 10.x.x.x, etc.
- **Internet only**: Can reach external APIs and services
- **Single exposed port**: Only web UI port accessible

### Data Security
- **Persistent volumes**: Data stored in Docker volumes
- **No host filesystem access**: Cannot read/write host files
- **Encrypted API communication**: All AI API calls use HTTPS

## Troubleshooting

### Container Won't Start
```bash
# Check service status
sudo systemctl status cli-agent-isolated

# View service logs
sudo journalctl -u cli-agent-isolated -f

# Check Docker logs
docker logs cli-agent-isolated

# Check if image exists
docker images | grep cli-agent
```

### Cannot Access Web UI
1. **Check if container is running**:
   ```bash
   docker ps | grep cli-agent-isolated
   ```

2. **Check port binding**:
   ```bash
   sudo netstat -tlnp | grep 7861
   ```

3. **Check firewall** (if using UFW):
   ```bash
   sudo ufw status
   sudo ufw allow 7861/tcp
   ```

4. **Test locally first**:
   ```bash
   curl http://localhost:7861
   ```

5. **Check container internal port**:
   ```bash
   # Web server runs on port 7860 inside container
   docker exec cli-agent-isolated curl -s http://localhost:7860/
   ```

6. **Verify aiohttp dependency**:
   ```bash
   docker exec cli-agent-isolated python3 -c "import aiohttp; print('aiohttp OK')"
   ```

### API Keys Not Working
1. **Check .env file**:
   ```bash
   cat .env
   ```

2. **Restart after changing keys**:
   ```bash
   sudo systemctl restart cli-agent-isolated
   ```

3. **Check container environment**:
   ```bash
   docker exec cli-agent-isolated env | grep API_KEY
   ```

### Web Server Shows "aiohttp not found"
This indicates missing web server dependency:

1. **Check if aiohttp is installed**:
   ```bash
   docker exec cli-agent-isolated python3 -c "import aiohttp; print('OK')"
   ```

2. **If missing, rebuild container**:
   ```bash
   cd /home/prob/repos/CLI-Agent
   docker build -f docker_files/Dockerfile.isolated -t cli-agent:isolated .
   sudo systemctl restart cli-agent-isolated
   ```

### Duplicate Command Arguments (--gui --gui)
If you see duplicate processes or startup issues:

1. **Check docker-compose command**:
   ```bash
   grep -A5 -B5 "command:" docker_files/docker-compose.isolated.yml
   ```

2. **Should be**:
   ```yaml
   command: ["python3", "main.py", "--gui"]
   ```

### Port Mapping Issues
Container uses port 7860 internally, mapped to 7861 on host:

1. **Verify port mapping**:
   ```bash
   docker port cli-agent-isolated
   ```

2. **Should show**:
   ```
   7860/tcp -> 0.0.0.0:7861
   ```

3. **Test internal port**:
   ```bash
   docker exec cli-agent-isolated curl -s http://localhost:7860/
   ```

### Container Using Too Many Resources
1. **Check resource usage**:
   ```bash
   docker stats cli-agent-isolated
   ```

2. **Modify resource limits** in `docker-compose.isolated.yml`:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 1G      # Reduce from 2G
         cpus: '0.5'     # Reduce from 1.0
   ```

3. **Restart with new limits**:
   ```bash
   sudo systemctl restart cli-agent-isolated
   ```

## Updating

### Update CLI-Agent Code
```bash
cd /home/prob/repos/CLI-Agent
git pull
docker build -f docker_files/Dockerfile.isolated -t cli-agent:isolated .
sudo systemctl restart cli-agent-isolated
```

### Update Container Configuration
```bash
# Edit configuration
nano docker_files/docker-compose.isolated.yml

# Restart to apply changes
sudo systemctl restart cli-agent-isolated
```

## Removal

To completely remove the isolated container setup:

```bash
# Stop and disable service
sudo systemctl stop cli-agent-isolated
sudo systemctl disable cli-agent-isolated

# Remove service file
sudo rm /etc/systemd/system/cli-agent-isolated.service
sudo systemctl daemon-reload

# Remove container and volumes
docker compose -f docker_files/docker-compose.isolated.yml down -v

# Remove Docker image
docker rmi cli-agent:isolated

# Remove configuration files (optional)
rm docker_files/Dockerfile.isolated
rm docker_files/docker-compose.isolated.yml
rm scripts/setup_isolated_container.sh
rm scripts/cli-agent-isolated.service
```

## Architecture & Dependencies

### Container Architecture
The isolated container uses a simplified, production-ready architecture:

- **Direct Python Execution**: No shell script wrapper for better process management
- **Clean Process Tree**: Main Python process + web server subprocess only
- **Port Mapping**: Container runs web server on port 7860, mapped to host port 7861
- **Dependencies**: All required packages including `aiohttp` for web server functionality

### Required Dependencies
The container includes all necessary dependencies in `requirements.txt`:
- **Core AI**: `openai`, `anthropic`, `groq`, `google-generativeai`, `ollama`
- **Web Server**: `aiohttp>=3.8.0` for GUI interface
- **Audio**: `pydub`, `soundfile`, `speechrecognition`, `vosk`
- **ML/Vector**: `chromadb`, `transformers`, `torch`, `scipy`, `numpy`
- **Utilities**: `prompt-toolkit`, `pyperclip`, `pexpect`, `pathspec`

## Advanced Configuration

### Custom Port
To use a different port, edit `docker-compose.isolated.yml`:
```yaml
ports:
  - "8080:7860"  # Use port 8080 instead of 7861 (container always uses 7860 internally)
```

### Additional Environment Variables
Add custom environment variables in `docker-compose.isolated.yml`:
```yaml
environment:
  - CUSTOM_SETTING=value
  - ANOTHER_VAR=another_value
```

### Resource Limits
Adjust CPU and memory limits in `docker-compose.isolated.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 4G      # Increase memory
      cpus: '2.0'     # Increase CPU
    reservations:
      memory: 1G      # Increase reservation
      cpus: '0.5'     # Increase reservation
```

### Multiple Instances
To run multiple isolated instances:
1. Copy the compose file with a different name
2. Change the container name and port mapping
3. Create a separate systemd service file

## Performance

### Startup Time
- **First run**: ~2-3 minutes (image download + build)
- **Subsequent runs**: ~30-60 seconds
- **Service restart**: ~15-30 seconds

### Resource Usage
- **Memory**: ~1-2GB (configurable)
- **CPU**: ~0.5-1.0 cores (configurable)
- **Disk**: ~2-5GB (base image + data)
- **Network**: Minimal (only AI API calls)

### Optimization Tips
1. **Reduce memory limit** if not using large models
2. **Use SSD storage** for better I/O performance
3. **Monitor resource usage** with `docker stats`
4. **Clean up old logs** periodically

## Quick Reference

### Key Information
- **Container Internal Port**: 7860 (web server)
- **Host Exposed Port**: 7861 (mapped from container)
- **Process Architecture**: Direct Python execution (no shell wrapper)
- **Web Server Dependency**: `aiohttp>=3.8.0` (required)
- **Firewall**: Port 7861/tcp must be allowed if UFW is active

### Verification Commands
```bash
# Check service status
sudo systemctl status cli-agent-isolated

# Test web UI locally
curl -s -o /dev/null -w "%{http_code}" http://localhost:7861/

# Test web UI from network
curl -s -o /dev/null -w "%{http_code}" http://YOUR_HOST_IP:7861/

# Check container processes
docker exec cli-agent-isolated ps aux

# Check firewall
sudo ufw status | grep 7861
```

## Development Environment

For development with automatic file watching and fast reloads:

### Quick Development Setup

```bash
# Start development environment with file watching
./scripts/dev_watch.sh

# Access development UI
# http://localhost:7862

# Stop development environment
./scripts/dev_watch.sh stop
```

### Development Features
- **🔄 Bind Mounts**: Source code changes are instantly available in container
- **⚡ Fast Reload**: Container restart instead of image rebuild (~5 seconds vs ~2 minutes)
- **📁 File Watching**: Automatic restart on Python file changes
- **🔗 Separate Port**: Development UI on port 7862, production on 7861
- **💾 Isolated Data**: Separate volumes for development data

### Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| **Port** | 7862 | 7861 |
| **Reload** | File watcher + restart | Manual restart |
| **Source** | Bind mount (live) | Copied into image |
| **Speed** | ~5 seconds | ~2 minutes |
| **Use Case** | Development/testing | Production deployment |

### File Watcher Options

```bash
# Fast development watcher (recommended)
./scripts/dev_watch.sh

# Full rebuild watcher (slower but thorough)
./scripts/watch_and_reload.sh
```

This setup provides a secure, isolated, and production-ready way to run CLI-Agent with network access while maintaining system security.