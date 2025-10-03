# Multi-Instance Container Setup

This document covers the multi-instance CLI-Agent setup that provides per-browser-tab isolation and default LLM configuration management.

## Overview

The multi-instance setup allows multiple isolated CLI-Agent instances to run simultaneously, with automatic session management that assigns each browser tab to its own dedicated container instance. This provides complete isolation between different user sessions.

## Architecture Components

### Core Components
1. **Session Manager** - Routes browser sessions to available instances
2. **CLI-Agent Instances** - Isolated container instances (default: 3)
3. **Docker Compose Orchestration** - Manages container lifecycle
4. **Per-Instance Configuration** - Isolated data, logs, and configuration

### Network Layout
```
┌─────────────────────────────────────────────────────────────┐
│ Session Manager (Port 7860)                                │
│ ├─ Routes new browser tabs to available instances          │
│ └─ Manages session cookies and load balancing             │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼──┐ ┌────▼────┐ ┌──▼──────┐
            │Instance 1│ │Instance 2│ │Instance 3│
            │Port 7861 │ │Port 7862 │ │Port 7863 │
            └──────────┘ └─────────┘ └─────────┘
```

## Default LLM Configuration

### Automatic Configuration
New container instances automatically receive default LLM configuration:

```json
{
  "planner": "gemini-2.5-pro",
  "router": "gemini-2.5-flash-lite-preview-06-17", 
  "fallbacks": [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17"
  ]
}
```

### Configuration Logic
The default configuration is applied during the LLM roles migration process when:
- No existing planner LLM is configured (`__planner_llm__` is null)
- No existing router LLM is configured (`__router_llm__` is null)  
- No existing fallback models are selected

### Implementation Location
File: `core/globals.py` - `load_llm_roles()` method

```python
# If no models were migrated, apply default configuration for new setups
if not selected_models and not config.get('__planner_llm__') and not config.get('__router_llm__'):
    logging.info("🔧 No existing configuration found, applying default LLM configuration")
    self.llm_roles['planner'] = 'gemini-2.5-pro'
    self.llm_roles['router'] = 'gemini-flash-lite-latest'
    self.llm_roles['fallbacks'] = [
        'gemini-flash-latest',
        'gemini-flash-lite-latest',
        'mistral-small3.2:latest'
    ]
```

## Deployment and Usage

### Starting the Multi-Instance Setup
```bash
# Navigate to the CLI-Agent directory
cd /home/prob/repos/CLI-Agent

# Start all instances using Docker Compose
docker compose -f docker_files/docker-compose.multi-simple.yml up -d

# Check status
docker compose -f docker_files/docker-compose.multi-simple.yml ps
```

### Accessing the System
1. **Main Entry Point**: `http://192.168.178.20:7860/`
   - Redirects to first available instance using round-robin
   
2. **New Browser Tab**: `http://192.168.178.20:7860/new`
   - Forces assignment to next available instance
   - Provides complete session isolation

3. **Direct Instance Access**:
   - Instance 1: `http://192.168.178.20:7861/`
   - Instance 2: `http://192.168.178.20:7862/`
   - Instance 3: `http://192.168.178.20:7863/`

### Container Configuration

#### Session Manager
```yaml
session-manager:
  image: cli-agent-session-manager
  container_name: cli-agent-session-manager
  restart: unless-stopped
  ports:
    - "7860:8000"    # Main entry point
  environment:
    - MAX_INSTANCES=3
    - INSTANCE_TIMEOUT=1800  # 30 minutes
    - BASE_PORT=7861
```

#### CLI-Agent Instance Template
```yaml
cli-agent-instance-1:
  env_file:
    - ../.env
  image: cli-agent:isolated
  container_name: cli-agent-instance-1
  restart: unless-stopped
  ports:
    - "7861:7860"  # Instance 1 direct access
  environment:
    - CLI_AGENT_GUI_MODE=true
    - CLI_AGENT_HOST=0.0.0.0
    - INSTANCE_ID=1
    - IDLE_TIMEOUT=1800  # 30 minutes
  volumes:
    - cli-agent-data-1:/workspace/data
    - cli-agent-logs-1:/workspace/logs
    - cli-agent-temp-1:/workspace/temp
    - cli-agent-config-1:/workspace/.cli-agent
    - cli-agent-user-config-1:/home/cliagent/.config/cli-agent
```

## Session Management

### Session Assignment Logic
1. **Existing Session Check**: Verify if browser has valid session cookie
2. **Round-Robin Assignment**: Assign new sessions to least loaded instance
3. **Session Isolation**: Each browser tab gets unique session ID and instance
4. **Cookie Management**: Sessions persist for 30 minutes (configurable)

### Session Manager API Endpoints

#### Core Endpoints
- `GET /` - Main entry point with session assignment
- `GET /new` - Force new session assignment
- `GET /health` - Health check endpoint

#### Management Endpoints
- `GET /api/instances` - View all instance statuses
- `GET /api/session/{session_id}` - Get session information
- `DELETE /api/session/{session_id}` - End specific session

#### Example API Response
```json
{
  "instances": {
    "1": {
      "port": 7861,
      "status": "running",
      "connections": 1,
      "created": "2025-09-25T05:50:37.891340",
      "last_activity": "2025-09-25T05:50:37.891362"
    }
  },
  "sessions": 3,
  "max_instances": 3
}
```

## Configuration Persistence

### Per-Instance Isolation
Each instance maintains completely isolated configuration:

```
Instance 1 Volumes:
├─ cli-agent-data-1:/workspace/data
├─ cli-agent-logs-1:/workspace/logs  
├─ cli-agent-temp-1:/workspace/temp
├─ cli-agent-config-1:/workspace/.cli-agent
└─ cli-agent-user-config-1:/home/cliagent/.config/cli-agent

Instance 2 Volumes:
├─ cli-agent-data-2:/workspace/data
├─ cli-agent-logs-2:/workspace/logs
├─ cli-agent-temp-2:/workspace/temp
├─ cli-agent-config-2:/workspace/.cli-agent  
└─ cli-agent-user-config-2:/home/cliagent/.config/cli-agent
```

### LLM Configuration Storage
- **Location**: `/home/cliagent/.config/cli-agent/llm_config.json`
- **Format**: Unified roles format
- **Persistence**: Survives container restarts
- **Isolation**: Each instance has independent configuration

### Permission Handling
Container directories are properly configured with `cliagent:cliagent` ownership to ensure:
- ✅ Configuration files can be written
- ✅ Settings persist across restarts
- ✅ No permission denied errors

## Resource Management

### Container Resources
Each instance is limited to:
```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

### Health Monitoring
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:7860/", "||", "exit", "1"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Auto-restart Functionality
- **Connection Monitoring**: Tracks WebSocket connections per instance
- **Idle Timeout**: Restarts after 5 minutes of inactivity (configurable)
- **Graceful Restart**: Preserves data and configuration during restart

## Monitoring and Troubleshooting

### Log Access
```bash
# View session manager logs
docker logs cli-agent-session-manager

# View specific instance logs
docker logs cli-agent-instance-1

# Follow logs in real-time
docker logs -f cli-agent-instance-2
```

### Health Checks
```bash
# Check all container statuses
docker compose -f docker_files/docker-compose.multi-simple.yml ps

# Test session manager API
curl -s "http://192.168.178.20:7860/api/instances" | python3 -m json.tool

# Test individual instance
curl -s "http://192.168.178.20:7861/api/llms" | python3 -m json.tool
```

### Common Issues

#### Configuration Not Persisting
```bash
# Check directory permissions
docker exec cli-agent-instance-1 ls -la /home/cliagent/.config/cli-agent/

# Expected: cliagent:cliagent ownership
# If root:root, fix with:
docker exec --user root cli-agent-instance-1 chown -R cliagent:cliagent /home/cliagent/.config/cli-agent
```

#### Session Manager Redirect Issues
```bash
# Test redirect functionality
curl -v -L "http://192.168.178.20:7860/"

# Check for proper host header extraction in logs
docker logs cli-agent-session-manager | grep "host header"
```

## Scaling and Customization

### Adding More Instances
To add a 4th instance, modify `docker-compose.multi-simple.yml`:

```yaml
cli-agent-instance-4:
  # Copy configuration from existing instance
  # Change ports: "7864:7860"
  # Update INSTANCE_ID=4
  # Create new volume mappings: cli-agent-*-4
```

### Customizing Default Configuration
Modify the default configuration in `core/globals.py`:

```python
# Update these values to change defaults
self.llm_roles['planner'] = 'your-preferred-planner-model'
self.llm_roles['router'] = 'your-preferred-router-model'
self.llm_roles['fallbacks'] = ['model1', 'model2', 'model3']
```

### Environment Configuration
Key environment variables in `.env`:

```bash
# Ollama hosts for local models
OLLAMA_HOST=192.168.178.37,192.168.178.25,localhost

# AI service API keys
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
```

## Related Documentation

- [Network Access](NETWORK_ACCESS.md) - Container network security
- [Docker Isolated](DOCKER_ISOLATED.md) - Single container setup
- [Architecture](ARCHITECTURE.md) - Overall system design
- [Persistent Files](PERSISTENT_FILES.md) - Data persistence patterns