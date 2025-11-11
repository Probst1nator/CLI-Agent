# Docker & Deployment Documentation

Docker and deployment configurations for CLI-Agent, including isolated containers, multi-instance setups, and production deployment.

## Architecture Overview

CLI-Agent supports multiple deployment modes:
- **Isolated mode**: Single-user secure container with network isolation
- **Multi-instance**: Per-browser-tab isolation with session management
- **Development**: Live source mounting for instant updates
- **Production**: Baked-in code for stable deployment

## Directory Structure

```
docker_files/
├── Dockerfile.isolated         # Production isolated container
├── docker-compose.isolated.yml # Isolated container orchestration
├── docker-compose.multi-simple.yml  # Production multi-instance
├── docker-compose.multi-dev.yml     # Development multi-instance
├── isolated_entrypoint.sh      # Container entry script
├── README_CONTAINER_MANAGER_SECURITY.md
├── README_EGRESS_FILTERING.md
├── README_SECCOMP.md
└── deprecated/                 # Old configurations
```

## Deployment Modes

### Isolated Mode

**Single-user, network-isolated deployment**:

**Features**:
- Ubuntu 22.04 base
- Security hardening (non-root user UID 1000)
- Network isolation (internet-only access)
- Persistent Docker volumes
- Resource limits (memory, CPU)
- Health checks
- Systemd integration

**Files**:
- `Dockerfile.isolated` - Container image
- `docker-compose.isolated.yml` - Orchestration
- `isolated_entrypoint.sh` - Startup script

**Usage**:
```bash
python main.py --isolated       # First run builds image (~2-3 min)
python main.py --isolated --gui # Subsequent runs instant (~2-5 sec)
```

**Source sync**:
- Source code auto-syncs into container
- `~/.cli-agent/` mounted for persistence
- Current directory at `/workspace`

**Rebuild**:
```bash
docker rmi cli-agent:isolated-cache
```

See: `docker_files/docs/ISOLATED_MODE.md`

### Multi-Instance (Production)

**Per-browser-tab isolation with session management**:

**Features**:
- 4 independent CLI-Agent instances
- Ports 7860-7863
- Automatic session routing
- Round-robin load balancing
- Independent instance configuration
- Baked-in code (no source mounting)

**File**: `docker-compose.multi-simple.yml`

**Usage**:
```bash
docker-compose -f docker_files/docker-compose.multi-simple.yml up -d
```

**Session management**:
- Each browser tab gets unique session
- Session tracked in backend
- Isolated conversation contexts

See: `docker_files/docs/MULTI_INSTANCE_SETUP.md`

### Multi-Instance (Development)

**Development mode with live source updates**:

**Features**:
- 4 independent instances
- Ports 7870-7873 (dev ports, avoiding conflicts)
- **Live source mounting**: Instant code updates
- No rebuild required for changes
- Per-browser-tab isolation

**File**: `docker-compose.multi-dev.yml`

**Usage**:
```bash
docker-compose -f docker_files/docker-compose.multi-dev.yml up -d
```

**Development workflow**:
1. Start containers with dev compose
2. Edit source code locally
3. Changes reflected instantly in containers
4. Test across multiple sessions simultaneously

See: `docker_files/docs/DEVELOPMENT_WORKFLOW.md`

## Container Security

### Network Isolation

**Isolated mode security**:
- Internet-only access (no LAN)
- UFW firewall configuration
- Egress filtering
- Port restrictions

**Accessible services**:
- AI APIs (Anthropic, OpenAI, Google, Groq)
- Ollama (if configured)
- Web search APIs
- Public internet

**Blocked**:
- LAN devices
- Internal services
- Host machine (except mounted volumes)

See: `docker_files/docs/NETWORK_ACCESS.md`

### Container Manager Security

**Docker security hardening**:
- Non-root user (UID 1000)
- Capability dropping
- Read-only filesystem where possible
- Resource limits

See: `docker_files/docs/README_CONTAINER_MANAGER_SECURITY.md`

### Egress Filtering

**Outbound traffic control**:
- Allow specific domains
- Block unauthorized outbound
- Log egress attempts

See: `docker_files/docs/README_EGRESS_FILTERING.md`

### Seccomp Profiles

**Syscall filtering**:
- Restricted system calls
- Security profile enforcement

See: `docker_files/docs/README_SECCOMP.md`

## Systemd Integration

### Service Management

**Isolated mode as systemd service**:

**Setup**:
```bash
./scripts/setup_isolated_container.sh
```

**Service file**: `scripts/cli-agent-isolated.service`

**Features**:
- Auto-restart on boot
- Proper dependency management (docker.service)
- Timeout handling
- Docker Compose integration

**Commands**:
```bash
sudo systemctl start cli-agent-isolated
sudo systemctl stop cli-agent-isolated
sudo systemctl status cli-agent-isolated
sudo journalctl -u cli-agent-isolated -f
```

**Verification**:
```bash
./scripts/verify_isolated_setup.sh
```

## Docker Images

### Base Image

**Ubuntu 22.04**:
- Python 3.8+
- System dependencies
- Security updates

### Dependencies

**Installed in container**:
- All Python packages from `requirements.txt`
- System audio libraries (for TTS)
- PDF processing tools
- Image processing libraries

### Size Optimization

- Multi-stage builds where applicable
- Layer caching
- Dependency cleanup after install

## Volumes and Persistence

### Persistent Storage

**Volumes**:
```
~/.cli-agent/              # User data persistence
  ├── projects/            # Project-specific data
  ├── logs/                # Log files
  ├── settings.json        # Configuration
  └── audit_logs/          # Session audits
```

**Source sync** (isolated mode):
- Current directory mounted at `/workspace`
- Changes sync bidirectionally

**Docker volumes** (multi-instance):
- Per-instance volumes for isolation
- Independent data storage

## Port Mapping

### Isolated Mode

- **7861**: Web UI (mapped from container 7860)

### Multi-Instance (Production)

- **7860-7863**: 4 instance web UIs

### Multi-Instance (Development)

- **7870-7873**: 4 dev instance web UIs

## Health Checks

Container health monitoring:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:7860"]
  interval: 30s
  timeout: 10s
  retries: 3
```

## Resource Limits

**Memory limits**:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

**CPU limits**: Configurable per deployment

## Environment Configuration

### Environment Variables

Passed to containers:
- API keys (Anthropic, OpenAI, Google, etc.)
- Configuration flags
- Feature toggles

### .env File

**Template**: `.env.example`

**Container usage**:
```yaml
env_file:
  - .env
```

## Troubleshooting

### Container Logs

```bash
docker logs <container_name>
docker-compose logs -f
```

### Debug Mode

```bash
docker exec -it <container_name> bash
python main.py --debug
```

### Network Issues

Check: `docker_files/docs/NETWORK_ACCESS.md`

### Rebuild Containers

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Deployment Workflows

### Development Workflow

1. Use `docker-compose.multi-dev.yml`
2. Edit code locally
3. Test in containers instantly
4. Commit when ready

### Production Deployment

1. Use `docker-compose.multi-simple.yml` or `isolated`
2. Build with specific version tag
3. Deploy to production
4. Monitor with health checks

See: `docker_files/docs/DEVELOPMENT_WORKFLOW.md`

## Deprecated Configurations

**Archived in** `docker_files/deprecated/`:
- Single-instance dev setup
- Traefik-based multi-instance
- Old configurations superseded by current implementations

## Setup Scripts

### Automated Setup

**Isolated container**:
```bash
./scripts/setup_isolated_container.sh
```

**Features**:
- Prerequisite checking
- Image building
- Environment file creation
- Systemd service installation
- Colored output and status reporting

### Verification

**Verify setup**:
```bash
./scripts/verify_isolated_setup.sh
```

**Checks**:
- Systemd service status
- Container status
- Web UI accessibility
- Dependencies
- Port mapping
- Firewall configuration
- Internet connectivity

## Related Documentation

- Isolated mode: `docker_files/docs/ISOLATED_MODE.md`
- Multi-instance: `docker_files/docs/MULTI_INSTANCE_SETUP.md`
- Development workflow: `docker_files/docs/DEVELOPMENT_WORKFLOW.md`
- Network access: `docker_files/docs/NETWORK_ACCESS.md`
- Deployment: `docker_files/docs/DEPLOYMENT.md`
- Legacy architecture: `docker_files/docs/VISUALRESPONSE_ARCHITECTURE.md`
- Container security: `docker_files/docs/README_CONTAINER_MANAGER_SECURITY.md`
- Egress filtering: `docker_files/docs/README_EGRESS_FILTERING.md`
- Seccomp: `docker_files/docs/README_SECCOMP.md`
