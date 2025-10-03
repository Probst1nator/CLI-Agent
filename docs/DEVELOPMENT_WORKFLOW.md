# Development Workflow Guide

## Overview

This project supports two deployment modes optimized for different use cases:
- **Development Mode**: Instant code updates via bind mounts (Option 2)
- **Production Mode**: Fully isolated containers with baked-in code (Option 1)

---

## Development Mode (Recommended for Active Development)

### Quick Start

```bash
# Stop production containers
docker compose -f docker_files/docker-compose.multi-simple.yml down

# Start development containers
docker compose -f docker_files/docker-compose.multi-dev.yml up -d
```

### Access Points
- **Session Manager**: http://localhost:7870
- **Instance 1**: http://localhost:7871
- **Instance 2**: http://localhost:7872
- **Instance 3**: http://localhost:7873

### How It Works

**Live Code Mounting:**
```yaml
volumes:
  - /home/prob/repos/CLI-Agent:/cli-agent:ro  # Read-only source code
```

**Update Workflow:**
1. Edit Python files in `/home/prob/repos/CLI-Agent/`
2. Restart affected container: `docker compose -f docker_files/docker-compose.multi-dev.yml restart cli-agent-dev-instance-1`
3. Changes reflected in **5-10 seconds**

**Benefits:**
- ✅ Instant feedback loop
- ✅ No Docker image rebuilds
- ✅ Low resource usage
- ✅ Rapid iteration

**Trade-offs:**
- ⚠️ Containers can read host source code (read-only)
- ⚠️ Different from production deployment
- ⚠️ Separate volumes from production instances

### Container Restart Commands

```bash
# Restart single instance
docker compose -f docker_files/docker-compose.multi-dev.yml restart cli-agent-dev-instance-1

# Restart all instances
docker compose -f docker_files/docker-compose.multi-dev.yml restart

# View logs
docker compose -f docker_files/docker-compose.multi-dev.yml logs -f cli-agent-dev-instance-1
```

---

## Production Mode (Isolated Deployment)

### Quick Start

```bash
# Stop development containers
docker compose -f docker_files/docker-compose.multi-dev.yml down

# Rebuild image with latest code
docker build -f docker_files/Dockerfile.isolated -t cli-agent:isolated .

# Start production containers
docker compose -f docker_files/docker-compose.multi-simple.yml up -d --force-recreate
```

### Access Points
- **Session Manager**: http://localhost:7860
- **Instance 1**: http://localhost:7861
- **Instance 2**: http://localhost:7862
- **Instance 3**: http://localhost:7863

### How It Works

**Baked-in Code:**
```dockerfile
# Dockerfile.isolated line 44-45
COPY . /cli-agent/
WORKDIR /cli-agent
```

**Update Workflow:**
1. Edit Python files
2. Rebuild Docker image (**2-3 minutes**)
3. Recreate containers
4. Changes reflected after full rebuild

**Benefits:**
- ✅ Fully isolated from host filesystem
- ✅ Matches production deployment exactly
- ✅ Maximum security
- ✅ Reproducible builds

**Trade-offs:**
- ⚠️ Slow update cycle (2-3 minutes)
- ⚠️ Resource intensive (2GB image rebuilds)
- ⚠️ High disk usage over time

### Automated Rebuild (Optional)

The file watcher script can automatically rebuild on code changes:

```bash
# Start background watcher
nohup ./scripts/watch_and_reload.sh > /tmp/watch-and-reload.log 2>&1 &

# Monitor watcher logs
tail -f /tmp/watch-and-reload.log

# Stop watcher
pkill -f watch_and_reload.sh
```

**Note**: File watcher triggers 2-3 minute rebuilds on every `.py` file change. Only recommended for final integration testing.

---

## Recommended Hybrid Workflow

### Phase 1: Active Development
Use **Development Mode** for rapid iteration:

```bash
# Switch to dev mode
docker compose -f docker_files/docker-compose.multi-simple.yml down
docker compose -f docker_files/docker-compose.multi-dev.yml up -d

# Code → Restart (5-10 seconds) → Test → Repeat
```

### Phase 2: Pre-Production Testing
Use **Production Mode** for final validation:

```bash
# Switch to production mode
docker compose -f docker_files/docker-compose.multi-dev.yml down
docker build -f docker_files/Dockerfile.isolated -t cli-agent:isolated .
docker compose -f docker_files/docker-compose.multi-simple.yml up -d

# Test with production-identical setup
```

### Phase 3: Deployment
Production containers are ready to deploy as-is:

```bash
# Already running production setup
# Containers are fully isolated and production-ready
```

---

## Troubleshooting

### Dev containers won't start
```bash
# Check if ports 7870-7873 are available
netstat -tuln | grep -E "787[0-3]"

# Check if dev volumes exist
docker volume ls | grep cli-agent-dev
```

### Code changes not reflected
```bash
# Verify bind mount exists
docker inspect cli-agent-dev-instance-1 | grep -A5 Mounts

# Restart container to reload Python imports
docker compose -f docker_files/docker-compose.multi-dev.yml restart cli-agent-dev-instance-1
```

### Production rebuild too slow
```bash
# Switch to dev mode for active development
docker compose -f docker_files/docker-compose.multi-simple.yml down
docker compose -f docker_files/docker-compose.multi-dev.yml up -d
```

### Volumes getting mixed up
```bash
# Development volumes
docker volume ls | grep "cli-agent-dev"

# Production volumes  
docker volume ls | grep "cli-agent-multi"

# They are completely separate
```

---

## Performance Comparison

| Metric | Development Mode | Production Mode |
|--------|------------------|-----------------|
| Update Speed | 5-10 seconds | 2-3 minutes |
| Resource Usage | Low (restart only) | High (full rebuild) |
| Disk Usage | Minimal | High (image layers) |
| Security Isolation | Reduced (read-only mount) | Full (baked-in) |
| Production Match | Different | Exact |

---

## Quick Reference

```bash
# DEVELOPMENT
docker compose -f docker_files/docker-compose.multi-dev.yml up -d      # Start
docker compose -f docker_files/docker-compose.multi-dev.yml restart    # Reload code
docker compose -f docker_files/docker-compose.multi-dev.yml logs -f    # Logs
docker compose -f docker_files/docker-compose.multi-dev.yml down       # Stop

# PRODUCTION
docker build -f docker_files/Dockerfile.isolated -t cli-agent:isolated .                    # Build
docker compose -f docker_files/docker-compose.multi-simple.yml up -d --force-recreate      # Start
docker compose -f docker_files/docker-compose.multi-simple.yml logs -f                      # Logs
docker compose -f docker_files/docker-compose.multi-simple.yml down                         # Stop
```

---

## Configuration Persistence

Both modes have **separate, isolated volumes**:

**Development volumes:** `cli-agent-dev-*`
- LLM configurations saved to `/home/cliagent/.config/cli-agent`
- Persist across dev container restarts
- Independent from production

**Production volumes:** `cli-agent-multi-*`
- LLM configurations saved to `/home/cliagent/.config/cli-agent`
- Persist across prod container restarts
- Independent from development

**Note**: You can export/import configurations between environments manually if needed.
