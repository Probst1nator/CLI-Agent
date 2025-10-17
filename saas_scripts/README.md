# SaaS Deployment Scripts

This directory contains scripts for deploying CLI-Agent as a **multi-user SaaS platform** with Docker-based container isolation.

## 📋 Scripts Overview

### `build_all.sh`
**Purpose**: Build all Docker images for the multi-user infrastructure

**What it does**:
- Builds Container Manager image (orchestrates per-user containers)
- Builds Backend image (per-user CLI-Agent instances)
- Builds Frontend image (web UI served via nginx)

**Usage**:
```bash
./saas_scripts/build_all.sh
```

**Time**: ~5-10 minutes (first build), ~2-3 minutes (subsequent builds)

---

### `deploy_infrastructure.sh`
**Purpose**: Deploy the complete multi-user infrastructure stack

**What it does**:
1. Verifies `.env` file exists with API keys
2. Builds all Docker images (calls `build_all.sh`)
3. Creates Docker network for service communication
4. Starts infrastructure services:
   - **Traefik** (reverse proxy and routing)
   - **Container Manager** (per-user container orchestration)
   - **Frontend** (web UI)
5. Performs health checks on all services

**Usage**:
```bash
./saas_scripts/deploy_infrastructure.sh
```

**Access Points** (after deployment):
- Main UI: http://localhost
- Traefik Dashboard: http://localhost:8080
- Container Manager API: http://localhost/api/manager/containers

**Stop Infrastructure**:
```bash
docker-compose -f docker_files/docker-compose.infrastructure.yml down
```

---

### `release_to_production.sh`
**Purpose**: Create a production release by squashing main branch commits

**What it does**:
1. Verifies you're on `main` branch with no uncommitted changes
2. Prompts for release version/title (e.g., "v1.2.0 - Multi-user support")
3. Squashes all commits from `main` into a single commit on `production` branch
4. Tags the release
5. Pushes to `production` branch

**Usage**:
```bash
./saas_scripts/release_to_production.sh
```

**Workflow**:
```
main branch (development)
    ↓ (squash all commits)
production branch (clean history)
    ↓ (tag)
v1.2.0
```

**Requirements**:
- Must be on `main` branch
- No uncommitted changes
- Clean working directory

---

## 🚀 Quick Start Guide

### First-Time Setup

1. **Prepare environment**:
   ```bash
   cp .env.example .env
   nano .env  # Add your API keys
   ```

2. **Deploy infrastructure**:
   ```bash
   ./saas_scripts/deploy_infrastructure.sh
   ```

3. **Access the platform**:
   - Open browser: http://localhost
   - Each user gets isolated container automatically

### Development Workflow

```bash
# Make changes to code
vim main.py

# Rebuild and redeploy
./saas_scripts/deploy_infrastructure.sh

# Monitor logs
docker-compose -f docker_files/docker-compose.infrastructure.yml logs -f

# Stop when done
docker-compose -f docker_files/docker-compose.infrastructure.yml down
```

### Release Workflow

```bash
# Ensure main branch is ready
git checkout main
git pull

# Test thoroughly
./saas_scripts/deploy_infrastructure.sh
# ... test the deployment ...

# Create release
./saas_scripts/release_to_production.sh
# Enter version: v1.2.0 - Multi-user support
```

---

## 🏗️ Architecture Overview

### Infrastructure Components

```
┌─────────────────────────────────────────────────┐
│                   Traefik                       │
│            (Reverse Proxy)                      │
│              Port 80 & 8080                     │
└────────┬──────────────────────┬─────────────────┘
         │                      │
         │                      │
    ┌────▼────────┐      ┌──────▼──────────────┐
    │  Frontend   │      │ Container Manager   │
    │  (Nginx)    │      │    (FastAPI)        │
    └─────────────┘      └──────┬──────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              ┌─────▼─────┐           ┌─────▼─────┐
              │ Backend   │           │ Backend   │
              │Container 1│  ...      │Container N│
              │(User A)   │           │(User N)   │
              └───────────┘           └───────────┘
```

### Per-User Isolation

- Each user gets a **dedicated Docker container**
- Containers are created on-demand when user connects
- UUID-based identification ensures isolation
- Automatic cleanup after retention period

---

## 📊 Resource Requirements

### Minimum System Requirements

- **CPU**: 4 cores (2 for infrastructure + 2 for user containers)
- **RAM**: 8GB (2GB infrastructure + 6GB for user containers)
- **Disk**: 20GB free space
- **Docker**: Version 20.10+ with compose V2

### Per-User Container Resources

Default limits (configurable in `docker_files/docker-compose.infrastructure.yml`):

- **Memory**: 2GB per container
- **CPU**: 1.0 core per container
- **Disk**: Shared, monitored

---

## 🔧 Configuration

### Environment Variables

Required in `.env`:
```bash
# AI Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...

# Optional: Additional Configuration
BRAVE_API_KEY=...           # Web search
NVIDIA_API_KEY=...          # NVIDIA AI
```

### Infrastructure Settings

Edit `docker_files/docker-compose.infrastructure.yml` to customize:

- Container resource limits
- Network configuration
- Volume mounts
- Health check intervals
- Port mappings

---

## 🐛 Troubleshooting

### Services Won't Start

```bash
# Check service status
docker-compose -f docker_files/docker-compose.infrastructure.yml ps

# View logs
docker-compose -f docker_files/docker-compose.infrastructure.yml logs

# Restart specific service
docker-compose -f docker_files/docker-compose.infrastructure.yml restart traefik
```

### Port Already in Use

```bash
# Check what's using port 80
sudo lsof -i :80

# Stop conflicting service
sudo systemctl stop apache2  # or nginx, etc.
```

### Images Not Building

```bash
# Clean Docker cache
docker system prune -a

# Rebuild from scratch
./saas_scripts/build_all.sh
```

### Container Manager Not Responding

```bash
# Check logs
docker logs cli-agent-container-manager

# Verify Docker socket is mounted
docker inspect cli-agent-container-manager | grep docker.sock

# Restart container manager
docker-compose -f docker_files/docker-compose.infrastructure.yml restart container-manager
```

---

## 📁 Related Files

### Docker Configuration
- `docker_files/Dockerfile.backend` - Per-user container image
- `docker_files/Dockerfile.container-manager` - Container orchestration service
- `docker_files/Dockerfile.frontend` - Web UI image
- `docker_files/docker-compose.infrastructure.yml` - Infrastructure orchestration
- `docker_files/traefik/` - Traefik reverse proxy configuration

### Documentation
- `docs/MULTI_USER_ARCHITECTURE.md` - Architecture details
- `docs/DEPLOYMENT.md` - Production deployment guide
- `docs/DOCKER_MULTI_CONTAINER.md` - Container management details

### API
- `api/container_manager.py` - Container lifecycle management

---

## 🔒 Security Considerations

### User Isolation

- Each user runs in **isolated Docker container**
- No shared filesystem between users
- Network isolation between containers
- Resource limits prevent resource exhaustion

### API Keys

- Store in `.env` file (not committed to git)
- Automatically injected into containers
- No API keys in logs or container images

### Network Security

- Traefik handles SSL/TLS termination (when configured)
- Internal services not exposed to public
- Container-to-container communication secured

---

## 📈 Monitoring

### View Logs

```bash
# All services
docker-compose -f docker_files/docker-compose.infrastructure.yml logs -f

# Specific service
docker logs -f cli-agent-traefik
docker logs -f cli-agent-container-manager
docker logs -f cli-agent-frontend

# Per-user container
docker logs <container-id>
```

### Check Container Status

```bash
# Infrastructure
docker-compose -f docker_files/docker-compose.infrastructure.yml ps

# User containers
curl http://localhost/api/manager/containers

# Traefik dashboard
open http://localhost:8080
```

### Resource Usage

```bash
# Overview
docker stats

# Specific container
docker stats <container-id>
```

---

## 🧹 Cleanup

### Stop Infrastructure

```bash
docker-compose -f docker_files/docker-compose.infrastructure.yml down
```

### Remove All Containers and Volumes

```bash
docker-compose -f docker_files/docker-compose.infrastructure.yml down -v
```

### Clean Up User Containers

```bash
# Via API
curl -X DELETE http://localhost/api/manager/container/<uuid>

# Or manual cleanup
docker rm -f $(docker ps -aq --filter label=cli-agent.user-container=true)
```

### Complete Reset

```bash
# Stop everything
docker-compose -f docker_files/docker-compose.infrastructure.yml down -v

# Remove images
docker rmi cli-agent-manager:latest cli-agent-backend:latest cli-agent-frontend:latest

# Prune system
docker system prune -a --volumes
```

---

## 📚 Additional Resources

- **Architecture**: See `docs/MULTI_USER_ARCHITECTURE.md` for detailed architecture
- **Deployment**: See `docs/DEPLOYMENT.md` for production deployment guide
- **API Docs**: See `api/container_manager.py` for API endpoints
- **Main README**: See `../README.md` for general CLI-Agent documentation

---

## ⚠️ Important Notes

1. **Experimental**: Multi-user architecture is still experimental
2. **Resource Intensive**: Each user container requires significant resources
3. **API Keys**: Ensure `.env` is properly configured before deployment
4. **Docker Required**: Docker must be installed and running
5. **Production**: For production use, review security settings and resource limits

---

**For single-user local development**, use:
```bash
python main.py              # Local mode
python main.py --isolated   # Single isolated container (see docs/ISOLATED_MODE.md)
```

**For multi-user SaaS deployment**, use the scripts in this directory.
