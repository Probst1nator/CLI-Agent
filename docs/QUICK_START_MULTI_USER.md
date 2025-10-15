# Quick Start: Multi-User Deployment

**Time to deploy:** ~10 minutes
**Target:** Production multi-user web service

---

## Prerequisites

- Ubuntu 22.04+ or similar Linux distribution
- Docker 24.0+ and Docker Compose 2.20+
- 8GB+ RAM, 50GB+ disk space
- `.env` file with at least one LLM API key

---

## 3-Step Deployment

### Step 1: Configure Environment

```bash
# Clone repository (if not already)
git clone <repository-url>
cd CLI-Agent

# Create environment file
cp .env.example .env
nano .env  # Add your API keys

# Required: At least one of these
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
```

### Step 2: Build & Deploy

```bash
# Build all Docker images
./scripts/build_all.sh

# Deploy infrastructure
./scripts/deploy.sh
```

**Expected output:**
```
✅ Deployment Complete!

Access points:
  - CLI-Agent:          http://localhost
  - Traefik Dashboard:  http://localhost:8080
  - Container Manager:  http://localhost/api/manager/containers
```

### Step 3: Verify

```bash
# Run automated tests
./scripts/test_multi_user.sh
```

**Expected result:** All tests pass ✅

---

## What Happens on First Access?

1. **User opens** `http://localhost` in browser
2. **Frontend** generates UUID and stores in localStorage
3. **Container Manager** receives discovery request with UUID
4. **Backend container** is created for that UUID
5. **User connects** to their isolated CLI-Agent instance

Each new user (UUID) gets their own container automatically.

---

## Common Commands

### Check Status

```bash
# View all services
docker-compose -f docker_files/docker-compose.infrastructure.yml ps

# Check Container Manager health
curl http://localhost/api/manager/health

# List active backend containers
curl http://localhost/api/manager/containers
```

### View Logs

```bash
# All services
docker-compose -f docker_files/docker-compose.infrastructure.yml logs -f

# Container Manager only
docker logs -f cli-agent-container-manager

# Traefik only
docker logs -f cli-agent-traefik
```

### Stop Services

```bash
# Stop all (preserves data)
docker-compose -f docker_files/docker-compose.infrastructure.yml down

# Stop and remove all data
docker-compose -f docker_files/docker-compose.infrastructure.yml down -v
```

### Restart Services

```bash
# Restart all
docker-compose -f docker_files/docker-compose.infrastructure.yml restart

# Restart specific service
docker-compose -f docker_files/docker-compose.infrastructure.yml restart container-manager
```

---

## Troubleshooting

### Issue: Port 80 already in use

```bash
# Check what's using port 80
sudo netstat -tlnp | grep :80

# Stop conflicting service (example: Apache)
sudo systemctl stop apache2

# Restart deployment
./scripts/deploy.sh
```

### Issue: Services show unhealthy

```bash
# Wait 30 seconds for health checks to pass
sleep 30
docker-compose -f docker_files/docker-compose.infrastructure.yml ps

# If still unhealthy, check logs
docker logs cli-agent-container-manager
docker logs cli-agent-traefik
```

### Issue: Can't create containers

```bash
# Check Docker permissions
ls -l /var/run/docker.sock

# Ensure current user in docker group
sudo usermod -aG docker $USER
# Log out and back in

# Restart Container Manager
docker-compose -f docker_files/docker-compose.infrastructure.yml restart container-manager
```

### Issue: Disk space running out

```bash
# Check usage
df -h
docker system df

# Manual cleanup of old containers
curl -X POST http://localhost/api/manager/cleanup

# Remove unused Docker resources
docker system prune -a -f
```

---

## Scaling & Performance

### Increase Container Manager Workers

Edit `docker_files/Dockerfile.container-manager`:

```dockerfile
# Change from 4 to 8 workers
CMD ["gunicorn", "-w", "8", ...]
```

Rebuild and redeploy:
```bash
./scripts/build_all.sh
docker-compose -f docker_files/docker-compose.infrastructure.yml up -d --force-recreate container-manager
```

### Adjust Retention Period

Edit `docker_files/docker-compose.infrastructure.yml`:

```yaml
container-manager:
  environment:
    - DEFAULT_RETENTION_DAYS=14  # Change from 7
```

Restart:
```bash
docker-compose -f docker_files/docker-compose.infrastructure.yml restart container-manager
```

### Monitor Resource Usage

```bash
# Real-time stats
docker stats

# Check number of active containers
docker ps --filter "name=cli-agent-" --format "{{.Names}}" | wc -l

# Check Container Manager metrics
curl http://localhost/api/manager/containers | jq '.count'
```

---

## Production Checklist

Before deploying to production:

- [ ] Enable HTTPS in Traefik configuration
- [ ] Secure Traefik dashboard (add auth or disable)
- [ ] Set up firewall rules (allow 80/443 only)
- [ ] Configure log rotation
- [ ] Set up automated backups
- [ ] Adjust retention period for your use case
- [ ] Monitor disk space and set up alerts
- [ ] Test disaster recovery procedure

**See:** [Deployment Guide](DEPLOYMENT.md) for production configuration details.

---

## Next Steps

- **Architecture Details:** [Multi-User Architecture](MULTI_USER_ARCHITECTURE.md)
- **Full Deployment Guide:** [Deployment Guide](DEPLOYMENT.md)
- **Backend/Frontend Separation:** [Backend Frontend Architecture](BACKEND_FRONTEND_ARCHITECTURE.md)
- **Logging System:** [Logging](LOGGING.md)

---

## Need Help?

1. Check logs: `docker-compose -f docker_files/docker-compose.infrastructure.yml logs`
2. Run tests: `./scripts/test_multi_user.sh`
3. Review [Deployment Guide](DEPLOYMENT.md) troubleshooting section
4. Check Container Manager health: `curl http://localhost/api/manager/health`

---

**Deployment successful?** 🎉

Your CLI-Agent multi-user service is now running at `http://localhost`

Each user will automatically get their own isolated container on first visit.
