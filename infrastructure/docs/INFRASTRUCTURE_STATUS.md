# Infrastructure Status

## Current Infrastructure State (2025-10-16)

### Production Status: **EXPERIMENTAL** ⚠️

The multi-user Docker infrastructure exists but is currently in **experimental/development** status.

## Quick Decision Guide

**Which deployment mode should I use?**

```
┌─ Just testing code locally?
│  → Use: python main.py
│  → See: CLAUDE.md
│
┌─ Want isolated Docker runs for security?
│  → Use: python main.py --isolated
│  → See: docs/ISOLATED_MODE.md
│  → First run: ~2-3 min (builds image)
│  → Next runs: ~2-5 sec (instant)
│
┌─ Need always-on containerized service?
│  → Use: systemd + docker-compose
│  → See: docs/experimental/DOCKER_PERSISTENT_SETUP.md
│  → Auto-starts on reboot
│
┌─ Need multiple sessions for development?
│  → Use: Session manager + 3 instances
│  → See: docs/experimental/MULTI_SESSION_SETUP.md
│  → Lighter than full multi-user
│
└─ Need production multi-tenant SaaS?
   → Use: Full infrastructure (Traefik + Container Manager)
   → See: docs/experimental/QUICK_START_MULTI_USER.md
   → Status: EXPERIMENTAL ⚠️
   → Requires extensive testing
```

**Recommendation:** Start with `python main.py --isolated` for most use cases.

---

## Available Infrastructure

### 1. Single-User Mode (Stable ✅)
**Status:** Production-ready, actively used

```bash
python main.py              # CLI mode
python main.py --gui        # Web interface (localhost)
python main.py --api        # API server only
```

**Use cases:**
- Local development
- Personal use
- Single-user deployments
- Testing and debugging

---

### 2. Multi-User Docker Infrastructure (Experimental ⚠️)
**Status:** Experimental, not production-tested

**Components:**
- `docker-compose.infrastructure.yml` - Multi-user setup
- Traefik reverse proxy
- Container Manager (FastAPI)
- Nginx frontend
- Per-user isolated Docker containers

**Build & Deploy:**
```bash
./saas_scripts/build_all.sh    # Build Docker images
./saas_scripts/deploy_infrastructure.sh       # Deploy infrastructure
```

**Architecture:**
```
Internet → Traefik (<YOUR_SERVER_IP>)
           ├→ Frontend (Nginx) - Static files
           ├→ Container Manager - UUID-based routing
           └→ Backend Containers - Per-user isolation
```

**Features:**
- UUID-based container isolation
- 7-day data retention (configurable)
- Multi-tab support per user
- Automatic container lifecycle management

**Known Status:**
- Last containers active: ~4 weeks ago
- Binds to local IP: `<YOUR_SERVER_IP>`
- Not currently deployed to production
- Requires significant testing before production use

**Documentation:**
- [Multi-User Architecture](MULTI_USER_ARCHITECTURE.md) (EXPERIMENTAL)
- [Deployment Guide](DEPLOYMENT.md) (EXPERIMENTAL)
- [Quick Start](experimental/QUICK_START_MULTI_USER.md) (EXPERIMENTAL)
- [Implementation Summary](experimental/MULTI_USER_IMPLEMENTATION_SUMMARY.md)

---

## GitHub Actions Workflows

### deploy-production.yml
**Status:** DISABLED (manual trigger only)

**Trigger:** Manual workflow_dispatch only

**Actions:**
- Builds multi-arch Docker images (amd64, arm64)
- Pushes to GitHub Container Registry
- SSH deploys to production server
- Uses `docker-compose.infrastructure.yml`

**Note:** Currently disabled to prevent accidental deployments

### deploy-staging.yml
**Status:** ACTIVE

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `release` branch
- Manual workflow_dispatch

**Actions:**
- Runs `python run_tests.py --quick`
- Runs linting (ruff, black)
- Builds staging Docker images
- Deploys to staging server (if configured)

---

## Production Deployment Options

### Option 1: Use Single-User Mode (Recommended)
Most production use cases are better served by single-user mode behind a reverse proxy:

```bash
# On production server
python main.py --api --port 8000
```

Then use Nginx/Caddy to:
- Handle SSL/TLS
- Add authentication
- Rate limiting
- Load balancing

### Option 2: Multi-User Infrastructure (Experimental)
If you need true multi-tenancy:

1. **Test thoroughly** in staging environment
2. Update `docker-compose.infrastructure.yml` for your network
3. Configure secrets in GitHub Actions
4. Enable `deploy-production.yml` workflow
5. Deploy via workflow or `saas_scripts/deploy_infrastructure.sh`

**Required secrets:**
- `DEPLOY_HOST` - Production server IP/domain
- `DEPLOY_USER` - SSH user
- `DEPLOY_SSH_KEY` - SSH private key
- `DEPLOY_PATH` - Deployment directory
- All API keys (ANTHROPIC, OPENAI, etc.)

---

## Recommendations

### For Production SaaS:
1. **Start with single-user mode** behind reverse proxy
2. Test multi-user infrastructure extensively in staging
3. Monitor resource usage (Docker containers can accumulate)
4. Implement proper authentication and rate limiting
5. Set up monitoring and alerting

### For Development:
1. Use single-user mode locally: `python main.py --gui`
2. Test multi-user mode if building multi-tenant features
3. Use `saas_scripts/deploy_infrastructure.sh` for local Docker testing

---

## Next Steps to Production

If deploying multi-user infrastructure to production:

- [ ] Test multi-user setup in staging environment
- [ ] Configure production server secrets
- [ ] Update IP bindings in `docker-compose.infrastructure.yml`
- [ ] Set up SSL/TLS certificates (Let's Encrypt)
- [ ] Enable production deployment workflow
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure backup strategy
- [ ] Test failover and recovery
- [ ] Load testing with expected user count
- [ ] Security audit of container isolation

---

## More Information

For production deployment questions, see:
- [DEPLOYMENT.md](DEPLOYMENT.md) - Detailed deployment guide (EXPERIMENTAL)
- [MULTI_USER_ARCHITECTURE.md](MULTI_USER_ARCHITECTURE.md) - Architecture details (EXPERIMENTAL)
- [experimental/](experimental/) - All experimental features and deployment modes
- [ISOLATED_MODE.md](ISOLATED_MODE.md) - Recommended isolated Docker mode (STABLE)
