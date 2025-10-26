# Security Updates Deployment Guide

**Version:** 2.0 (Phase 1 + Phase 2 Security Hardening)
**Target Audience:** Deployment Engineer
**Estimated Time:** 30-60 minutes
**Downtime Required:** Yes (~5-10 minutes for container rebuild)

---

## Overview

This deployment adds 10 security improvements across input validation, rate limiting, container isolation, resource quotas, security profiles, and network filtering.

**Risk Level:** 🟡 Low risk - All changes are backwards compatible except network isolation (users can't attack each other anymore - this is intentional)

---

## Pre-Deployment Checklist

- [ ] Backup current configuration
- [ ] Verify you have root/sudo access on the host
- [ ] Check Docker is running: `docker ps`
- [ ] Note current Docker group GID: `getent group docker`
- [ ] Review `.env` file for required changes

---

## Step 1: Pull Latest Code

```bash
cd /path/to/CLI-Agent
git pull origin main
```

**Changed Files:**
- `api/container_manager.py` - Major security updates
- `api/server.py` - CORS middleware
- `.env.example` - New configuration options
- `docker_files/Dockerfile.container-manager` - Non-root user
- `docker_files/traefik/*.yml` - Rate limit updates
- Multiple new documentation and script files

---

## Step 2: Update Configuration

### 2.1 Environment Variables

Edit your `.env` file and add these new settings:

```bash
# Security Configuration
CORS_ALLOWED_ORIGINS=*                # Change to specific origins for production
MAX_CONTAINERS_PER_USER=3             # Limit containers per user
```

**Production Recommendation:**
```bash
CORS_ALLOWED_ORIGINS=https://your-domain.com,https://api.your-domain.com
```

### 2.2 Verify Docker Group GID

```bash
getent group docker
```

**Expected output:** `docker:x:999:...`

If the GID is NOT 999:
1. Edit `docker_files/Dockerfile.container-manager` line 34
2. Change `groupadd -g 999 docker` to match your host GID
3. Change `useradd -m -u 1000 -g 999` to match your host GID

---

## Step 3: Rebuild Container Manager

```bash
docker build -f docker_files/Dockerfile.container-manager \
  -t cli-agent-container-manager:latest .
```

**Expected:** Build completes successfully
**Time:** ~2-3 minutes

---

## Step 4: Restart Services

### Option A: Docker Compose (Recommended)

```bash
cd docker_files
docker-compose -f docker-compose.infrastructure.yml down
docker-compose -f docker-compose.infrastructure.yml up -d
```

### Option B: Manual Restart

```bash
# Stop container manager
docker stop cli-agent-container-manager
docker rm cli-agent-container-manager

# Restart with docker-compose or your deployment method
```

**Expected:** All services start successfully
**Verify:**
```bash
docker ps | grep cli-agent
# Should show: traefik, container-manager, frontend (if deployed)
```

---

## Step 5: Setup Egress Filtering (Recommended but Optional)

**This restricts outbound connections from user containers to only AI APIs.**

### 5.1 Prerequisites

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install iptables iptables-persistent

# RHEL/CentOS
sudo yum install iptables iptables-services
```

### 5.2 Run Setup Script

```bash
sudo bash docker_files/setup-egress-filtering.sh
```

**Expected output:**
```
[1/5] Backing up current iptables rules...
[2/5] Creating CLI-Agent iptables chain...
[3/5] Linking chain to FORWARD...
[4/5] Configuring egress filtering rules...
[5/5] Egress filtering configured successfully!
```

**What this does:**
- ✅ Blocks SSH, FTP, database ports from containers
- ✅ Whitelists AI APIs (Claude, GPT, Gemini, Groq, etc.)
- ✅ Allows DNS and package repositories
- ✅ Makes rules persistent across reboots

**To skip:** If you want to allow unrestricted outbound access, skip this step.

---

## Step 6: Verification Tests

### 6.1 Basic Health Check

```bash
# Container manager should be healthy
docker ps --filter "name=container-manager" --format "{{.Status}}"
# Expected: "Up X seconds (healthy)"

# Traefik should be running
curl -I http://localhost/api/health
# Expected: HTTP 200 or 404 (depending on your setup)
```

### 6.2 Test UUID Validation

```bash
# Valid UUID (should work)
curl -X GET http://localhost/api/manager/discover \
  -H "X-User-UUID: 550e8400-e29b-41d4-a716-446655440000"
# Expected: 200 OK, container created

# Invalid UUID (should be rejected)
curl -X GET http://localhost/api/manager/discover \
  -H "X-User-UUID: ../../../etc/passwd"
# Expected: 400 Bad Request, "Invalid UUID format"
```

### 6.3 Test Container Limits

```bash
# Create 3 containers with same UUID (should work)
UUID="test-$(date +%s)"
for i in {1..3}; do
  curl -X GET http://localhost/api/manager/discover \
    -H "X-User-UUID: ${UUID}"
  echo ""
done
# Expected: All 3 succeed

# Try 4th container (should fail)
curl -X GET http://localhost/api/manager/discover \
  -H "X-User-UUID: ${UUID}"
# Expected: 429 Too Many Requests, "Maximum containers per user reached"
```

### 6.4 Test Egress Filtering (if enabled)

```bash
# Get a running container ID
CONTAINER_ID=$(docker ps | grep "cli-agent-" | grep -v "container-manager\|traefik\|frontend" | head -1 | awk '{print $1}')

# Test AI API access (should work)
docker exec $CONTAINER_ID curl -I -m 5 https://api.anthropic.com
# Expected: HTTP 200 or 401 (reachable)

# Test blocked port (should timeout)
docker exec $CONTAINER_ID curl -m 5 telnet://example.com:22 || echo "Blocked as expected"
# Expected: Connection timeout after 5 seconds
```

### 6.5 Test Security Profiles

```bash
# Try to use blocked syscall (should fail)
docker exec $CONTAINER_ID mount -t tmpfs tmpfs /tmp 2>&1 | grep -i "operation not permitted"
# Expected: "Operation not permitted" message
```

### 6.6 Test Network Isolation

```bash
# List user networks
docker network ls | grep "cli-agent-user"
# Expected: One network per active user UUID

# Verify containers are isolated
# (Containers from different users should not be able to reach each other)
```

---

## Step 7: Monitor for Issues

### First 24 Hours

**Watch logs:**
```bash
# Container manager logs
docker logs -f cli-agent-container-manager

# Look for errors related to:
# - Network creation failures
# - Permission denied errors
# - Container creation failures
```

**Check metrics:**
```bash
# How many containers are running?
docker ps | grep cli-agent | wc -l

# How many user networks exist?
docker network ls | grep cli-agent-user | wc -l

# Check iptables rules (if egress filtering enabled)
sudo iptables -L CLI-AGENT-EGRESS -n --line-numbers
```

---

## Troubleshooting

### Issue: Container creation fails with "permission denied"

**Cause:** Docker group GID mismatch

**Fix:**
```bash
# Check host Docker GID
getent group docker

# Update Dockerfile.container-manager to match
# Then rebuild: docker build -f docker_files/Dockerfile.container-manager -t cli-agent-container-manager:latest .
```

### Issue: Containers can't reach AI APIs

**Cause 1:** DNS resolution issue
```bash
docker exec <container-id> nslookup api.anthropic.com
# If this fails, it's a DNS issue, not egress filtering
```

**Cause 2:** Egress filtering too restrictive
```bash
# Temporarily disable to test
sudo iptables -F CLI-AGENT-EGRESS

# If that fixes it, you need to add the endpoint to whitelist
# Edit docker_files/setup-egress-filtering.sh and add:
# iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d your-api.com -j ACCEPT
```

### Issue: Rate limiting too aggressive

**Symptom:** Legitimate users getting 429 errors

**Fix:**
```bash
# Edit docker_files/traefik/dynamic.yml
# Increase limits:
container-ops-limit:
  rateLimit:
    average: 10    # Was: 5
    burst: 20      # Was: 10
    period: 5m

# Restart Traefik
docker restart cli-agent-traefik
```

### Issue: Container limit too restrictive

**Symptom:** Users need more than 3 containers

**Fix:**
```bash
# Edit .env
MAX_CONTAINERS_PER_USER=5  # Or your preferred limit

# Restart container manager
docker restart cli-agent-container-manager
```

### Issue: Disk I/O limits not working

**Cause:** Wrong device path (cloud VMs often use /dev/vda or /dev/nvme0n1)

**Check:**
```bash
lsblk
# or
df -h
```

**Fix:**
```bash
# Edit api/container_manager.py, lines 344-345
# Change from:
device_write_bps=[{"Path": "/dev/sda", "Rate": 10485760}],

# To:
device_write_bps=[{"Path": "/dev/vda", "Rate": 10485760}],  # For cloud VMs
# or
device_write_bps=[{"Path": "/dev/nvme0n1", "Rate": 10485760}],  # For NVMe

# Rebuild and restart
```

### Issue: Egress filtering blocks needed service

**Fix:**
```bash
# Edit docker_files/setup-egress-filtering.sh
# Add before the default DROP rule:
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d your-service.com -j ACCEPT

# Re-run setup
sudo bash docker_files/setup-egress-filtering.sh
```

---

## Rollback Procedure

If you need to rollback to the previous version:

### Quick Rollback (Egress Filtering Only)

```bash
sudo bash docker_files/remove-egress-filtering.sh
```

### Full Rollback

```bash
# 1. Stop current services
docker-compose -f docker_files/docker-compose.infrastructure.yml down

# 2. Checkout previous version
git checkout <previous-commit>

# 3. Rebuild container manager
docker build -f docker_files/Dockerfile.container-manager -t cli-agent-container-manager:latest .

# 4. Restart services
docker-compose -f docker_files/docker-compose.infrastructure.yml up -d

# 5. Remove egress filtering
sudo bash docker_files/remove-egress-filtering.sh
```

---

## What Changed (Summary)

### Security Features Added

1. **UUID Validation** - Blocks injection attacks
2. **CORS Configuration** - Controls API access origins
3. **Rate Limiting** - 5 container ops per 5 minutes
4. **Dashboard Localhost-Only** - Not exposed to network
5. **Container Manager Non-Root** - Runs as UID 1000
6. **Container Creation Limits** - Max 3 per user
7. **Resource Quotas** - PID (100), Disk I/O (10MB/s)
8. **Network Isolation** - Per-user isolated networks
9. **Security Profiles** - AppArmor, seccomp, capability dropping
10. **Egress Filtering** - Whitelist-only outbound (optional)

### Breaking Changes

**None** - All changes are backwards compatible.

**Behavioral Change:** Users can no longer connect to other users' containers (this is the intended security improvement).

---

## Performance Impact

**Expected:** < 1% overall performance impact
- Container creation: +200ms (network setup)
- Request processing: +1ms (UUID validation)
- Network throughput: No measurable impact
- Egress filtering: ~1-5µs per packet

---

## Support & Documentation

### Quick Reference
- **Deployment Guide:** `DEPLOYMENT_GUIDE.md` (this file)
- **Overall Summary:** `SECURITY_IMPROVEMENTS_SUMMARY.md`
- **Detailed Phase 1:** `PHASE1_SECURITY_IMPROVEMENTS.md`
- **Detailed Phase 2:** `PHASE2_CONTAINER_SECURITY_COMPLETE.md`

### Feature-Specific Guides
- **Dashboard Access:** `docker_files/README_DASHBOARD_ACCESS.md`
- **Non-Root Setup:** `docker_files/README_CONTAINER_MANAGER_SECURITY.md`
- **Seccomp Profiles:** `docker_files/README_SECCOMP.md`
- **Egress Filtering:** `docker_files/README_EGRESS_FILTERING.md`

### Getting Help

**If deployment fails:**
1. Check logs: `docker logs cli-agent-container-manager`
2. Review troubleshooting section above
3. Check original security analysis: `SECURITY_WARNING.md`

---

## Post-Deployment

### Monitor These Metrics

```bash
# Container count
docker ps | grep cli-agent | wc -l

# User network count
docker network ls | grep cli-agent-user | wc -l

# Failed container creations (check logs)
docker logs cli-agent-container-manager 2>&1 | grep "ERROR\|FAIL"

# Blocked egress attempts (if logging enabled)
sudo tail -f /var/log/kern.log | grep "CLI-AGENT-BLOCKED"
```

### Expected Behavior

✅ Containers start successfully
✅ AI APIs are reachable from containers
✅ Users cannot create more than 3 containers
✅ Invalid UUIDs are rejected
✅ Cross-user container access is blocked
✅ SSH/database connections from containers are blocked (if egress filtering enabled)

---

## Configuration Reference

### Environment Variables (.env)

```bash
# Required (existing)
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
# ... other API keys

# New (optional)
CORS_ALLOWED_ORIGINS=*                    # Default: allow all
MAX_CONTAINERS_PER_USER=3                 # Default: 3
```

### Configurable Limits

**In `.env`:**
- Container limit per user (default: 3)
- CORS origins (default: all)

**In code (requires rebuild):**
- Memory limit: 2GB (api/container_manager.py:303)
- CPU limit: 1 core (api/container_manager.py:301-302)
- PID limit: 100 (api/container_manager.py:306)
- Disk I/O: 10 MB/s (api/container_manager.py:310-311)

**In Traefik config:**
- Rate limits (docker_files/traefik/dynamic.yml)

---

## Success Criteria

Deployment is successful when:

- [ ] All containers start and are healthy
- [ ] UUID validation rejects invalid input
- [ ] Container creation limit enforced (4th fails)
- [ ] AI APIs are reachable from containers
- [ ] Traefik dashboard NOT accessible from network
- [ ] No errors in container manager logs
- [ ] User networks are created automatically
- [ ] Egress filtering blocks SSH (if enabled)

---

## Questions?

- Review: `SECURITY_IMPROVEMENTS_SUMMARY.md`
- Detailed docs: `PHASE1_SECURITY_IMPROVEMENTS.md` and `PHASE2_CONTAINER_SECURITY_COMPLETE.md`
- Original analysis: `SECURITY_WARNING.md`

---

**Deployment Guide Version:** 2.0
**Last Updated:** 2025-10-22
**Estimated Deployment Time:** 30-60 minutes
**Risk Level:** 🟡 Low (all changes are backwards compatible)
