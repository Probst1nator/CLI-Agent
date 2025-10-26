# Container Manager Security Configuration

## Overview

The Container Manager now runs as a non-root user (`containermgr`, UID 1000, GID 999) for improved security. This follows the principle of least privilege and reduces the attack surface if the container is compromised.

## Why This Matters

Running as root inside a container with Docker socket access is a **critical security risk**:
- If the container is compromised, the attacker has root access
- Combined with Docker socket access, this allows complete host compromise
- Non-root execution limits damage from potential exploits

## Host Configuration Required

### Step 1: Verify Docker Group GID

The container's docker group (GID 999) must match the host's docker group GID for socket access.

Check your host's docker group GID:

```bash
getent group docker
```

Example output:
```
docker:x:999:user1,user2
        ^^^
        This is the GID
```

### Step 2A: If Host Docker GID is 999 (Most Common)

No changes needed! The Dockerfile is already configured correctly. Proceed to rebuild the image.

### Step 2B: If Host Docker GID is Different

You have two options:

**Option 1: Adjust the Dockerfile (Recommended)**

Edit `docker_files/Dockerfile.container-manager` line 34 to match your host's GID:

```dockerfile
# If your host docker GID is 998:
RUN groupadd -g 998 docker || true && \
    useradd -m -u 1000 -g 998 -s /bin/bash containermgr && \
    chown -R containermgr:docker /app /data
```

**Option 2: Change Host Docker GID (Not Recommended)**

This affects all users in the docker group on the host:

```bash
sudo groupmod -g 999 docker
# Restart Docker daemon
sudo systemctl restart docker
```

### Step 3: Rebuild the Container Manager Image

After making any GID adjustments:

```bash
cd /path/to/CLI-Agent
docker build -f docker_files/Dockerfile.container-manager -t cli-agent-container-manager:latest .
```

### Step 4: Verify Permissions

After starting the container, verify it's running as non-root:

```bash
# Get container ID
CONTAINER_ID=$(docker ps | grep container-manager | awk '{print $1}')

# Check user
docker exec $CONTAINER_ID whoami
# Should output: containermgr

# Check UID/GID
docker exec $CONTAINER_ID id
# Should output: uid=1000(containermgr) gid=999(docker) groups=999(docker)

# Verify Docker socket access
docker exec $CONTAINER_ID python3 -c "import docker; print(docker.from_env().ping())"
# Should output: True
```

## Troubleshooting

### Error: "Permission denied" accessing /var/run/docker.sock

**Cause:** GID mismatch between container and host

**Solution:**
1. Check host docker GID: `getent group docker`
2. Update Dockerfile with correct GID
3. Rebuild image

### Error: "Cannot write to /data"

**Cause:** Volume permissions not set correctly

**Solution:**

```bash
# On host, set correct ownership of mounted volume
sudo chown -R 1000:999 /path/to/data/volume
```

Or update docker-compose to set the volume permissions:

```yaml
services:
  container-manager:
    volumes:
      - type: volume
        source: container-manager-data
        target: /data
        volume:
          nocopy: false
    user: "1000:999"
```

### Error: "Cannot create containers"

**Cause:** Docker socket not accessible

**Solution:**

Verify socket is mounted with correct permissions:

```yaml
services:
  container-manager:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    # Ensure the socket is readable by GID 999
```

On the host, check socket permissions:

```bash
ls -l /var/run/docker.sock
# Should show: srw-rw---- 1 root docker ... /var/run/docker.sock
#                           ^^^^^
#                           Group is docker
```

## Security Benefits

✅ **Reduced Attack Surface**: Non-root user limits damage from exploits
✅ **Defense in Depth**: Multiple security layers instead of single root user
✅ **Audit Trail**: UID/GID tracking for forensics
✅ **Compliance**: Meets security best practices for containerized applications

## Known Limitations

⚠️ **Docker Socket Access Still Required**: The container needs Docker socket access to manage user containers. This is an inherent risk of the architecture.

⚠️ **Not Complete Isolation**: While better than root, a compromised container with Docker socket access can still escalate privileges.

**For production deployments**, consider:
- VM-level isolation (Firecracker, Kata Containers)
- Dedicated host for Container Manager
- Network segmentation
- Comprehensive monitoring and alerting

## Next Steps

After completing Phase 1 (Quick Wins), proceed to Phase 2 for additional hardening:
- TLS/HTTPS configuration
- Security profiles (AppArmor, seccomp)
- Network isolation
- Resource quotas

See `SECURITY_WARNING.md` for the complete security roadmap.
