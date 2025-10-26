# Traefik Dashboard Access

## Overview

The Traefik dashboard is **NOT exposed to the network**. It is only accessible from localhost (127.0.0.1:8080), providing maximum security by eliminating network-based attack vectors.

## Security Model

**Port Binding:** `127.0.0.1:8080:8080` (localhost only)

This means:
- ✅ **Safe:** Dashboard only accessible from the host machine itself
- ✅ **No authentication needed:** Since it's not on the network, no password required
- ✅ **No network exposure:** Cannot be accessed from other machines on the LAN
- ✅ **Protection from attacks:** Network-based attacks cannot reach the dashboard

## Accessing the Dashboard

### Option 1: From the Host Machine (Direct)

If you're on the host machine where Docker is running:

```bash
# Open in browser
http://localhost:8080/dashboard/

# Or use curl
curl http://localhost:8080/dashboard/
```

### Option 2: Via SSH Port Forwarding (Remote)

If you need to access the dashboard from a remote machine:

```bash
# From your remote machine, create SSH tunnel
ssh -L 8080:localhost:8080 user@<YOUR_SERVER_IP>

# Then open in your browser
http://localhost:8080/dashboard/
```

The SSH tunnel securely forwards port 8080 from the remote host to your local machine.

### Option 3: VS Code Remote SSH

If using VS Code Remote SSH extension:

1. Connect to the remote host
2. Go to "Ports" tab
3. Add port forwarding: Local 8080 → Remote 8080
4. Open http://localhost:8080/dashboard/ in browser

## What the Dashboard Shows

The Traefik dashboard provides real-time visibility into:

- **Routers:** HTTP routing rules and priorities
- **Services:** Backend services (Container Manager, Frontend)
- **Middlewares:** Rate limiting, CORS, security headers, compression
- **Entrypoints:** Port 80 (HTTP), Port 443 (HTTPS)
- **Health Status:** Live/healthy services
- **Metrics:** Request counts, response times

## Why Localhost-Only is Secure

Traditional approach: Dashboard on network + authentication
- ❌ Vulnerable to password attacks
- ❌ Vulnerable to authentication bypass exploits
- ❌ Requires password management and rotation
- ❌ Still exposed to network scanning

**Our approach:** Dashboard NOT on network
- ✅ **Zero network attack surface**
- ✅ **No password attacks possible**
- ✅ **No authentication bypass possible**
- ✅ **Invisible to network scans**
- ✅ **Simpler configuration**

Access control is enforced at the **network layer** (port binding), not the **application layer** (authentication). This is more secure because:
1. Fewer moving parts = fewer things that can break or be misconfigured
2. Kernel-level port binding is harder to bypass than application auth
3. No credentials to leak or get compromised

## When You Need Dashboard Access

### Development/Debugging

Use SSH port forwarding (Option 2 above) to access the dashboard when needed.

### Production Monitoring

Consider these alternatives to Traefik dashboard:
- **Prometheus + Grafana:** Better suited for production monitoring
- **ELK Stack:** Centralized logging and metrics
- **Datadog/New Relic:** Commercial monitoring solutions
- **Traefik Access Logs:** Parse `/var/log/traefik/access.log` for metrics

The dashboard is designed for development and debugging, not production monitoring.

## Exposing Dashboard to Network (Not Recommended)

If you absolutely must expose the dashboard to the network:

### Step 1: Add Authentication

Follow the instructions in the security warning document to add HTTP BasicAuth.

### Step 2: Change Port Binding

Edit `docker_files/docker-compose.infrastructure.yml`:

```yaml
# BEFORE (secure - localhost only):
- "127.0.0.1:8080:8080"

# AFTER (insecure - network exposed):
- "<YOUR_SERVER_IP>:8080:8080"
```

### Step 3: Set Strong Password

Generate a strong password and add BasicAuth middleware (see SECURITY_WARNING.md).

### Step 4: Consider the Risks

Even with authentication:
- Dashboard becomes visible to network scanners
- Vulnerable to brute force attacks
- Requires password management
- Additional attack surface

**Recommendation:** Use SSH port forwarding instead of network exposure.

## Troubleshooting

### Dashboard not accessible on localhost

**Check port binding:**
```bash
docker ps | grep traefik
# Look for: 127.0.0.1:8080->8080/tcp
```

**Check Traefik is running:**
```bash
docker logs cli-agent-traefik
```

**Check from within container:**
```bash
docker exec cli-agent-traefik wget -O- http://localhost:8080/dashboard/
```

### SSH port forwarding not working

**Check SSH connection:**
```bash
ssh -v -L 8080:localhost:8080 user@<YOUR_SERVER_IP>
# Look for "Local forwarding listening on" message
```

**Check if port 8080 is already in use locally:**
```bash
lsof -i :8080
# or
netstat -an | grep 8080
```

If port 8080 is in use locally, use a different port:
```bash
ssh -L 9090:localhost:8080 user@<YOUR_SERVER_IP>
# Then access: http://localhost:9090/dashboard/
```

## Related Documentation

- `docker_files/docker-compose.infrastructure.yml` - Port binding configuration
- `docker_files/traefik/traefik.yml` - Dashboard enablement
- `docker_files/traefik/dynamic.yml` - Routing and middleware rules
- `SECURITY_WARNING.md` - Complete security analysis

## Summary

- **Default:** Dashboard on localhost only (secure by design)
- **Access:** Use SSH port forwarding for remote access
- **No password needed:** Network isolation provides security
- **Production:** Use proper monitoring tools, not dashboard
