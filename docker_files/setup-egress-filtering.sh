#!/bin/bash
#
# Egress Network Filtering for CLI-Agent Containers
#
# This script configures iptables rules to restrict outbound connections
# from user containers to only allow necessary AI API endpoints.
#
# IMPORTANT: Run this script on the HOST machine, not in containers
# IMPORTANT: Requires root privileges
#
# Usage:
#   sudo bash docker_files/setup-egress-filtering.sh

set -e

echo "========================================="
echo "CLI-Agent Egress Filtering Setup"
echo "========================================="
echo ""

# Check for root
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: This script must be run as root"
  echo "Usage: sudo bash $0"
  exit 1
fi

# Detect Docker network interface
DOCKER_IF=$(ip route | grep "docker" | head -1 | awk '{print $3}')
if [ -z "$DOCKER_IF" ]; then
    echo "WARNING: Could not auto-detect Docker interface, using default 'docker0'"
    DOCKER_IF="docker0"
fi

echo "Docker interface detected: $DOCKER_IF"
echo ""

# Backup current iptables rules
echo "[1/5] Backing up current iptables rules..."
iptables-save > /tmp/iptables-backup-$(date +%Y%m%d-%H%M%S).rules
echo "      Backup saved to /tmp/iptables-backup-*.rules"
echo ""

# Create CLI-Agent chain if it doesn't exist
echo "[2/5] Creating CLI-Agent iptables chain..."
if ! iptables -L CLI-AGENT-EGRESS -n >/dev/null 2>&1; then
    iptables -N CLI-AGENT-EGRESS
    echo "      Chain created: CLI-AGENT-EGRESS"
else
    echo "      Chain already exists, flushing rules"
    iptables -F CLI-AGENT-EGRESS
fi
echo ""

# Jump to our chain from FORWARD
echo "[3/5] Linking chain to FORWARD..."
if ! iptables -C FORWARD -i $DOCKER_IF -j CLI-AGENT-EGRESS 2>/dev/null; then
    iptables -I FORWARD 1 -i $DOCKER_IF -j CLI-AGENT-EGRESS
    echo "      Added jump rule to FORWARD chain"
else
    echo "      Jump rule already exists"
fi
echo ""

# Apply egress filtering rules
echo "[4/5] Configuring egress filtering rules..."

# Block dangerous outbound ports from containers
echo "      Blocking dangerous ports..."
iptables -A CLI-AGENT-EGRESS -p tcp --dport 22 -j DROP  # SSH
iptables -A CLI-AGENT-EGRESS -p tcp --dport 21 -j DROP  # FTP
iptables -A CLI-AGENT-EGRESS -p tcp --dport 3306 -j DROP  # MySQL
iptables -A CLI-AGENT-EGRESS -p tcp --dport 5432 -j DROP  # PostgreSQL
iptables -A CLI-AGENT-EGRESS -p tcp --dport 6379 -j DROP  # Redis
iptables -A CLI-AGENT-EGRESS -p tcp --dport 27017 -j DROP  # MongoDB
iptables -A CLI-AGENT-EGRESS -p tcp --dport 3389 -j DROP  # RDP
iptables -A CLI-AGENT-EGRESS -p tcp --dport 5900 -j DROP  # VNC

# Allow DNS (required for name resolution)
echo "      Allowing DNS (UDP 53)..."
iptables -A CLI-AGENT-EGRESS -p udp --dport 53 -j ACCEPT
iptables -A CLI-AGENT-EGRESS -p tcp --dport 53 -j ACCEPT

# Allow HTTP/HTTPS to AI API endpoints (whitelist approach)
echo "      Whitelisting AI API endpoints..."

# Anthropic (Claude)
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d api.anthropic.com -j ACCEPT

# OpenAI (GPT)
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d api.openai.com -j ACCEPT

# Google (Gemini)
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d generativelanguage.googleapis.com -j ACCEPT
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d ai.google.dev -j ACCEPT

# Groq
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d api.groq.com -j ACCEPT

# NVIDIA
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d integrate.api.nvidia.com -j ACCEPT

# Brave Search
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d api.search.brave.com -j ACCEPT

# HuggingFace
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d huggingface.co -j ACCEPT
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d api-inference.huggingface.co -j ACCEPT

# Allow HTTP/HTTPS to common CDNs and package repositories (for pip, npm, etc.)
echo "      Allowing package repositories..."
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d pypi.org -j ACCEPT
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d files.pythonhosted.org -j ACCEPT
iptables -A CLI-AGENT-EGRESS -p tcp --dport 443 -d registry.npmjs.org -j ACCEPT

# Log blocked traffic (optional - can generate lots of logs)
# Uncomment if you want to see what's being blocked
# iptables -A CLI-AGENT-EGRESS -j LOG --log-prefix "CLI-AGENT-BLOCKED: " --log-level 4

# Default: DROP all other outbound traffic
echo "      Setting default policy to DROP..."
iptables -A CLI-AGENT-EGRESS -j DROP

echo ""
echo "[5/5] Egress filtering configured successfully!"
echo ""

# Display current rules
echo "Current CLI-AGENT-EGRESS rules:"
iptables -L CLI-AGENT-EGRESS -n --line-numbers
echo ""

# Make rules persistent
echo "========================================="
echo "Making Rules Persistent"
echo "========================================="
echo ""

if command -v netfilter-persistent &> /dev/null; then
    echo "Using netfilter-persistent..."
    netfilter-persistent save
    echo "Rules saved and will persist across reboots"
elif command -v iptables-save &> /dev/null && [ -d /etc/iptables ]; then
    echo "Using iptables-save..."
    iptables-save > /etc/iptables/rules.v4
    echo "Rules saved to /etc/iptables/rules.v4"
else
    echo "WARNING: Could not make rules persistent automatically"
    echo "On Debian/Ubuntu, install: sudo apt-get install iptables-persistent"
    echo "On RHEL/CentOS, install: sudo yum install iptables-services"
    echo ""
    echo "Manual save command:"
    echo "  sudo iptables-save > /etc/iptables/rules.v4"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Egress filtering is now active. User containers can only connect to:"
echo "  ✓ Whitelisted AI API endpoints (Claude, GPT, Gemini, etc.)"
echo "  ✓ DNS servers (for name resolution)"
echo "  ✓ Python/NPM package repositories"
echo ""
echo "Blocked connections:"
echo "  ✗ SSH (port 22)"
echo "  ✗ Database ports (3306, 5432, 6379, 27017)"
echo "  ✗ Remote desktop (3389, 5900)"
echo "  ✗ All other outbound connections"
echo ""
echo "To remove filtering: bash docker_files/remove-egress-filtering.sh"
echo "To view current rules: sudo iptables -L CLI-AGENT-EGRESS -n"
echo ""
