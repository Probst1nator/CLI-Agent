#!/bin/bash
#
# Remove CLI-Agent Egress Network Filtering
#
# This script removes the iptables rules created by setup-egress-filtering.sh
#
# Usage:
#   sudo bash docker_files/remove-egress-filtering.sh

set -e

echo "========================================="
echo "CLI-Agent Egress Filtering Removal"
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
    DOCKER_IF="docker0"
fi

echo "Docker interface: $DOCKER_IF"
echo ""

# Remove jump rule from FORWARD chain
echo "[1/3] Removing jump rule from FORWARD chain..."
if iptables -C FORWARD -i $DOCKER_IF -j CLI-AGENT-EGRESS 2>/dev/null; then
    iptables -D FORWARD -i $DOCKER_IF -j CLI-AGENT-EGRESS
    echo "      Jump rule removed"
else
    echo "      Jump rule not found (already removed?)"
fi
echo ""

# Flush and delete the CLI-AGENT-EGRESS chain
echo "[2/3] Removing CLI-AGENT-EGRESS chain..."
if iptables -L CLI-AGENT-EGRESS -n >/dev/null 2>&1; then
    iptables -F CLI-AGENT-EGRESS
    iptables -X CLI-AGENT-EGRESS
    echo "      Chain removed"
else
    echo "      Chain not found (already removed?)"
fi
echo ""

# Save persistent rules
echo "[3/3] Saving changes..."
if command -v netfilter-persistent &> /dev/null; then
    netfilter-persistent save
    echo "      Rules saved"
elif command -v iptables-save &> /dev/null && [ -d /etc/iptables ]; then
    iptables-save > /etc/iptables/rules.v4
    echo "      Rules saved to /etc/iptables/rules.v4"
fi
echo ""

echo "========================================="
echo "Egress filtering removed successfully!"
echo "========================================="
echo ""
echo "User containers can now make outbound connections to any destination."
echo ""
echo "To re-enable filtering: bash docker_files/setup-egress-filtering.sh"
echo ""
