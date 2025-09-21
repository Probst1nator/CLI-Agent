#!/bin/bash

# CLI-Agent Isolated Container Verification Script
# Verifies that the isolated container setup is working correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CLI-Agent Isolated Container Verification ===${NC}"
echo ""

# Function to check status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1${NC}"
        return 1
    fi
}

# Check if systemd service is running
echo -e "${BLUE}Checking systemd service...${NC}"
sudo systemctl is-active --quiet cli-agent-isolated
check_status "Systemd service is active"

# Check if container is running
echo -e "${BLUE}Checking container status...${NC}"
docker ps | grep -q cli-agent-isolated
check_status "Container is running"

# Check if web UI is accessible locally
echo -e "${BLUE}Testing local web UI access...${NC}"
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7861/)
if [ "$response" = "200" ]; then
    echo -e "${GREEN}✓ Web UI accessible locally (HTTP $response)${NC}"
else
    echo -e "${RED}✗ Web UI not accessible locally (HTTP $response)${NC}"
fi

# Check if aiohttp is installed
echo -e "${BLUE}Checking web server dependencies...${NC}"
docker exec cli-agent-isolated python3 -c "import aiohttp; print('aiohttp OK')" > /dev/null 2>&1
check_status "aiohttp dependency is installed"

# Check port mapping
echo -e "${BLUE}Checking port mapping...${NC}"
docker port cli-agent-isolated | grep -q "7860/tcp -> 0.0.0.0:7861"
check_status "Port mapping is correct (7860→7861)"

# Check firewall
echo -e "${BLUE}Checking firewall configuration...${NC}"
if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
    ufw status | grep -q "7861/tcp.*ALLOW"
    check_status "UFW allows port 7861"
else
    echo -e "${YELLOW}ℹ UFW not active or not installed${NC}"
fi

# Check container processes
echo -e "${BLUE}Checking container processes...${NC}"
process_count=$(docker exec cli-agent-isolated ps aux | grep "python3 main.py" | wc -l)
if [ "$process_count" -eq 2 ]; then
    echo -e "${GREEN}✓ Correct number of processes (main + web server)${NC}"
else
    echo -e "${YELLOW}⚠ Unexpected process count: $process_count${NC}"
fi

# Check data persistence
echo -e "${BLUE}Checking data persistence...${NC}"
docker volume ls | grep -q cli-agent-isolated-data
check_status "Data volume exists"

# Check internet connectivity
echo -e "${BLUE}Testing internet connectivity...${NC}"
docker exec cli-agent-isolated curl -s --connect-timeout 5 https://api.openai.com/v1/models > /dev/null 2>&1
check_status "Internet connectivity for AI APIs"

# Get host IP for final report
HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+' || echo "localhost")

echo ""
echo -e "${GREEN}=== Verification Complete ===${NC}"
echo ""
echo -e "${BLUE}Access Information:${NC}"
echo "  • Local: http://localhost:7861"
echo "  • Network: http://$HOST_IP:7861"
echo ""
echo -e "${BLUE}Management Commands:${NC}"
echo "  • Status: sudo systemctl status cli-agent-isolated"
echo "  • Logs: docker logs cli-agent-isolated -f"
echo "  • Restart: sudo systemctl restart cli-agent-isolated"
echo ""