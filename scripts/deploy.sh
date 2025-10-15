#!/bin/bash
set -e

echo "======================================"
echo "CLI-Agent Multi-User Infrastructure"
echo "Deployment Script"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/.."

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo ""
    echo "Please create .env file with your API keys:"
    echo "  cp .env.example .env"
    echo "  nano .env"
    echo ""
    exit 1
fi

echo -e "${BLUE}Step 1: Building Docker images...${NC}"
./scripts/build_all.sh
echo ""

echo -e "${BLUE}Step 2: Creating Docker network...${NC}"
docker network create cli-agent-network 2>/dev/null || echo "Network already exists"
echo -e "${GREEN}✓ Network ready${NC}"
echo ""

echo -e "${BLUE}Step 3: Starting infrastructure services...${NC}"
docker-compose -f docker_files/docker-compose.infrastructure.yml up -d
echo -e "${GREEN}✓ Services started${NC}"
echo ""

echo -e "${BLUE}Step 4: Waiting for services to be ready...${NC}"
sleep 10

# Check Traefik
echo -n "Checking Traefik... "
if curl -sf http://localhost:8080/api/http/routers > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${YELLOW}⚠ Not responding (may take a few more seconds)${NC}"
fi

# Check Container Manager
echo -n "Checking Container Manager... "
if curl -sf http://localhost/api/manager/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${YELLOW}⚠ Not responding (may take a few more seconds)${NC}"
fi

# Check Frontend
echo -n "Checking Frontend... "
if curl -sf http://localhost/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${YELLOW}⚠ Not responding (may take a few more seconds)${NC}"
fi

echo ""
echo -e "${GREEN}======================================"
echo "✅ Deployment Complete!"
echo "======================================${NC}"
echo ""
echo "Access points:"
echo "  - CLI-Agent:          http://localhost"
echo "  - Traefik Dashboard:  http://localhost:8080"
echo "  - Container Manager:  http://localhost/api/manager/containers"
echo ""
echo "Monitor logs:"
echo "  - All services:       docker-compose -f docker_files/docker-compose.infrastructure.yml logs -f"
echo "  - Traefik:            docker logs -f cli-agent-traefik"
echo "  - Container Manager:  docker logs -f cli-agent-container-manager"
echo "  - Frontend:           docker logs -f cli-agent-frontend"
echo ""
echo "Stop services:"
echo "  docker-compose -f docker_files/docker-compose.infrastructure.yml down"
echo ""
