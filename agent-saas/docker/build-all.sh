#!/bin/bash
# Build all agent Docker images for CLI-Agent SaaS

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Building Agent SaaS Docker Images                 ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Change to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"
echo ""

# Step 1: Build base image
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1/3: Building base image (cli-agent-base:latest)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

docker build \
    -f agent-saas/docker/Dockerfile.agent-base \
    -t cli-agent-base:latest \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Base image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build base image${NC}"
    exit 1
fi

echo ""

# Step 2: Build minimal_agent image
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2/3: Building minimal_agent image${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Verify agent exists
if [ ! -d "agents/minimal_agent" ]; then
    echo -e "${RED}✗ Error: agents/minimal_agent directory not found${NC}"
    exit 1
fi

docker build \
    -f agent-saas/docker/Dockerfile.minimal-agent \
    -t cli-agent-minimal_agent:latest \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ minimal_agent image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build minimal_agent image${NC}"
    exit 1
fi

echo ""

# Step 3: Build developer_agent image
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3/3: Building developer_agent image${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Verify agent exists
if [ ! -d "agents/developer_agent" ]; then
    echo -e "${RED}✗ Error: agents/developer_agent directory not found${NC}"
    exit 1
fi

docker build \
    -f agent-saas/docker/Dockerfile.developer-agent \
    -t cli-agent-developer_agent:latest \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ developer_agent image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build developer_agent image${NC}"
    exit 1
fi

echo ""

# Summary
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 Build Complete!                           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Images built:${NC}"
docker images | grep -E "(cli-agent-base|cli-agent-minimal|cli-agent-developer)" || true

echo ""
echo -e "${YELLOW}Test the images:${NC}"
echo -e "  ${BLUE}docker run --rm -p 8000:8000 cli-agent-minimal_agent:latest${NC}"
echo -e "  ${BLUE}docker run --rm -p 8001:8000 cli-agent-developer_agent:latest${NC}"
echo ""

echo -e "${YELLOW}Verify health:${NC}"
echo -e "  ${BLUE}curl http://localhost:8000/health${NC}"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Test agent containers manually"
echo -e "  2. Start agent-container-manager"
echo -e "  3. Create a test session via API"
echo ""
