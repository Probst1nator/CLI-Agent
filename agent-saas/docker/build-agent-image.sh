#!/bin/bash
# Build a single agent Docker image

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 <agent-name>${NC}"
    echo ""
    echo -e "${YELLOW}Available agents:${NC}"
    echo "  - minimal_agent"
    echo "  - developer_agent"
    echo ""
    echo -e "${YELLOW}Example:${NC}"
    echo "  $0 minimal_agent"
    exit 1
fi

AGENT_NAME=$1

# Change to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

echo -e "${BLUE}Building agent: ${AGENT_NAME}${NC}"
echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"
echo ""

# Verify agent exists
if [ ! -d "agents/${AGENT_NAME}" ]; then
    echo -e "${RED}✗ Error: agents/${AGENT_NAME} directory not found${NC}"
    echo ""
    echo -e "${YELLOW}Available agents:${NC}"
    ls -1 agents/ | grep -v "CLAUDE.md"
    exit 1
fi

# Ensure base image exists
if ! docker image inspect cli-agent-base:latest >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Base image not found. Building base image first...${NC}"
    echo ""

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
fi

# Build agent image
echo -e "${BLUE}Building ${AGENT_NAME} image...${NC}"

docker build \
    -f agent-saas/docker/Dockerfile.${AGENT_NAME} \
    -t cli-agent-${AGENT_NAME}:latest \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ ${AGENT_NAME} image built successfully${NC}"
    echo ""

    # Show image info
    echo -e "${YELLOW}Image details:${NC}"
    docker images cli-agent-${AGENT_NAME}:latest

    echo ""
    echo -e "${YELLOW}Test the image:${NC}"
    echo -e "  ${BLUE}docker run --rm -p 8000:8000 cli-agent-${AGENT_NAME}:latest${NC}"
    echo ""
    echo -e "${YELLOW}Verify health:${NC}"
    echo -e "  ${BLUE}curl http://localhost:8000/health${NC}"
    echo ""
else
    echo -e "${RED}✗ Failed to build ${AGENT_NAME} image${NC}"
    exit 1
fi
