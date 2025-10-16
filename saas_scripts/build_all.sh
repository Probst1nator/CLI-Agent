#!/bin/bash
set -e

echo "======================================"
echo "CLI-Agent Multi-User Infrastructure"
echo "Build Script"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/.."

echo -e "${BLUE}Building all Docker images...${NC}"
echo ""

# Build Container Manager
echo -e "${YELLOW}[1/3] Building Container Manager...${NC}"
docker build \
    -f docker_files/Dockerfile.container-manager \
    -t cli-agent-manager:latest \
    .
echo -e "${GREEN}✓ Container Manager built${NC}"
echo ""

# Build Backend
echo -e "${YELLOW}[2/3] Building Backend Container...${NC}"
docker build \
    -f docker_files/Dockerfile.backend \
    -t cli-agent-backend:latest \
    .
echo -e "${GREEN}✓ Backend Container built${NC}"
echo ""

# Build Frontend
echo -e "${YELLOW}[3/3] Building Frontend Container...${NC}"
docker build \
    -f docker_files/Dockerfile.frontend \
    -t cli-agent-frontend:latest \
    .
echo -e "${GREEN}✓ Frontend Container built${NC}"
echo ""

echo -e "${GREEN}======================================"
echo "✅ All images built successfully!"
echo "======================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Review docker_files/docker-compose.infrastructure.yml"
echo "  2. Run: ./scripts/deploy.sh"
echo "  3. Access: http://localhost"
echo ""
echo "For manual deployment:"
echo "  docker-compose -f docker_files/docker-compose.infrastructure.yml up -d"
