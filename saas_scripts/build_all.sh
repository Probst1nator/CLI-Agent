#!/bin/bash
set -e

# Source deployment logger for persistent audit logging
source "$(dirname "$0")/deployment_logger.sh"

# Change to project root
cd "$(dirname "$0")/.."

log_info "Building all Docker images..."
echo ""

# Build Container Manager
log_info "[1/3] Building Container Manager..."
docker build \
    -f docker_files/Dockerfile.container-manager \
    -t cli-agent-manager:latest \
    . || die "Failed to build Container Manager"
log_success "Container Manager built"
echo ""

# Build Backend
log_info "[2/3] Building Backend Container..."
docker build \
    -f docker_files/Dockerfile.backend \
    -t cli-agent-backend:latest \
    . || die "Failed to build Backend Container"
log_success "Backend Container built"
echo ""

# Build Frontend
log_info "[3/3] Building Frontend Container..."
docker build \
    -f docker_files/Dockerfile.frontend \
    -t cli-agent-frontend:latest \
    . || die "Failed to build Frontend Container"
log_success "Frontend Container built"
echo ""

log_success "All images built successfully!"
echo ""
echo "Next steps:"
echo "  1. Review docker_files/docker-compose.infrastructure.yml"
echo "  2. Run: ./saas_scripts/deploy_infrastructure.sh"
echo "  3. Access: http://localhost"
echo ""
echo "For manual deployment:"
echo "  docker-compose -f docker_files/docker-compose.infrastructure.yml up -d"

success
