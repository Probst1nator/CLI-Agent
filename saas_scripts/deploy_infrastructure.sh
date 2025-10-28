#!/bin/bash
set -e

# Source deployment logger for persistent audit logging
source "$(dirname "$0")/deployment_logger.sh"

# Change to project root
cd "$(dirname "$0")/.."

# Check if .env exists
if [ ! -f .env ]; then
    die ".env file not found. Please create .env file with your API keys:
  cp .env.example .env
  nano .env"
fi

log_info "Step 1: Building Docker images..."
./saas_scripts/build_all.sh || die "Failed to build Docker images"
echo ""

log_info "Step 2: Creating Docker network..."
docker network create cli-agent-network 2>/dev/null || log_info "Network already exists"
log_success "Network ready"
echo ""

log_info "Step 3: Starting infrastructure services..."
docker-compose -f docker_files/docker-compose.infrastructure.yml up -d || die "Failed to start services"
log_success "Services started"
echo ""

log_info "Step 4: Waiting for services to be ready..."
sleep 10

# Check Traefik
echo -n "Checking Traefik... "
if curl -sf http://localhost:8080/api/http/routers > /dev/null 2>&1; then
    log_success "Healthy"
else
    log_warning "Not responding (may take a few more seconds)"
fi

# Check Container Manager
echo -n "Checking Container Manager... "
if curl -sf http://localhost/api/manager/health > /dev/null 2>&1; then
    log_success "Healthy"
else
    log_warning "Not responding (may take a few more seconds)"
fi

# Check Frontend
echo -n "Checking Frontend... "
if curl -sf http://localhost/ > /dev/null 2>&1; then
    log_success "Healthy"
else
    log_warning "Not responding (may take a few more seconds)"
fi

echo ""
log_success "Deployment Complete!"
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

success
