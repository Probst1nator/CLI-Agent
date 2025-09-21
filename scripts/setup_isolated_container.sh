#!/bin/bash

# CLI-Agent Isolated Container Setup Script
# Sets up a containerized CLI-Agent instance with:
# - Browser UI exposed to local network on port 7861
# - Full isolation from local network (internet access only)
# - Auto-restart on reboot via systemd

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/prob/repos/CLI-Agent"
COMPOSE_FILE="$PROJECT_DIR/docker_files/docker-compose.isolated.yml"
DOCKERFILE="$PROJECT_DIR/docker_files/Dockerfile.isolated"
SERVICE_FILE="$PROJECT_DIR/scripts/cli-agent-isolated.service"
SERVICE_NAME="cli-agent-isolated"

echo -e "${BLUE}=== CLI-Agent Isolated Container Setup ===${NC}"
echo ""

# Function to check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

# Function to check if user is in docker group
check_docker_group() {
    if ! groups "$USER" | grep -q docker; then
        echo -e "${YELLOW}Warning: User $USER is not in the docker group${NC}"
        echo "You may need to run: sudo usermod -aG docker $USER"
        echo "Then log out and back in for changes to take effect"
    fi
}

# Function to get host IP
get_host_ip() {
    # Try to get the main interface IP
    ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+' || echo "localhost"
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"
check_command docker
# Check for docker compose (modern syntax)
if ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: docker compose is not available${NC}"
    exit 1
fi
check_docker_group

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Error: Project directory $PROJECT_DIR not found${NC}"
    exit 1
fi

# Check if required files exist
for file in "$COMPOSE_FILE" "$DOCKERFILE" "$SERVICE_FILE"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: Required file $file not found${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

# Build the Docker image
echo -e "${BLUE}Building CLI-Agent isolated Docker image...${NC}"
cd "$PROJECT_DIR"
docker build -f "$DOCKERFILE" -t cli-agent:isolated .
echo -e "${GREEN}✓ Docker image built successfully${NC}"
echo ""

# Create .env file if it doesn't exist
ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Creating .env file template...${NC}"
    cat > "$ENV_FILE" << 'EOF'
# API Keys for CLI-Agent
# Add your actual API keys here

OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional: Additional configuration
# CLI_AGENT_HOST=0.0.0.0
# CLI_AGENT_GUI_MODE=true
EOF
    echo -e "${YELLOW}Please edit $ENV_FILE with your actual API keys${NC}"
fi

# Install systemd service
echo -e "${BLUE}Installing systemd service...${NC}"
sudo cp "$SERVICE_FILE" "/etc/systemd/system/$SERVICE_NAME.service"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME.service"
echo -e "${GREEN}✓ Systemd service installed and enabled${NC}"
echo ""

# Start the service
echo -e "${BLUE}Starting CLI-Agent isolated container...${NC}"
sudo systemctl start "$SERVICE_NAME.service"

# Wait a moment for startup
echo "Waiting for container to start..."
sleep 10

# Check service status
if sudo systemctl is-active --quiet "$SERVICE_NAME.service"; then
    echo -e "${GREEN}✓ CLI-Agent isolated container is running${NC}"
else
    echo -e "${RED}✗ Service failed to start${NC}"
    echo "Check logs with: sudo journalctl -u $SERVICE_NAME.service -f"
    exit 1
fi

# Get host IP for access instructions
HOST_IP=$(get_host_ip)

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${BLUE}Access Information:${NC}"
echo "  • Web UI: http://$HOST_IP:7861"
echo "  • From other devices: http://YOUR_HOST_IP:7861"
echo ""
echo -e "${BLUE}Container Details:${NC}"
echo "  • Container Name: cli-agent-isolated"
echo "  • Port Mapping: 7861:7861"
echo "  • Network: Isolated (internet only)"
echo "  • Auto-restart: Enabled"
echo ""
echo -e "${BLUE}Management Commands:${NC}"
echo "  • Check status: sudo systemctl status $SERVICE_NAME"
echo "  • View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "  • Stop service: sudo systemctl stop $SERVICE_NAME"
echo "  • Start service: sudo systemctl start $SERVICE_NAME"
echo "  • Restart service: sudo systemctl restart $SERVICE_NAME"
echo "  • Disable auto-start: sudo systemctl disable $SERVICE_NAME"
echo ""
echo -e "${BLUE}Docker Commands:${NC}"
echo "  • View container: docker ps | grep cli-agent-isolated"
echo "  • Container logs: docker logs cli-agent-isolated -f"
echo "  • Enter container: docker exec -it cli-agent-isolated /bin/bash"
echo "  • Manual start: docker compose -f $COMPOSE_FILE up -d"
echo "  • Manual stop: docker compose -f $COMPOSE_FILE down"
echo ""
echo -e "${YELLOW}Notes:${NC}"
echo "  • The container will auto-restart on system reboot"
echo "  • Data is persisted in Docker volumes"
echo "  • The container has no access to your local network"
echo "  • Internet access is available for AI API calls"
echo "  • Edit $ENV_FILE to add your API keys"
echo ""

# Check if firewall needs configuration
if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
    echo -e "${YELLOW}UFW Firewall detected. You may need to allow port 7861:${NC}"
    echo "  sudo ufw allow 7861/tcp"
    echo ""
fi

echo -e "${GREEN}Setup completed successfully!${NC}"