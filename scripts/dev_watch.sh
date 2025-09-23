#!/bin/bash

# CLI-Agent Fast Development File Watcher
# Uses bind mounts for instant file changes without rebuilding

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

PROJECT_DIR="/home/prob/repos/CLI-Agent"
COMPOSE_FILE="$PROJECT_DIR/docker_files/docker-compose.dev.yml"
CONTAINER_NAME="cli-agent-dev"
DEV_PORT="7862"

echo -e "${BLUE}=== CLI-Agent Fast Development Watcher ===${NC}"
echo -e "${CYAN}Using bind mounts for instant code changes${NC}"
echo -e "${CYAN}Project: $PROJECT_DIR${NC}"
echo -e "${CYAN}Container: $CONTAINER_NAME${NC}"
echo -e "${CYAN}Dev Port: $DEV_PORT${NC}"
echo ""

# Check if inotify-tools is installed
if ! command -v inotifywait &> /dev/null; then
    echo -e "${YELLOW}Installing inotify-tools for file watching...${NC}"
    sudo apt-get update > /dev/null 2>&1
    sudo apt-get install -y inotify-tools > /dev/null 2>&1
    echo -e "${GREEN}✓ inotify-tools installed${NC}"
fi

# Function to start development container
start_dev_container() {
    echo -e "${BLUE}🚀 Starting development container...${NC}"
    cd "$PROJECT_DIR"
    
    # Stop and remove existing dev container
    docker compose -f "$COMPOSE_FILE" down > /dev/null 2>&1 || true
    
    # Start new dev container
    if docker compose -f "$COMPOSE_FILE" up -d; then
        echo -e "${GREEN}✓ Development container started${NC}"
        echo -e "${CYAN}📍 Development UI: http://localhost:$DEV_PORT${NC}"
        
        # Wait for container to be ready
        echo -e "${CYAN}Waiting for web UI...${NC}"
        for i in {1..12}; do
            if curl -s -o /dev/null -w "%{http_code}" http://localhost:$DEV_PORT/ | grep -q "200"; then
                echo -e "${GREEN}✓ Development UI is ready${NC}"
                return 0
            else
                echo -e "${YELLOW}⏳ Waiting... ($i/12)${NC}"
                sleep 5
            fi
        done
        echo -e "${RED}⚠ Web UI not responding, but container is running${NC}"
    else
        echo -e "${RED}✗ Failed to start development container${NC}"
        return 1
    fi
}

# Function to restart container (fast reload)
restart_container() {
    local changed_file="$1"
    echo ""
    echo -e "${MAGENTA}📝 File changed: ${changed_file}${NC}"
    echo -e "${BLUE}🔄 Restarting container (bind mount reload)...${NC}"
    
    cd "$PROJECT_DIR"
    if docker compose -f "$COMPOSE_FILE" restart > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Container restarted${NC}"
        
        # Quick health check
        sleep 5
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$DEV_PORT/ | grep -q "200"; then
            echo -e "${GREEN}✓ Development UI is ready${NC}"
        else
            echo -e "${YELLOW}⏳ Web UI starting...${NC}"
        fi
    else
        echo -e "${RED}✗ Container restart failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}🚀 Fast reload complete!${NC}"
    echo -e "${CYAN}📍 Access: http://localhost:$DEV_PORT${NC}"
    echo ""
    echo -e "${BLUE}👀 Watching for changes...${NC}"
}

# Function to handle cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping development watcher...${NC}"
    echo -e "${CYAN}Development container will continue running${NC}"
    echo -e "${CYAN}To stop: docker compose -f $COMPOSE_FILE down${NC}"
    exit 0
}

# Function to stop development environment
stop_dev() {
    echo -e "${YELLOW}🛑 Stopping development environment...${NC}"
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" down > /dev/null 2>&1
    echo -e "${GREEN}✓ Development environment stopped${NC}"
    exit 0
}

# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

# Check command line arguments
case "${1:-}" in
    "stop")
        stop_dev
        ;;
    "start"|"")
        # Default action: start and watch
        ;;
    *)
        echo "Usage: $0 [start|stop]"
        echo "  start (default): Start development container and watch for changes"
        echo "  stop: Stop development environment"
        exit 1
        ;;
esac

# Start development container
start_dev_container

echo ""
echo -e "${GREEN}✓ Development environment ready${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop watching (container keeps running)${NC}"
echo -e "${CYAN}Run '$0 stop' to stop the development environment${NC}"
echo ""
echo -e "${BLUE}👀 Watching for Python file changes...${NC}"

# Watch for Python file changes
inotifywait -m -r -e modify,create,delete,move \
    --exclude='__pycache__' \
    --exclude='\.pyc$' \
    --exclude='tests/' \
    --exclude='\.git/' \
    "$PROJECT_DIR" |
while read -r directory events filename; do
    # Only process Python files
    if [[ "$filename" =~ \.py$ ]]; then
        # Debounce rapid changes (wait 1 second for more changes)
        sleep 1
        
        # Check if there are more pending events
        if read -t 0; then
            continue  # Skip this event, more are coming
        fi
        
        restart_container "$directory$filename"
    fi
done