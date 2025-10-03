#!/bin/bash

# CLI-Agent Development File Watcher
# Watches for Python file changes and automatically reloads the isolated container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_DIR="~/repos/CLI-Agent"
SERVICE_NAME="cli-agent-isolated"

echo -e "${BLUE}=== CLI-Agent Development File Watcher ===${NC}"
echo -e "${CYAN}Watching for Python file changes in: $PROJECT_DIR${NC}"
echo -e "${CYAN}Service: $SERVICE_NAME${NC}"
echo ""

# Check if inotify-tools is installed
if ! command -v inotifywait &> /dev/null; then
    echo -e "${YELLOW}Installing inotify-tools for file watching...${NC}"
    sudo apt-get update > /dev/null 2>&1
    sudo apt-get install -y inotify-tools > /dev/null 2>&1
    echo -e "${GREEN}✓ inotify-tools installed${NC}"
fi

# Function to reload container
reload_container() {
    local changed_file="$1"
    echo ""
    echo -e "${YELLOW}📝 File changed: ${changed_file}${NC}"
    echo -e "${BLUE}🔄 Reloading container...${NC}"
    
    # Rebuild the Docker image
    echo -e "${CYAN}Building new image...${NC}"
    cd "$PROJECT_DIR"
    if docker build -f docker_files/Dockerfile.isolated -t cli-agent:isolated . > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Image rebuilt successfully${NC}"
    else
        echo -e "${RED}✗ Image build failed${NC}"
        return 1
    fi
    
    # Restart the service
    echo -e "${CYAN}Restarting service...${NC}"
    if sudo systemctl restart "$SERVICE_NAME" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Service restarted successfully via systemd${NC}"
    else
        echo -e "${YELLOW}⚠️ Systemd restart failed. Falling back to direct Docker Compose restart...${NC}"
        cd "$PROJECT_DIR"
        # Mimic the service file's down/up logic to ensure a clean start
        if docker compose -f docker_files/docker-compose.multi-simple.yml down --remove-orphans > /dev/null 2>&1 && \
           docker compose -f docker_files/docker-compose.multi-simple.yml up -d > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Service restarted successfully via Docker Compose fallback${NC}"
        else
            echo -e "${RED}✗ Fallback restart via Docker Compose also failed${NC}"
            return 1
        fi
    fi
    
    # Wait for container to be ready
    echo -e "${CYAN}Waiting for container to be ready...${NC}"
    sleep 10
    
    # Test if web UI is accessible
    for i in {1..6}; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:7861/ | grep -q "200"; then
            echo -e "${GREEN}✓ Web UI is ready${NC}"
            break
        else
            echo -e "${YELLOW}⏳ Waiting for web UI... ($i/6)${NC}"
            sleep 5
        fi
    done
    
    echo -e "${GREEN}🚀 Container reloaded successfully!${NC}"
    echo -e "${CYAN}📍 Access: http://localhost:7861${NC}"
    echo ""
    echo -e "${BLUE}👀 Watching for changes...${NC}"
}

# Function to handle cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping file watcher...${NC}"
    exit 0
}

# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

echo -e "${GREEN}✓ File watcher started${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop watching${NC}"
echo ""
echo -e "${BLUE}👀 Watching for changes...${NC}"

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
        # Debounce rapid changes (wait 2 seconds for more changes)
        sleep 2
        
        # Check if there are more pending events
        if read -t 0; then
            continue  # Skip this event, more are coming
        fi
        
        reload_container "$directory$filename"
    fi
done