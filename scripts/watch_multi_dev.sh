#!/bin/bash

# CLI-Agent Multi-Instance Development File Watcher
# Automatically restarts dev containers on Python and frontend file changes

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_DIR="/home/prob/repos/CLI-Agent"
COMPOSE_FILE="$PROJECT_DIR/docker_files/docker-compose.multi-dev.yml"

echo -e "${BLUE}=== CLI-Agent Multi-Dev File Watcher ===${NC}"
echo -e "${CYAN}Watching: $PROJECT_DIR${NC}"
echo -e "${CYAN}Ports: 7860 (session manager), 7861-7863 (instances)${NC}"
echo ""

# Check inotify-tools
if ! command -v inotifywait &> /dev/null; then
    echo -e "${YELLOW}Installing inotify-tools...${NC}"
    sudo apt-get update > /dev/null 2>&1
    sudo apt-get install -y inotify-tools > /dev/null 2>&1
    echo -e "${GREEN}✓ inotify-tools installed${NC}"
fi

# Restart function
restart_containers() {
    local changed_file="$1"
    echo ""
    echo -e "${YELLOW}📝 Changed: ${changed_file}${NC}"
    echo -e "${BLUE}🔄 Restarting dev containers...${NC}"
    
    cd "$PROJECT_DIR"
    if docker compose -f "$COMPOSE_FILE" restart > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Containers restarted (5-10s)${NC}"
        
        # Quick health check
        sleep 5
        if curl -s http://localhost:7860/health | grep -q "healthy"; then
            echo -e "${GREEN}✓ Session manager ready${NC}"
        fi
    else
        echo -e "${RED}✗ Restart failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}🚀 Auto-reload complete${NC}"
    echo -e "${BLUE}👀 Watching for changes...${NC}"
}

# Cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping file watcher${NC}"
    echo -e "${CYAN}Containers still running on port 7860${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "${GREEN}✓ File watcher started${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop watching${NC}"
echo ""
echo -e "${BLUE}👀 Watching for Python and frontend changes...${NC}"

# Watch Python and frontend files
inotifywait -m -r -e modify,create,delete,move \
    --exclude='__pycache__' \
    --exclude='\.pyc$' \
    --exclude='tests/' \
    --exclude='\.git/' \
    --exclude='node_modules/' \
    "$PROJECT_DIR" |
while read -r directory events filename; do
    # Match Python, JavaScript, HTML, and CSS files
    if [[ "$filename" =~ \.(py|js|html|css)$ ]]; then
        # Debounce (1 second)
        sleep 1
        
        # Skip if more events pending
        if read -t 0; then
            continue
        fi
        
        restart_containers "$directory$filename"
    fi
done
