#!/bin/bash
# Test an agent container by sending WebSocket messages

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
HOST="localhost"
PORT="8000"

# Parse arguments
if [ $# -ge 1 ]; then
    PORT=$1
fi

if [ $# -ge 2 ]; then
    HOST=$2
fi

BASE_URL="http://${HOST}:${PORT}"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Testing Agent Container                        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Target: ${BASE_URL}${NC}"
echo ""

# Test 1: Health check
echo -e "${BLUE}Test 1: Health Check${NC}"
echo -e "${YELLOW}→ GET /health${NC}"

HEALTH_RESPONSE=$(curl -s "${BASE_URL}/health")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Health check successful${NC}"
    echo "  Response: ${HEALTH_RESPONSE}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    exit 1
fi

echo ""

# Test 2: Agent info
echo -e "${BLUE}Test 2: Agent Info${NC}"
echo -e "${YELLOW}→ GET /api/info${NC}"

INFO_RESPONSE=$(curl -s "${BASE_URL}/api/info")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Agent info retrieved${NC}"
    echo "  Response: ${INFO_RESPONSE}"
else
    echo -e "${RED}✗ Failed to get agent info${NC}"
    exit 1
fi

echo ""

# Test 3: Send message (HTTP)
echo -e "${BLUE}Test 3: Send Message (HTTP)${NC}"
echo -e "${YELLOW}→ POST /api/chat/send${NC}"

MESSAGE='{"content": "Hello! What tools do you have?"}'

CHAT_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "${MESSAGE}" \
    "${BASE_URL}/api/chat/send")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Message sent successfully${NC}"
    echo "  Response: ${CHAT_RESPONSE}" | head -c 200
    echo "..."
else
    echo -e "${RED}✗ Failed to send message${NC}"
    exit 1
fi

echo ""

# Test 4: Get chat history
echo -e "${BLUE}Test 4: Get Chat History${NC}"
echo -e "${YELLOW}→ GET /api/chat/history${NC}"

HISTORY_RESPONSE=$(curl -s "${BASE_URL}/api/chat/history")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Chat history retrieved${NC}"
    MESSAGE_COUNT=$(echo "${HISTORY_RESPONSE}" | grep -o '"role"' | wc -l)
    echo "  Messages in history: ${MESSAGE_COUNT}"
else
    echo -e "${RED}✗ Failed to get chat history${NC}"
    exit 1
fi

echo ""

# Test 5: Clear chat history
echo -e "${BLUE}Test 5: Clear Chat History${NC}"
echo -e "${YELLOW}→ POST /api/chat/clear${NC}"

CLEAR_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/chat/clear")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Chat history cleared${NC}"
    echo "  Response: ${CLEAR_RESPONSE}"
else
    echo -e "${RED}✗ Failed to clear chat history${NC}"
    exit 1
fi

echo ""

# Summary
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              All Tests Passed! ✓                          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}WebSocket endpoint:${NC}"
echo -e "  ws://${HOST}:${PORT}/ws"
echo ""

echo -e "${YELLOW}Test WebSocket manually:${NC}"
echo -e "  ${BLUE}# Install wscat: npm install -g wscat${NC}"
echo -e "  ${BLUE}wscat -c ws://${HOST}:${PORT}/ws${NC}"
echo ""
echo -e "${YELLOW}Send test message:${NC}"
echo -e '  {"type": "chat_message", "content": "Hello agent!"}'
echo ""
