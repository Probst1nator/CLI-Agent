#!/bin/bash

# Test script for CLI-Agent production Docker container
# Usage: ./test-docker-prod.sh

set -e

echo "=== CLI-Agent Production Docker Test ==="
echo "Timestamp: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if container is running
check_container() {
    local container_name=$1
    if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        return 0
    else
        return 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local port=$1
    local timeout=${2:-60}
    local count=0

    print_status $YELLOW "Waiting for service on port $port..."

    while [ $count -lt $timeout ]; do
        if curl -s -f "http://localhost:$port/" > /dev/null 2>&1; then
            print_status $GREEN "✅ Service on port $port is ready!"
            return 0
        fi
        sleep 2
        count=$((count + 2))
        if [ $((count % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done

    print_status $RED "❌ Service on port $port not ready after ${timeout}s"
    return 1
}

# Test 1: Check if Docker is available
print_status $BLUE "🔍 Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    print_status $RED "❌ Docker is not installed or not in PATH"
    exit 1
fi
print_status $GREEN "✅ Docker is available"

# Test 2: Check if image exists
print_status $BLUE "🔍 Checking if cli-agent:prod image exists..."
if ! docker image inspect cli-agent:prod &> /dev/null; then
    print_status $RED "❌ cli-agent:prod image not found. Please build it first:"
    echo "   docker build -f Dockerfile.prod -t cli-agent:prod ."
    exit 1
fi
print_status $GREEN "✅ cli-agent:prod image found"

# Test 3: Stop any existing containers
print_status $BLUE "🧹 Cleaning up existing containers..."
if check_container "cli-agent-test"; then
    docker stop cli-agent-test > /dev/null 2>&1 || true
    docker rm cli-agent-test > /dev/null 2>&1 || true
    print_status $YELLOW "Removed existing test container"
fi

# Test 4: Start container
print_status $BLUE "🚀 Starting CLI-Agent container..."
docker run -d \
    --name cli-agent-test \
    -p 7860:7860 \
    -p 7861:7861 \
    -p 7862:7862 \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-dummy}" \
    cli-agent:prod

if check_container "cli-agent-test"; then
    print_status $GREEN "✅ Container started successfully"
else
    print_status $RED "❌ Failed to start container"
    exit 1
fi

# Test 5: Wait for service to be ready
print_status $BLUE "⏳ Waiting for visual response interface..."
if wait_for_service 7860 90; then
    SERVICE_PORT=7860
elif wait_for_service 7861 30; then
    SERVICE_PORT=7861
elif wait_for_service 7862 30; then
    SERVICE_PORT=7862
else
    print_status $RED "❌ Visual response interface not accessible on any port"
    print_status $YELLOW "Container logs:"
    docker logs --tail 20 cli-agent-test
    docker stop cli-agent-test > /dev/null 2>&1 || true
    docker rm cli-agent-test > /dev/null 2>&1 || true
    exit 1
fi

# Test 6: Test HTTP endpoints
print_status $BLUE "🌐 Testing HTTP endpoints..."

# Test main page
if curl -s -f "http://localhost:$SERVICE_PORT/" > /dev/null; then
    print_status $GREEN "✅ Main page accessible"
else
    print_status $RED "❌ Main page not accessible"
fi

# Test WebSocket endpoint (just check if it returns WebSocket upgrade error, which is expected)
WS_RESPONSE=$(curl -s -w "%{http_code}" "http://localhost:$SERVICE_PORT/ws" -o /dev/null)
if [ "$WS_RESPONSE" = "400" ] || [ "$WS_RESPONSE" = "426" ]; then
    print_status $GREEN "✅ WebSocket endpoint responding"
else
    print_status $YELLOW "⚠️  WebSocket endpoint returned: $WS_RESPONSE"
fi

# Test 7: Check container health
print_status $BLUE "💊 Checking container health..."
sleep 10  # Wait for health check to run

HEALTH_STATUS=$(docker inspect cli-agent-test --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
case $HEALTH_STATUS in
    "healthy")
        print_status $GREEN "✅ Container is healthy"
        ;;
    "starting")
        print_status $YELLOW "⏳ Container health check still starting..."
        ;;
    "unhealthy")
        print_status $RED "❌ Container is unhealthy"
        ;;
    *)
        print_status $YELLOW "⚠️  Health status: $HEALTH_STATUS"
        ;;
esac

# Test 8: Show container info
print_status $BLUE "📊 Container Information:"
echo "Container ID: $(docker ps --filter name=cli-agent-test --format '{{.ID}}')"
echo "Status: $(docker ps --filter name=cli-agent-test --format '{{.Status}}')"
echo "Ports: $(docker ps --filter name=cli-agent-test --format '{{.Ports}}')"
echo "Access URL: http://localhost:$SERVICE_PORT"

# Test 9: Show recent logs
print_status $BLUE "📝 Recent container logs:"
docker logs --tail 10 cli-agent-test

echo ""
print_status $GREEN "🎉 Production container test completed successfully!"
print_status $BLUE "💡 Access the visual interface at: http://localhost:$SERVICE_PORT"
print_status $YELLOW "📌 To stop the test container:"
echo "   docker stop cli-agent-test && docker rm cli-agent-test"

echo ""
print_status $BLUE "✨ Ready for production deployment with docker-compose!"