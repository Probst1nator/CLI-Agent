#!/bin/bash
set -e

echo "======================================"
echo "CLI-Agent Multi-User Infrastructure"
echo "Test Script"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    PASSED=$((PASSED + 1))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    FAILED=$((FAILED + 1))
}

echo -e "${BLUE}Test 1: Infrastructure Health Checks${NC}"
echo "-------------------------------------"

# Test Traefik
echo -n "Traefik API... "
if curl -sf http://localhost:8080/api/http/routers > /dev/null 2>&1; then
    test_pass "Traefik responding"
else
    test_fail "Traefik not responding"
fi

# Test Container Manager
echo -n "Container Manager health... "
HEALTH=$(curl -sf http://localhost/api/manager/health)
if [ $? -eq 0 ]; then
    test_pass "Container Manager healthy"
    echo "  Response: $HEALTH"
else
    test_fail "Container Manager not responding"
fi

# Test Frontend
echo -n "Frontend... "
if curl -sf http://localhost/ > /dev/null 2>&1; then
    test_pass "Frontend serving"
else
    test_fail "Frontend not responding"
fi

echo ""
echo -e "${BLUE}Test 2: UUID-Based Container Discovery${NC}"
echo "-------------------------------------"

# Generate test UUIDs
UUID1=$(uuidgen | tr '[:upper:]' '[:lower:]')
UUID2=$(uuidgen | tr '[:upper:]' '[:lower:]')

echo "Test UUID 1: ${UUID1:0:8}..."
echo "Test UUID 2: ${UUID2:0:8}..."
echo ""

# Test discovery for UUID 1
echo -n "Discovering container for UUID 1... "
DISCOVERY1=$(curl -sf http://localhost/api/manager/discover -H "X-User-UUID: $UUID1")
if [ $? -eq 0 ]; then
    CONTAINER1=$(echo $DISCOVERY1 | grep -o '"container_name":"[^"]*"' | cut -d'"' -f4)
    if [ -n "$CONTAINER1" ]; then
        test_pass "Container created: $CONTAINER1"
    else
        test_fail "Container name not found in response"
    fi
else
    test_fail "Discovery failed"
fi

# Test discovery for UUID 2
echo -n "Discovering container for UUID 2... "
DISCOVERY2=$(curl -sf http://localhost/api/manager/discover -H "X-User-UUID: $UUID2")
if [ $? -eq 0 ]; then
    CONTAINER2=$(echo $DISCOVERY2 | grep -o '"container_name":"[^"]*"' | cut -d'"' -f4)
    if [ -n "$CONTAINER2" ]; then
        test_pass "Container created: $CONTAINER2"
    else
        test_fail "Container name not found in response"
    fi
else
    test_fail "Discovery failed"
fi

# Verify containers are different
if [ "$CONTAINER1" != "$CONTAINER2" ]; then
    test_pass "Containers are isolated (different names)"
else
    test_fail "Containers have same name (isolation broken)"
fi

echo ""
echo -e "${BLUE}Test 3: Container List${NC}"
echo "-------------------------------------"

CONTAINERS=$(curl -sf http://localhost/api/manager/containers)
if [ $? -eq 0 ]; then
    COUNT=$(echo $CONTAINERS | grep -o '"count":[0-9]*' | cut -d':' -f2)
    test_pass "Container list retrieved (count: $COUNT)"

    if [ "$COUNT" -ge 2 ]; then
        test_pass "At least 2 containers running"
    else
        test_fail "Expected at least 2 containers, found $COUNT"
    fi
else
    test_fail "Failed to retrieve container list"
fi

echo ""
echo -e "${BLUE}Test 4: Docker Resources${NC}"
echo "-------------------------------------"

# Check if containers are actually running in Docker
RUNNING=$(docker ps --filter "name=cli-agent-" --format "{{.Names}}" | wc -l)
echo "Docker containers with cli-agent- prefix: $RUNNING"

if [ "$RUNNING" -ge 5 ]; then
    test_pass "Expected infrastructure containers running (Traefik, Manager, Frontend + backends)"
else
    test_fail "Expected at least 5 containers, found $RUNNING"
fi

# Check volumes with labels
VOLUMES=$(docker volume ls --filter "label=cli-agent.managed=true" --format "{{.Name}}" | wc -l)
echo "Labeled volumes: $VOLUMES"

if [ "$VOLUMES" -ge 6 ]; then
    test_pass "Expected volumes created (3 per backend container)"
else
    test_fail "Expected at least 6 volumes, found $VOLUMES"
fi

echo ""
echo -e "${BLUE}Test 5: Cleanup Test Containers${NC}"
echo "-------------------------------------"

# Clean up test containers
for UUID in $UUID1 $UUID2; do
    echo -n "Deleting test container for UUID ${UUID:0:8}... "
    RESULT=$(curl -sf -X DELETE http://localhost/api/manager/container/$UUID)
    if [ $? -eq 0 ]; then
        test_pass "Container deleted"
    else
        test_fail "Delete failed"
    fi
done

echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
