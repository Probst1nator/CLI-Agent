#!/bin/bash
#
# Unified Log Viewer for CLI-Agent Debugging
#
# Shows logs from all sources in one view:
# - Backend session logs
# - Frontend client errors
# - Tool-specific logs
# - Recent audit sessions
# - Docker container logs
#
# Usage:
#   ./view_logs.sh           # Last 100 lines
#   ./view_logs.sh 500       # Last 500 lines
#   ./view_logs.sh | less    # Scrollable view

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

LOGS_DIR="$HOME/.cli-agent/logs"
CLIENT_LOGS="$LOGS_DIR/client_errors.log"
SESSION_LOG="$LOGS_DIR/session.log"
AUDIT_DIR="$HOME/.cli-agent/audit_logs/sessions"

# Get last N lines (default 100)
LINES=${1:-100}

echo -e "${BLUE}=== CLI-Agent Unified Log View (last $LINES lines) ===${NC}\n"

# 1. Backend Session Logs
if [ -f "$SESSION_LOG" ]; then
    echo -e "${GREEN}[Backend Session Logs]${NC}"
    tail -n $LINES "$SESSION_LOG" | grep -E "ERROR|WARNING|req_" --color=always || tail -n 10 "$SESSION_LOG"
    echo ""
fi

# 2. Client Errors (Frontend)
if [ -f "$CLIENT_LOGS" ]; then
    echo -e "${GREEN}[Frontend Errors]${NC}"
    if [ -s "$CLIENT_LOGS" ]; then
        tail -n $LINES "$CLIENT_LOGS" || echo "  (no client errors)"
    else
        echo "  (no client errors)"
    fi
    echo ""
fi

# 3. Tool Logs with errors
echo -e "${GREEN}[Tool Errors]${NC}"
found_errors=false
for tool_log in "$LOGS_DIR"/tools_*.log; do
    if [ -f "$tool_log" ]; then
        tool_name=$(basename "$tool_log" .log | sed 's/tools_//')
        errors=$(tail -n $LINES "$tool_log" 2>/dev/null | grep -c -E "ERROR|WARNING" || echo "0")
        if [ "$errors" -gt 0 ]; then
            echo -e "  ${YELLOW}$tool_name${NC}: $errors errors/warnings"
            tail -n $LINES "$tool_log" | grep -E "ERROR|WARNING" | tail -5
            found_errors=true
        fi
    fi
done
if [ "$found_errors" = false ]; then
    echo "  (no tool errors)"
fi
echo ""

# 4. Recent Audit Sessions
echo -e "${GREEN}[Recent Sessions]${NC}"
if [ -d "$AUDIT_DIR" ]; then
    session_count=$(ls -1 "$AUDIT_DIR"/session_*.json 2>/dev/null | wc -l)
    if [ "$session_count" -gt 0 ]; then
        ls -lt "$AUDIT_DIR"/session_*.json | head -5 | while read line; do
            file=$(echo $line | awk '{print $9}')
            if [ -n "$file" ]; then
                session_id=$(basename "$file" .json | sed 's/session_//')
                # Extract start time from JSON if jq is available
                if command -v jq &> /dev/null; then
                    start_time=$(jq -r '.start_time' "$file" 2>/dev/null || echo "unknown")
                    echo "  $session_id (started: $start_time)"
                else
                    echo "  $session_id"
                fi
            fi
        done
    else
        echo "  (no audit logs)"
    fi
else
    echo "  (audit directory not found)"
fi
echo ""

# 5. Docker Logs (if running in container)
if command -v docker &> /dev/null; then
    echo -e "${GREEN}[Container Logs]${NC}"
    containers=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -E "cli-agent|traefik|container-manager" || true)
    if [ -n "$containers" ]; then
        echo "$containers" | while read container; do
            echo -e "  ${YELLOW}$container${NC}:"
            error_count=$(docker logs --tail $LINES "$container" 2>&1 | grep -c -E "ERROR|error|WARNING|warn" || echo "0")
            if [ "$error_count" -gt 0 ]; then
                docker logs --tail 5 "$container" 2>&1 | grep -E "ERROR|error|WARNING|warn" || echo "    (no errors)"
            else
                echo "    (no errors in last $LINES lines)"
            fi
        done
    else
        echo "  (no CLI-Agent containers running)"
    fi
    echo ""
fi

# 6. Summary Statistics
echo -e "${BLUE}=== Summary ===${NC}"
if [ -f "$SESSION_LOG" ]; then
    total_errors=$(tail -n $LINES "$SESSION_LOG" | grep -c "ERROR" || echo "0")
    total_warnings=$(tail -n $LINES "$SESSION_LOG" | grep -c "WARNING" || echo "0")
    echo "  Backend: $total_errors errors, $total_warnings warnings"
fi

if [ -f "$CLIENT_LOGS" ]; then
    client_errors=$(wc -l < "$CLIENT_LOGS" 2>/dev/null || echo "0")
    echo "  Frontend: $client_errors total error log entries"
fi

echo ""
echo -e "${BLUE}Tip: Use './view_logs.sh | less' for scrollable view${NC}"
echo -e "${BLUE}Tip: Use './view_logs.sh 1000' to see more lines${NC}"
