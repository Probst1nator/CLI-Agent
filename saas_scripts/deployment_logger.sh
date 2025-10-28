#!/bin/bash
#
# Deployment Logger - Persistent audit logging for deployment scripts
#
# Source this at the start of deployment scripts to enable logging:
#   source "$(dirname "$0")/deployment_logger.sh"
#
# Features:
# - All output logged to timestamped file
# - Color-coded console output preserved
# - Automatic cleanup of old logs (>30 days)
# - Helper functions for structured logging
#

# Create log directory
DEPLOY_LOG_DIR="${DEPLOY_LOG_DIR:-$HOME/.cli-agent/deployment_logs}"
mkdir -p "$DEPLOY_LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_NAME=$(basename "${BASH_SOURCE[1]}" .sh 2>/dev/null || echo "deployment")
DEPLOY_LOG="$DEPLOY_LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.log"

# Redirect all output to both console and log file
exec > >(tee -a "$DEPLOY_LOG")
exec 2>&1

# Color codes (preserved in console, stripped in log by tee)
export GREEN='\033[0;32m'
export BLUE='\033[0;34m'
export YELLOW='\033[1;33m'
export RED='\033[0;31m'
export NC='\033[0m'

# Log header
echo "========================================="
echo "Deployment: $SCRIPT_NAME"
echo "Started: $(date)"
echo "Log file: $DEPLOY_LOG"
echo "========================================="
echo ""

# Enhanced logging functions with timestamps
log() {
    echo "[$(date +%H:%M:%S)] $*"
}

log_info() {
    echo -e "[$(date +%H:%M:%S)] ${BLUE}INFO${NC}: $*"
}

log_success() {
    echo -e "[$(date +%H:%M:%S)] ${GREEN}SUCCESS${NC}: $*"
}

log_warning() {
    echo -e "[$(date +%H:%M:%S)] ${YELLOW}WARNING${NC}: $*"
}

log_error() {
    echo -e "[$(date +%H:%M:%S)] ${RED}ERROR${NC}: $*"
}

# Die function for fatal errors
die() {
    log_error "$*"
    echo ""
    echo "========================================="
    echo "Deployment FAILED at $(date)"
    echo "========================================="
    exit 1
}

# Success function for clean exits
success() {
    echo ""
    echo "========================================="
    echo -e "${GREEN}Deployment SUCCESSFUL at $(date)${NC}"
    echo "========================================="
    exit 0
}

# Cleanup old deployment logs (keep 30 days)
cleanup_old_logs() {
    log_info "Cleaning up old deployment logs (>30 days)..."
    find "$DEPLOY_LOG_DIR" -name "*.log" -type f -mtime +30 -delete 2>/dev/null
    local removed=$?
    if [ $removed -eq 0 ]; then
        log_info "Old logs cleaned up"
    fi
}

# Run cleanup in background to not slow down deployment
cleanup_old_logs &

# Trap to ensure footer is written even on interrupt
cleanup() {
    local exit_code=$?
    echo ""
    echo "========================================="
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Deployment COMPLETED at $(date)${NC}"
    else
        echo -e "${RED}Deployment TERMINATED with exit code $exit_code at $(date)${NC}"
    fi
    echo "Full log: $DEPLOY_LOG"
    echo "========================================="
}
trap cleanup EXIT

# Log environment info
log_info "User: $(whoami)"
log_info "Hostname: $(hostname)"
log_info "Working directory: $(pwd)"
log_info "Shell: $SHELL"

# Check if git repo and log commit info
if [ -d .git ]; then
    log_info "Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    log_info "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
fi

echo ""
