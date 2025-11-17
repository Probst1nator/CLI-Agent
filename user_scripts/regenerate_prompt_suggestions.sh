#!/bin/bash
#
# Update Study Prompts - CLI-Agent User Script
#
# Finds all files modified in a given time frame (e.g., last month)
# in ~/OneDrive/Studium, randomly selects a few, and uses CLI-Agent
# to generate creative, contextual prompts for the day.
#
# Usage:
#   ./update_study_prompts.sh [num_files] [time_frame_days]
#
# Arguments:
#   num_files:        Number of random files to pick (default: 5)
#   time_frame_days:  How many days back to look for files (default: 30)
#

set -e  # Exit on error

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Configuration ---
STUDIUM_DIR="$HOME/OneDrive/Studium"
SUGGESTED_PROMPTS_FILE="$HOME/.cli-agent/suggested_prompts.json"
CLI_AGENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# NEW: We now call the Python module instead of running CLI-Agent with message

# --- NEW: Configuration from script arguments with defaults ---
NUM_FILES_TO_PICK="${1:-5}"     # Default to 5 random files
TIME_FRAME_DAYS="${2:-30}"    # Default to 30 days (a month)

# Logging configuration
LOG_DIR="$HOME/.cli-agent/logs/user_scripts"
LOG_FILE="$LOG_DIR/regenerate_prompt_suggestions.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log with timestamp (also outputs to console)
log_message() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message" | tee -a "$LOG_FILE"
}

# Rotate log if larger than 10MB
if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -gt 10485760 ]; then
    mv "$LOG_FILE" "$LOG_FILE.old"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log rotated (previous log saved to .old)" >> "$LOG_FILE"
fi

# --- Load .env file for API keys ---
ENV_FILE="$CLI_AGENT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    log_message "${BLUE}Loading environment from $ENV_FILE${NC}"
    # Export variables from .env file (skip comments and empty lines)
    set -a  # Automatically export all variables
    source "$ENV_FILE"
    set +a  # Turn off auto-export
else
    log_message "${YELLOW}Warning: .env file not found at $ENV_FILE${NC}"
    log_message "${YELLOW}API keys may not be available for LLM generation${NC}"
fi

# --- Prerequisite check for Python (no longer need shuf) ---
if ! command -v python3 &> /dev/null; then
    log_message "${RED}Error: python3 is required but was not found.${NC}"
    exit 1
fi

# Check if Studium directory exists
if [ ! -d "$STUDIUM_DIR" ]; then
    log_message "${RED}Error: Studium directory not found at $STUDIUM_DIR${NC}"
    exit 1
fi

log_message "${BLUE}=== CLI-Agent Study Prompt Generator ===${NC}"
log_message "${GREEN}Analyzing files from the last ${TIME_FRAME_DAYS} days in Studium...${NC}"

# --- NEW: Call Python module directly instead of running full CLI-Agent ---
# This is much faster and better integrated with the framework
cd "$CLI_AGENT_DIR"

echo ""
log_message "${BLUE}Calling Python regenerator module...${NC}"
echo -e "${YELLOW}python3 -m shared.utils.prompt_regenerator \\${NC}"
echo -e "${YELLOW}    --studium-dir \"$STUDIUM_DIR\" \\${NC}"
echo -e "${YELLOW}    --num-files $NUM_FILES_TO_PICK \\${NC}"
echo -e "${YELLOW}    --time-frame-days $TIME_FRAME_DAYS${NC}"
echo ""

python3 -m shared.utils.prompt_regenerator \
    --studium-dir "$STUDIUM_DIR" \
    --num-files "$NUM_FILES_TO_PICK" \
    --time-frame-days "$TIME_FRAME_DAYS"

# Capture exit code
REGENERATE_EXIT_CODE=$?

# Check if the prompts file was updated
if [ $REGENERATE_EXIT_CODE -eq 0 ] && [ -f "$SUGGESTED_PROMPTS_FILE" ]; then
    echo ""
    log_message "${GREEN}✓ Successfully updated study prompts!${NC}"
    log_message "${BLUE}New prompts:${NC}"

    # Pretty-print the prompts if jq is available
    if command -v jq &> /dev/null; then
        jq -r '.prompts[]' "$SUGGESTED_PROMPTS_FILE" | while IFS= read -r prompt; do
            echo -e "  ${YELLOW}•${NC} $prompt"
        done
    else
        echo -e "${YELLOW}(Install 'jq' for prettier output)${NC}"
        cat "$SUGGESTED_PROMPTS_FILE"
    fi
else
    log_message "${RED}✗ Failed to update prompts file (exit code: $REGENERATE_EXIT_CODE)${NC}"
    exit 1
fi