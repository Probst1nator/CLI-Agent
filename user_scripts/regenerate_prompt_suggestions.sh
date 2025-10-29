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

# --- NEW: Prerequisite check for 'shuf' command ---
if ! command -v shuf &> /dev/null; then
    log_message "${RED}Error: The 'shuf' command is required for random selection but was not found.${NC}"
    log_message "${YELLOW}Please install GNU coreutils ('sudo apt-get install coreutils' on Debian/Ubuntu, or 'brew install coreutils' on macOS).${NC}"
    exit 1
fi

# Check if Studium directory exists
if [ ! -d "$STUDIUM_DIR" ]; then
    log_message "${RED}Error: Studium directory not found at $STUDIUM_DIR${NC}"
    exit 1
fi

log_message "${BLUE}=== CLI-Agent Study Prompt Generator ===${NC}"
log_message "${GREEN}Searching for files modified in the last ${TIME_FRAME_DAYS} days in Studium...${NC}"

# --- MODIFIED: Find all files in the last month/week, then randomly pick 5 ---
# 1. `find`: Locates all files modified within the last `TIME_FRAME_DAYS`.
#    Excludes hidden files and common binary extensions.
# 2. `shuf -n`: Randomly shuffles the input lines and picks `NUM_FILES_TO_PICK`.
SELECTED_FILES=$(find "$STUDIUM_DIR" -type f -mtime "-${TIME_FRAME_DAYS}" \
    ! -path "*/.*" \
    ! -name "*.exe" \
    ! -name "*.dll" \
    ! -name "*.so" \
    ! -name "*.dylib" \
    ! -name "*.zip" \
    ! -name "*.tar.gz" \
    ! -name "*.rar" \
    2>/dev/null \
    | shuf -n "$NUM_FILES_TO_PICK")

# Check if we found any files
if [ -z "$SELECTED_FILES" ]; then
    log_message "${RED}Error: No files found in $STUDIUM_DIR modified in the last ${TIME_FRAME_DAYS} days.${NC}"
    exit 1
fi

# --- MODIFIED: Updated log messages to reflect the new logic ---
file_count=$(echo "$SELECTED_FILES" | wc -l)
log_message "${GREEN}Randomly selected ${file_count} files:${NC}"
echo "$SELECTED_FILES" | while read -r file; do
    filename=$(basename "$file")
    mod_time=$(stat -c '%y' "$file" 2>/dev/null | cut -d' ' -f1)
    echo -e "  ${YELLOW}•${NC} $filename (modified: $mod_time)"
done

echo ""
log_message "${BLUE}Generating creative prompts with CLI-Agent...${NC}"
echo ""

# Build the file list for the prompt
FILE_LIST=""
while IFS= read -r file; do
    filename=$(basename "$file")
    mod_time=$(stat -c '%y' "$file" 2>/dev/null | cut -d' ' -f1)
    FILE_LIST="${FILE_LIST}- $file (modified: $mod_time)\n"
done <<< "$SELECTED_FILES"

# Craft the prompt for CLI-Agent
PROMPT="I would like you to come up with some enjoyable tasks you can perform for the user based on his recent computer activities. Here are some randomly selected files from the last month that seem relevant.
You do not need to read all of them (although you may), focus on generating conversation starters which the user will prompt the agent with.
Make them relevant in the sense that they kickstart a fun and engaging workflow.
Keep them short, include any relevant filepaths, be intelligent about the trajectories they lead the user towards.

Randomly selected recent files in ~/OneDrive/Studium:
${FILE_LIST}

Your task:
1. Read the content of these files to understand the topics, subjects, and current work.
2. Generate 5 creative, actionable prompts that would help me start my day productively.
3. Consider relationships between files (e.g., if there's a lab report and related lecture notes).
4. Mix different types of day starters: review, explore, challenge knowledge, visualize, etc.
5. Write the prompts to $SUGGESTED_PROMPTS_FILE in the correct JSON format:
   {
     \"prompts\": [
       \"Prompt 1 here\",
       \"Prompt 2 here\",
       ...
     ]
   }"

# Run CLI-Agent with restricted permissions
cd "$CLI_AGENT_DIR"

# Debug: Print full command before execution
echo ""
echo -e "${BLUE}=== Executing CLI-Agent Command ===${NC}"
echo -e "${YELLOW}python3 main.py \\${NC}"
echo -e "${YELLOW}    --read-only \\${NC}"
echo -e "${YELLOW}    --allow-write-path \"$SUGGESTED_PROMPTS_FILE\" \\${NC}"
echo -e "${YELLOW}    -m \"<prompt hidden for brevity>\" \\${NC}" # Hide long prompt
echo -e "${YELLOW}    --yolo${NC}"
echo ""

python3 main.py \
    --read-only \
    --allow-write-path "$SUGGESTED_PROMPTS_FILE" \
    -m "$PROMPT" \
    --yolo

# Check if the prompts file was updated
if [ -f "$SUGGESTED_PROMPTS_FILE" ]; then
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
    log_message "${RED}✗ Failed to update prompts file${NC}"
    exit 1
fi