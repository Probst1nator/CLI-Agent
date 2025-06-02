#!/bin/bash
# shellcheck disable=SC1090

# Source necessary environment setup files (copying from run_screen_agent.sh)
# Replace with actual paths if different
source /home/prob/repos/CLI-Agent/shell_scripts/setup_env.sh # Assuming this exists and is needed

# Navigate to the CLI-Agent repository directory
cd /home/prob/repos/CLI-Agent || { echo "Error: Could not change directory to /home/prob/repos/CLI-Agent"; exit 1; }

# Activate the Python virtual environment if necessary (copying from run_screen_agent.sh pattern)
# Replace with actual activation command if different
source .venv/bin/activate # Assuming a .venv exists

# Execute the CLI Agent with the -m argument
# Assuming the main entry point is run as a module
python -m cli_agent -m

# Keep the terminal open after execution (optional, similar to screen_agent)
echo "CLI Agent execution completed with -m argument."
echo "Press Enter to close this window..."
read
