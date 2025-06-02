#!/bin/bash

# Change to the CLI-Agent directory
cd /home/prob/repos/CLI-Agent/

# Load environment variables (if needed, adjust path if your .env is elsewhere)
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Activate the Python virtual environment (adjust path if your venv is elsewhere)
source .venv/bin/activate

# Execute the main agent script with the -m argument
python3 main.py -m

# Print a message to the console
echo "Minimal agent execution completed."
echo "Press Enter to close this window..."
read
