#!/bin/bash
# shellcheck disable=SC1090
source "$(grep PYTHON_ENV_PATH ~/repos/CLI-Agent/.env | cut -d '=' -f2)"

# Run the CLI-Agent with the screen grabber
python3 ~/repos/CLI-Agent/generate_podcast.py

# Print a message to the console
echo "Script execution completed."
echo "Press Enter to close this window..."
read
