#!/bin/bash
# shellcheck disable=SC1090
source "$(grep PYTHON_ENV_PATH /home/prob/repos/CLI-Agent/.env | cut -d '=' -f2)"

# Run the CLI-Agent with the screen grabber
python3 /home/prob/repos/CLI-Agent/generate_podcast.py -o /home/prob/OneDrive/Podcasts

# Print a message to the console
echo "Script execution completed."
echo "Press Enter to close this window..."
read
