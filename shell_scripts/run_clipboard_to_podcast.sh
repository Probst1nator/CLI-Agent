#!/bin/bash
# shellcheck disable=SC1090
source "$(grep PYTHON_ENV_PATH /home/prob/repos/CLI-Agent/.env | cut -d '=' -f2)"

# Run generate_podcast.py with a custom output directory
python3 /home/prob/repos/CLI-Agent/generate_podcast.py -o /home/prob/OneDrive/Podcasts
python3 /home/prob/repos/CLI-Agent/generate_podcast.py -o /home/prob/OneDrive/Podcasts --llms "gemma3:1b"

# Print a message to the console
echo "Script execution completed."
echo "Press Enter to close this window..."
read
