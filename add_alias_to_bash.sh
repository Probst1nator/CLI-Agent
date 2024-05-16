#!/bin/bash

# Get the directory of the currently executing script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Path to the main.py file
MAIN_PY="${SCRIPT_DIR}/main.py"

# Create a symbolic link in /usr/local/bin
sudo ln -sf "${MAIN_PY}" /usr/local/bin/cli-agent
