#!/bin/bash

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_SCRIPT="${PROJECT_DIR}/main.py"

# Check if main script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: ${MAIN_SCRIPT} does not exist!"
    exit 1
fi

# Construct the shell initialization and command with error handling and terminal pause
SHELL_INIT="export TERM=xterm-256color; source ~/.profile; source ~/.bashrc; echo \"Shell environment status:\"; echo \"Python path: \$(which python3)\"; echo \"Current PATH: \$PATH\"; echo \"Virtual env: \$VIRTUAL_ENV\""

BASE_CMD="echo \"\\nEnvironment loaded. Starting CLI-Agent...\\n\" && python3 ${MAIN_SCRIPT} --voice --local"

EXEC_CMD="/bin/bash --login -i -c '${SHELL_INIT} && { ${BASE_CMD} -m \"Hi, can you introduce yourself as Nova and give the user a friendly greeting?\" || { echo \"\\nAn error occurred. Press Enter to close this window.\"; read; }; }; read -p \"\\nPress Enter to close this window...\"'"

# Create desktop entry content
DESKTOP_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=CLI-Agent
Comment=CLI Agent for task automation
Exec=${EXEC_CMD}
Icon=terminal
Terminal=true
StartupNotify=true
Categories=Utility;Development;"

# Define the desktop file path
DESKTOP_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${DESKTOP_DIR}/cli-agent.desktop"

# Create directory if it doesn't exist
mkdir -p "${DESKTOP_DIR}"

# Write the desktop file
echo "${DESKTOP_CONTENT}" > "${DESKTOP_FILE}"
chmod 755 "${DESKTOP_FILE}"

echo "Successfully created/updated desktop entry at: ${DESKTOP_FILE}"

# Update desktop database
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${DESKTOP_DIR}"
fi

echo "You can now launch the application from your system's application menu" 