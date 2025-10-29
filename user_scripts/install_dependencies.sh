#!/bin/bash
#
# This script installs all dependencies for the CLI-Agent project.
# It is self-contained and location-aware.
#
# 1. Installs system-level packages (apt).
# 2. Creates a Python virtual environment named '.venv' *inside this directory*.
# 3. Installs all required Python packages from 'requirements.txt' into it.
# 4. Sets up the .env file from .env.example if it doesn't exist.
#
# USAGE:
# You can run this script from any location, e.g.:
#   bash /path/to/CLI-Agent/install_dependencies.sh
#
# Or navigate to its directory and run:
#   cd CLI-Agent && ./install_dependencies.sh
#

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Determine the script's own directory to make paths reliable ---
# This ensures that no matter where the script is called from, all paths are correct.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Define the project root (one level up from user_scripts/) ---
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# --- Define Paths Relative to the Project Root ---
VENV_DIR="$PROJECT_ROOT/.venv"
REQS_FILE="$PROJECT_ROOT/requirements.txt"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"
ENV_FILE="$PROJECT_ROOT/.env"

echo "======================================================================"
echo "CLI-Agent Installation Script"
echo "======================================================================"
echo ""

echo "--- Installing System-Level Dependencies (apt) ---"
echo "This may ask for your password."

# Update package lists
sudo apt-get update

# Install all required system packages, including python3-venv for creating environments.
# CLI-Agent needs: Python 3, pip, venv, ffmpeg (for audio), build tools (for some Python packages)
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    ffmpeg \
    portaudio19-dev \
    libsndfile1

echo "--- System dependencies installed successfully. ---"
echo ""

# --- Set up Python Virtual Environment ---
if [ -d "$VENV_DIR" ]; then
    # Check if the existing venv is compatible with the current Python version
    if [ ! -f "$VENV_DIR/bin/python3" ] || ! "$VENV_DIR/bin/python3" --version &>/dev/null; then
        echo "--- Existing virtual environment is incompatible. Removing and recreating... ---"
        rm -rf "$VENV_DIR"
        echo "--- Creating Python virtual environment at '$VENV_DIR' ---"
        python3 -m venv "$VENV_DIR"
    else
        echo "--- Python virtual environment already exists at '$VENV_DIR'. ---"
    fi
else
    echo "--- Creating Python virtual environment at '$VENV_DIR' ---"
    python3 -m venv "$VENV_DIR"
fi

echo ""
echo "--- Activating virtual environment and installing Python packages ---"

# Activate the environment using its absolute path and install packages.
# This activation is temporary and only exists for the duration of this script.
source "$VENV_DIR/bin/activate"

# Upgrade pip to ensure it's the latest version within the venv
pip install --upgrade pip

# Install all Python packages from the requirements.txt file
echo "Installing Python packages (this may take a few minutes)..."
pip install -r "$REQS_FILE"

echo ""
echo "--- Applying Python 3.8 compatibility fixes ---"

# Activate the environment using its absolute path to run Python fixes
source "$VENV_DIR/bin/activate"

# Fix ChromaDB SQLite compatibility issue
python3 << 'EOF'
import os
import re

# Make ChromaDB import optional due to SQLite version issues
filepath = os.path.join(os.environ.get('PROJECT_ROOT', '.'), 'shared', 'common_utils.py')
if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if 'import chromadb' in content and 'try:' not in content.split('import chromadb')[0][-50:]:
        content = content.replace(
            'import chromadb\n',
            '''try:
    import chromadb
    CHROMADB_AVAILABLE = True
except (ImportError, RuntimeError):
    CHROMADB_AVAILABLE = False
'''
        )
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed ChromaDB import in {filepath}")

EOF

# Deactivate is not strictly necessary as the script will exit, but it's good practice.
deactivate

echo ""
echo "--- Setting up environment configuration ---"

# Set up .env file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE" ]; then
        echo "Creating .env file from .env.example..."
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo ".env file created. Please edit it to add your API keys."
    else
        echo "Warning: .env.example not found. You may need to create .env manually."
    fi
else
    echo ".env file already exists. Skipping creation."
fi

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "The Python environment has been created at the CLI-Agent project root."
echo ""
echo "Next Steps:"
echo "1. Activate the virtual environment:"
echo "   source \"$VENV_DIR/bin/activate\""
echo ""
echo "   Or, if you are in the project's root directory:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Edit the .env file to add your API keys:"
echo "   nano \"$ENV_FILE\""
echo ""
echo "3. Run CLI-Agent:"
echo "   python main.py              # CLI mode"
echo "   python main.py --gui        # Web GUI mode"
echo "   python main.py --help       # See all options"
echo ""
echo "To leave the environment, simply type:"
echo "   deactivate"
echo "======================================================================"
