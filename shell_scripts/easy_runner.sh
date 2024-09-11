#!/bin/bash

# Check if the virtual environment directory exists
if [ ! -d "env" ]; then
  echo "Virtual environment 'env' does not exist. Please create it first."
  exit 1
fi

# Activate the virtual environment
source env/bin/activate

# Run the main script
python3 main.py

# Deactivate the virtual environment
deactivate
