#!/bin/bash
# shellcheck disable=SC1090
source "$(grep PYTHON_ENV_PATH /home/prob/repos/CLI-Agent/.env | cut -d '=' -f2)"

# Function to print colored text (mimicking Python's termcolor.colored)
print_colored() {
    local text="$1"
    local color="$2"
    
    case "$color" in
        "red")     echo -e "\e[31m$text\e[0m" ;;
        "green")   echo -e "\e[32m$text\e[0m" ;;
        "yellow")  echo -e "\e[33m$text\e[0m" ;;
        "blue")    echo -e "\e[34m$text\e[0m" ;;
        "magenta") echo -e "\e[35m$text\e[0m" ;;
        "cyan")    echo -e "\e[36m$text\e[0m" ;;
        *)         echo "$text" ;;
    esac
}

# Read clipboard content (requires xclip on Linux)
if command -v xclip > /dev/null 2>&1; then
    clipboard_content=$(xclip -selection clipboard -o 2>/dev/null)
elif command -v xsel > /dev/null 2>&1; then
    clipboard_content=$(xsel --clipboard --output 2>/dev/null)
else
    print_colored "Error: Neither xclip nor xsel found. Please install one of them to read clipboard." "red"
    echo "On Ubuntu/Debian: sudo apt install xclip"
    echo "On RHEL/Fedora: sudo dnf install xsel"
    exit 1
fi

# Check if clipboard has content
if [ -z "$clipboard_content" ] || [ -z "$(echo "$clipboard_content" | tr -d '[:space:]')" ]; then
    print_colored "No text in clipboard" "red"
    exit 1
fi

# Run generate_podcast.py with clipboard content, custom output directory, and interactive options
print_colored "Starting interactive podcast generation..." "green"
python3 /home/prob/repos/CLI-Agent/generate_podcast.py \
    -o /home/prob/OneDrive/Podcasts \
    --clipboard-content "$clipboard_content"

# Print a message to the console
print_colored "Script execution completed." "green"
echo "Press Enter to close this window..."
read
