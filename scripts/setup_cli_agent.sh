#!/bin/bash
# Setup script for CLI-Agent command line integration
# Configures the cli-agent command for global system access

set -e

# Configuration
CLI_AGENT_DIR="/home/prob/repos/CLI-Agent"
CLI_AGENT_SCRIPT="$CLI_AGENT_DIR/cli-agent"
USER_BIN_DIR="$HOME/.local/bin"
SYSTEM_BIN_DIR="/usr/local/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

show_help() {
    echo "CLI-Agent Setup Script"
    echo ""
    echo "Usage:"
    echo "  ./setup_cli_agent.sh [options]"
    echo ""
    echo "Options:"
    echo "  --user        Install for current user only (~/.local/bin)"
    echo "  --system      Install system-wide (/usr/local/bin) [requires sudo]"
    echo "  --path        Add CLI-Agent directory to PATH in shell profile"
    echo "  --uninstall   Remove cli-agent command and PATH entries"
    echo "  --check       Check current installation status"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./setup_cli_agent.sh --user     # Install for current user"
    echo "  ./setup_cli_agent.sh --system   # Install system-wide"
    echo "  ./setup_cli_agent.sh --check    # Check installation"
}

check_installation() {
    print_status $BLUE "🔍 Checking CLI-Agent installation status..."
    echo ""

    # Check if cli-agent command exists
    if command -v cli-agent &> /dev/null; then
        CLI_AGENT_PATH=$(which cli-agent)
        print_status $GREEN "✅ cli-agent command found at: $CLI_AGENT_PATH"

        # Test the command
        if $CLI_AGENT_PATH --help &> /dev/null; then
            print_status $GREEN "✅ cli-agent command is working"
        else
            print_status $YELLOW "⚠️  cli-agent command found but not working properly"
        fi
    else
        print_status $YELLOW "⚠️  cli-agent command not found in PATH"
    fi

    # Check PATH for CLI-Agent directory
    if echo "$PATH" | grep -q "$CLI_AGENT_DIR"; then
        print_status $GREEN "✅ CLI-Agent directory is in PATH"
    else
        print_status $YELLOW "⚠️  CLI-Agent directory not in PATH"
    fi

    # Check shell profiles for CLI-Agent entries
    echo ""
    print_status $BLUE "Shell profile entries:"
    for profile in ~/.bashrc ~/.zshrc ~/.profile; do
        if [[ -f "$profile" ]]; then
            if grep -q "CLI-Agent\|cli-agent" "$profile"; then
                print_status $GREEN "✅ $profile contains CLI-Agent entries"
            else
                print_status $YELLOW "⚠️  $profile has no CLI-Agent entries"
            fi
        fi
    done

    # Check if Docker is available
    echo ""
    if command -v docker &> /dev/null; then
        print_status $GREEN "✅ Docker is available"
        if command -v docker-compose &> /dev/null; then
            print_status $GREEN "✅ Docker Compose is available"
        else
            print_status $YELLOW "⚠️  Docker Compose not found"
        fi
    else
        print_status $YELLOW "⚠️  Docker not found (required for -docker mode)"
    fi
}

install_user() {
    print_status $BLUE "📦 Installing CLI-Agent for current user..."

    # Create user bin directory
    mkdir -p "$USER_BIN_DIR"

    # Create symlink
    if [[ -L "$USER_BIN_DIR/cli-agent" ]]; then
        rm "$USER_BIN_DIR/cli-agent"
    fi

    ln -s "$CLI_AGENT_SCRIPT" "$USER_BIN_DIR/cli-agent"
    chmod +x "$CLI_AGENT_SCRIPT"

    print_status $GREEN "✅ Created symlink: $USER_BIN_DIR/cli-agent -> $CLI_AGENT_SCRIPT"

    # Add to PATH if not already there
    setup_path
}

install_system() {
    print_status $BLUE "📦 Installing CLI-Agent system-wide..."

    # Check if we have sudo access
    if ! sudo -n true 2>/dev/null; then
        print_status $YELLOW "⚠️  Requesting sudo access for system installation..."
    fi

    # Create symlink
    if [[ -L "$SYSTEM_BIN_DIR/cli-agent" ]]; then
        sudo rm "$SYSTEM_BIN_DIR/cli-agent"
    fi

    sudo ln -s "$CLI_AGENT_SCRIPT" "$SYSTEM_BIN_DIR/cli-agent"
    chmod +x "$CLI_AGENT_SCRIPT"

    print_status $GREEN "✅ Created system symlink: $SYSTEM_BIN_DIR/cli-agent -> $CLI_AGENT_SCRIPT"
}

setup_path() {
    print_status $BLUE "🛣️  Setting up PATH for CLI-Agent..."

    # Determine which shell profile to use
    SHELL_PROFILE=""
    if [[ -n "$ZSH_VERSION" ]]; then
        SHELL_PROFILE="$HOME/.zshrc"
    elif [[ -n "$BASH_VERSION" ]]; then
        SHELL_PROFILE="$HOME/.bashrc"
    else
        SHELL_PROFILE="$HOME/.profile"
    fi

    # Check if CLI-Agent is already in PATH
    if echo "$PATH" | grep -q "$USER_BIN_DIR"; then
        print_status $GREEN "✅ $USER_BIN_DIR already in PATH"
    else
        # Add ~/.local/bin to PATH
        echo "" >> "$SHELL_PROFILE"
        echo "# CLI-Agent setup" >> "$SHELL_PROFILE"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_PROFILE"
        print_status $GREEN "✅ Added $USER_BIN_DIR to PATH in $SHELL_PROFILE"
    fi

    # Add CLI-Agent environment variable
    if ! grep -q "CLI_AGENT_DIR" "$SHELL_PROFILE"; then
        echo "export CLI_AGENT_DIR=\"$CLI_AGENT_DIR\"" >> "$SHELL_PROFILE"
        print_status $GREEN "✅ Added CLI_AGENT_DIR environment variable"
    fi

    # Add alias for convenience
    if ! grep -q "alias ca=" "$SHELL_PROFILE"; then
        echo "alias ca='cli-agent'" >> "$SHELL_PROFILE"
        print_status $GREEN "✅ Added 'ca' alias for cli-agent"
    fi

    print_status $YELLOW "⚠️  Please restart your shell or run: source $SHELL_PROFILE"
}

uninstall() {
    print_status $BLUE "🗑️  Uninstalling CLI-Agent..."

    # Remove user symlink
    if [[ -L "$USER_BIN_DIR/cli-agent" ]]; then
        rm "$USER_BIN_DIR/cli-agent"
        print_status $GREEN "✅ Removed user symlink"
    fi

    # Remove system symlink (if exists and we have permission)
    if [[ -L "$SYSTEM_BIN_DIR/cli-agent" ]]; then
        if sudo -n true 2>/dev/null; then
            sudo rm "$SYSTEM_BIN_DIR/cli-agent"
            print_status $GREEN "✅ Removed system symlink"
        else
            print_status $YELLOW "⚠️  System symlink exists but cannot remove without sudo"
        fi
    fi

    # Remove PATH entries from shell profiles
    for profile in ~/.bashrc ~/.zshrc ~/.profile; do
        if [[ -f "$profile" ]]; then
            # Create backup
            cp "$profile" "${profile}.backup"

            # Remove CLI-Agent related lines
            sed -i '/# CLI-Agent setup/d' "$profile"
            sed -i '/CLI_AGENT_DIR/d' "$profile"
            sed -i '/alias ca=.*cli-agent/d' "$profile"

            print_status $GREEN "✅ Cleaned $profile (backup saved as ${profile}.backup)"
        fi
    done

    print_status $GREEN "✅ Uninstallation completed"
}

# Parse command line arguments
case "${1:-}" in
    --user)
        install_user
        ;;
    --system)
        install_system
        ;;
    --path)
        setup_path
        ;;
    --uninstall)
        uninstall
        ;;
    --check)
        check_installation
        ;;
    -h|--help)
        show_help
        ;;
    "")
        # Default: install for user
        print_status $BLUE "🚀 Setting up CLI-Agent (user installation)..."
        install_user
        print_status $GREEN "✅ Setup completed!"
        echo ""
        print_status $BLUE "💡 Usage examples:"
        echo "  cli-agent -docker 3              # Start 3 containers"
        echo "  cli-agent -m 'Hello world'       # Local execution"
        echo "  cli-agent --docker-status        # Check containers"
        echo "  ca -docker 5                     # Using alias"
        ;;
    *)
        print_status $RED "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac