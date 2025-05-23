#!/bin/bash

# Minimal script to install Dia 1.6B to the current Python environment

# --- Configuration ---
DIA_REPO_URL="https://github.com/nari-labs/dia.git"
DIA_TEMP_DIR="dia_temp" # Temporary directory for the repo

# --- Helper Functions ---
print_info() {
    echo -e "\033[34mINFO: $1\033[0m"
}

print_success() {
    echo -e "\033[32mSUCCESS: $1\033[0m"
}

print_warning() {
    echo -e "\033[33mWARNING: $1\033[0m"
}

print_error() {
    echo -e "\033[31mERROR: $1\033[0m"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 could not be found. Please install it and ensure it's in your PATH."
        exit 1
    fi
}

cleanup() {
    print_info "Cleaning up temporary files..."
    if [ -d "$DIA_TEMP_DIR" ]; then
        rm -rf "$DIA_TEMP_DIR"
        print_success "Temporary files removed."
    fi
}

verify_installation() {
    print_info "Verifying Dia installation..."
    
    # Check for soundfile
    python3 -c "import soundfile" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "The soundfile package is not properly installed."
        print_info "Attempting to install soundfile again..."
        pip3 install soundfile --force-reinstall
    else
        print_success "soundfile is properly installed."
    fi

    # Check for Dia
    python3 -c "import dia" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "The Dia package is not properly installed in the Python path."
        return 1
    else
        print_success "Dia is properly installed and accessible."
        return 0
    fi
}

# Handle script interruption, but don't clean up if we're going to try alternate installation
trap cleanup EXIT

# --- Main Script ---
print_info "Starting Dia installation to current Python environment..."

# 1. Check for prerequisites
print_info "Checking for Git and Python3..."
check_command "git"
check_command "python3"
check_command "pip3"

# 2. Clone Dia repository to temporary directory
print_info "Cloning Dia repository to temporary directory..."
if [ -d "$DIA_TEMP_DIR" ]; then
    rm -rf "$DIA_TEMP_DIR"
fi
git clone "$DIA_REPO_URL" "$DIA_TEMP_DIR" --depth 1
if [ $? -ne 0 ]; then
    print_error "Failed to clone Dia repository."
    exit 1
fi
print_success "Dia repository cloned to temporary directory."

# 3. Install PyTorch with CUDA
print_info "Installing PyTorch (with CUDA 12.1 support), torchvision, and torchaudio..."
print_warning "This step can take a while and requires a good internet connection."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    print_error "Failed to install PyTorch. Please check your internet connection and CUDA compatibility."
    exit 1
fi
print_success "PyTorch installed."

# 4. Install soundfile
print_info "Installing soundfile..."
pip3 install soundfile
if [ $? -ne 0 ]; then
    print_error "Failed to install soundfile."
    exit 1
fi
print_success "soundfile installed."

# 5. Install uv
print_info "Installing uv..."
pip3 install uv
if [ $? -ne 0 ]; then
    print_warning "Failed to install uv, but continuing with installation."
else
    print_success "uv installed."
fi

# 6. Install Dia - First attempt with standard methods
print_info "Installing Dia to current environment..."
cd "$DIA_TEMP_DIR" || { print_error "Failed to navigate to temporary directory."; exit 1; }

# First check if there's a setup.py file
if [ -f "setup.py" ]; then
    print_info "Found setup.py. Installing Dia using pip..."
    pip3 install -e .
    if [ $? -ne 0 ]; then
        print_error "Failed to install Dia using setup.py."
        SETUP_FAILED=true
    else
        print_success "Dia installed using setup.py."
    fi
# Check if there's a requirements.txt file
elif [ -f "requirements.txt" ]; then
    print_info "Found requirements.txt. Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies from requirements.txt."
        SETUP_FAILED=true
    else
        print_success "Dependencies installed from requirements.txt."
        # Install the package itself in development mode if no setup.py
        print_info "Installing Dia package..."
        pip3 install -e .
        if [ $? -ne 0 ]; then
            print_warning "Installed dependencies but failed to install Dia package itself."
            SETUP_FAILED=true
        else
            print_success "Dia package installed."
        fi
    fi
else
    print_warning "No setup.py or requirements.txt found. Installing Dia directory as a package..."
    pip3 install -e .
    if [ $? -ne 0 ]; then
        print_error "Failed to install Dia as a package."
        SETUP_FAILED=true
    else
        print_success "Dia installed as a package."
    fi
fi

# Verify the installation
verify_installation
VERIFY_RESULT=$?

# If installation verification failed, try alternative methods
if [ $VERIFY_RESULT -ne 0 ]; then
    print_warning "Standard installation methods did not make Dia accessible in Python. Trying alternative approach..."
    
    # Get the site-packages directory
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    if [ -z "$SITE_PACKAGES" ]; then
        print_error "Could not determine site-packages directory."
    else
        print_info "Found site-packages directory: $SITE_PACKAGES"
        
        # Create a .pth file to add the Dia directory to Python path
        CURRENT_DIR=$(pwd)
        echo "$CURRENT_DIR" > "$SITE_PACKAGES/dia.pth"
        if [ $? -ne 0 ]; then
            print_error "Failed to create .pth file in site-packages."
        else
            print_success "Created dia.pth file in $SITE_PACKAGES to add Dia to Python path."
            
            # Copy the dia module directly to site-packages as a fallback
            if [ -d "dia" ]; then
                print_info "Copying Dia module directly to site-packages..."
                cp -r dia "$SITE_PACKAGES/"
                if [ $? -ne 0 ]; then
                    print_error "Failed to copy Dia module to site-packages."
                else
                    print_success "Copied Dia module to site-packages."
                    
                    # Verify again
                    cd ..
                    verify_installation
                    if [ $? -eq 0 ]; then
                        print_success "Dia is now properly installed and accessible!"
                    else
                        print_error "Failed to install Dia even with alternative methods."
                        print_info "You may need to manually add the Dia directory to your PYTHONPATH:"
                        print_info "export PYTHONPATH=\$PYTHONPATH:$CURRENT_DIR"
                    fi
                fi
            else
                print_error "Could not find 'dia' directory in the repository."
            fi
        fi
    fi
else
    print_success "Dia has been installed to your current Python environment!"
    print_info "You can now import and use Dia in your Python scripts."
    cd ..
fi

# Create a test script to verify Dia is working
print_info "Creating a test script to verify Dia installation..."
cat > dia_test.py << 'EOL'
try:
    import dia
    print("✅ Dia module imported successfully!")
    try:
        from dia.model import Dia
        print("✅ Dia model class imported successfully!")
        try:
            print("⚠️ Note: Creating a Dia instance requires GPU with CUDA. This test will not create an actual instance.")
            print("✅ Dia installation appears to be complete!")
        except Exception as e:
            print(f"❌ Error initializing Dia: {str(e)}")
    except ImportError as e:
        print(f"❌ Could not import Dia model: {str(e)}")
except ImportError as e:
    print(f"❌ Could not import dia module: {str(e)}")
    print("Try adding the Dia directory to your PYTHONPATH or reinstalling.")
EOL

print_info "You can run the test script with: python3 dia_test.py"
print_info "Installation complete." 