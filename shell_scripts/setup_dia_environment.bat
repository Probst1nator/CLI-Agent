@echo off
setlocal

REM Minimal script to install Dia 1.6B to the current Python environment

REM --- Configuration ---
set DIA_REPO_URL=https://github.com/nari-labs/dia.git
set DIA_TEMP_DIR=dia_temp

REM --- Main Script ---
echo INFO: Starting Dia installation to current Python environment...

REM 1. Check for prerequisites
echo INFO: Checking for Git and Python...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git could not be found. Please install it and ensure it's in your PATH.
    goto :cleanup_and_exit
)

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python could not be found. Please install it and ensure it's in your PATH.
    goto :cleanup_and_exit
)
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip could not be found. Ensure Python and pip are correctly installed.
    goto :cleanup_and_exit
)
echo INFO: Git and Python found.

REM 2. Clone Dia repository to temporary directory
echo INFO: Cloning Dia repository to temporary directory...
if exist "%DIA_TEMP_DIR%\" (
    echo INFO: Removing existing temporary directory...
    rmdir /s /q "%DIA_TEMP_DIR%"
)
git clone "%DIA_REPO_URL%" "%DIA_TEMP_DIR%" --depth 1
if %errorlevel% neq 0 (
    echo ERROR: Failed to clone Dia repository.
    goto :cleanup_and_exit
)
echo SUCCESS: Dia repository cloned to temporary directory.

REM 3. Install PyTorch with CUDA
echo INFO: Installing PyTorch (with CUDA 12.1 support), torchvision, and torchaudio...
echo WARNING: This step can take a while and requires a good internet connection.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch. Please check your internet connection and CUDA compatibility.
    goto :cleanup_and_exit
)
echo SUCCESS: PyTorch installed.

REM 4. Install soundfile
echo INFO: Installing soundfile...
pip install soundfile --force-reinstall
if %errorlevel% neq 0 (
    echo ERROR: Failed to install soundfile.
    goto :cleanup_and_exit
)
echo SUCCESS: soundfile installed.

REM 5. Install uv
echo INFO: Installing uv...
pip install uv
if %errorlevel% neq 0 (
    echo WARNING: Failed to install uv, but continuing with installation.
) else (
    echo SUCCESS: uv installed.
)

REM 6. Install Dia - First attempt with standard methods
echo INFO: Installing Dia to current environment...
cd "%DIA_TEMP_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to navigate to temporary directory.
    goto :cleanup_and_exit
)

REM First check if there's a setup.py file
if exist "setup.py" (
    echo INFO: Found setup.py. Installing Dia using pip...
    pip install -e .
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Dia using setup.py.
        set SETUP_FAILED=1
    ) else (
        echo SUCCESS: Dia installed using setup.py.
    )
REM Check if there's a requirements.txt file
) else if exist "requirements.txt" (
    echo INFO: Found requirements.txt. Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies from requirements.txt.
        set SETUP_FAILED=1
    ) else (
        echo SUCCESS: Dependencies installed from requirements.txt.
        REM Install the package itself in development mode if no setup.py
        echo INFO: Installing Dia package...
        pip install -e .
        if %errorlevel% neq 0 (
            echo WARNING: Installed dependencies but failed to install Dia package itself.
            set SETUP_FAILED=1
        ) else (
            echo SUCCESS: Dia package installed.
        )
    )
) else (
    echo WARNING: No setup.py or requirements.txt found. Installing Dia directory as a package...
    pip install -e .
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Dia as a package.
        set SETUP_FAILED=1
    ) else (
        echo SUCCESS: Dia installed as a package.
    )
)

REM Verify the installation
echo INFO: Verifying Dia installation...

REM Check for soundfile
python -c "import soundfile" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: The soundfile package is not properly installed.
    echo INFO: Attempting to install soundfile again...
    pip install soundfile --force-reinstall
) else (
    echo SUCCESS: soundfile is properly installed.
)

REM Check for Dia
python -c "import dia" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: The Dia package is not properly installed in the Python path.
    set VERIFY_FAILED=1
) else (
    echo SUCCESS: Dia is properly installed and accessible.
    set VERIFY_FAILED=0
)

REM If installation verification failed, try alternative methods
if "%VERIFY_FAILED%"=="1" (
    echo WARNING: Standard installation methods did not make Dia accessible in Python. Trying alternative approach...
    
    REM Get the site-packages directory
    for /f "tokens=*" %%a in ('python -c "import site; print(site.getsitepackages()[0])"') do set SITE_PACKAGES=%%a
    if "%SITE_PACKAGES%"=="" (
        echo ERROR: Could not determine site-packages directory.
    ) else (
        echo INFO: Found site-packages directory: %SITE_PACKAGES%
        
        REM Create a .pth file to add the Dia directory to Python path
        set CURRENT_DIR=%cd%
        echo %CURRENT_DIR% > "%SITE_PACKAGES%\dia.pth"
        if %errorlevel% neq 0 (
            echo ERROR: Failed to create .pth file in site-packages.
        ) else (
            echo SUCCESS: Created dia.pth file in %SITE_PACKAGES% to add Dia to Python path.
            
            REM Copy the dia module directly to site-packages as a fallback
            if exist "dia" (
                echo INFO: Copying Dia module directly to site-packages...
                xcopy /E /I /Y dia "%SITE_PACKAGES%\dia\"
                if %errorlevel% neq 0 (
                    echo ERROR: Failed to copy Dia module to site-packages.
                ) else (
                    echo SUCCESS: Copied Dia module to site-packages.
                    
                    REM Verify again
                    cd ..
                    python -c "import dia" >nul 2>&1
                    if %errorlevel% equ 0 (
                        echo SUCCESS: Dia is now properly installed and accessible!
                    ) else (
                        echo ERROR: Failed to install Dia even with alternative methods.
                        echo INFO: You may need to manually add the Dia directory to your PYTHONPATH:
                        echo INFO: set PYTHONPATH=%%PYTHONPATH%%;%CURRENT_DIR%
                    )
                )
            ) else (
                echo ERROR: Could not find 'dia' directory in the repository.
            )
        )
    )
) else (
    echo SUCCESS: Dia has been installed to your current Python environment!
    echo INFO: You can now import and use Dia in your Python scripts.
    cd ..
)

REM Create a test script to verify Dia installation
echo INFO: Creating a test script to verify Dia installation...
(
  echo try:
  echo     import dia
  echo     print("✅ Dia module imported successfully!")
  echo     try:
  echo         from dia.model import Dia
  echo         print("✅ Dia model class imported successfully!")
  echo         try:
  echo             print("⚠️ Note: Creating a Dia instance requires GPU with CUDA. This test will not create an actual instance.")
  echo             print("✅ Dia installation appears to be complete!")
  echo         except Exception as e:
  echo             print(f"❌ Error initializing Dia: {str(e)}")
  echo     except ImportError as e:
  echo         print(f"❌ Could not import Dia model: {str(e)}")
  echo except ImportError as e:
  echo     print(f"❌ Could not import dia module: {str(e)}")
  echo     print("Try adding the Dia directory to your PYTHONPATH or reinstalling.")
) > dia_test.py

echo INFO: You can run the test script with: python dia_test.py

:cleanup_and_exit
REM Clean up temporary files
echo INFO: Cleaning up temporary files...
if exist "%DIA_TEMP_DIR%\" (
    rmdir /s /q "%DIA_TEMP_DIR%"
    echo SUCCESS: Temporary files removed.
)

echo INFO: Installation complete.
endlocal 