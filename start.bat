@echo off
echo ============================================
echo   Anima LoRA Training Web UI
echo   Default port: 9005 (auto-fallback if busy)
echo ============================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer, then run this again.
    pause
    exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate

echo Installing/updating Python dependencies...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

if exist sd-scripts-main\requirements.txt (
    if not exist sd-scripts-main\venv (
        echo.
        echo Preparing bundled sd-scripts environment for local training...
        echo This first-time setup can take a while because PyTorch is large.
        python -m venv sd-scripts-main\venv
        if errorlevel 1 (
            echo Failed to create sd-scripts virtual environment.
            pause
            exit /b 1
        )

        call sd-scripts-main\venv\Scripts\activate
        python -m pip install --upgrade pip
        if errorlevel 1 (
            echo Failed to upgrade pip in sd-scripts environment.
            pause
            exit /b 1
        )

        python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
        if errorlevel 1 (
            echo Failed to install PyTorch for sd-scripts.
            pause
            exit /b 1
        )

        python -m pip install -r sd-scripts-main\requirements.txt
        if errorlevel 1 (
            echo Failed to install sd-scripts requirements.
            pause
            exit /b 1
        )

        call deactivate
        call venv\Scripts\activate
    )
)

python app.py
pause
