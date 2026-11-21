@echo off
REM Eco Tycoon - Windows Launcher
REM This script activates the virtual environment and starts the Streamlit app

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please run install_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python -m streamlit run app.py