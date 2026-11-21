@echo off
REM Eco Tycoon - Windows Installation Helper
REM This script creates a virtual environment and installs dependencies

cd /d "%~dp0"

echo ==========================================
echo Eco Tycoon - Windows Installation
echo ==========================================
echo.

echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    echo Make sure Python is installed and in PATH.
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install core dependencies.
    pause
    exit /b 1
)

echo.
echo Installation complete!
echo.
echo To run the application:
echo   1. Double-click run_app.bat
echo   OR
echo   2. Run: .venv\Scripts\activate && python -m streamlit run app.py
echo.
echo For advanced features (MaxEnt, LSTM, Earth Engine, GIS):
echo   Run: .venv\Scripts\activate && pip install -r requirements-advanced.txt
echo.
pause