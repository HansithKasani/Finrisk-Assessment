@echo off
REM Setup script for Windows
REM Runs the Python setup environment script

echo ========================================
echo AI Credit Risk Assessment - Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Run the setup script
python scripts\setup_environment.py

echo.
echo Setup complete!
pause
