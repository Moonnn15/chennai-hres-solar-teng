@echo off
title Chennai HRES Platform Launcher
echo ==================================================================
echo        CHENNAI HRES - HYBRID RENEWABLE ENERGY PLATFORM            
echo             One-Click Startup Script (Local Virtual Env)          
echo ==================================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system PATH.
    echo Please install Python 3.10+ and select "Add Python to PATH".
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo [Setup] Creating Python virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo [Setup] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install/Upgrade dependencies
echo [Setup] Verifying and installing package dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install package dependencies.
    pause
    exit /b 1
)

:: Launch Streamlit app
echo.
echo ==================================================================
echo [Success] Starting HRES Streamlit Dev Server...
echo Your browser should open automatically to http://localhost:8501
echo ==================================================================
echo.
streamlit run dashboard_streamlit.py

pause
