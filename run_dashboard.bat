@echo off
REM AI Data Center Cooling Optimization - Dashboard Launcher
echo Starting Data Center Cooling Dashboard...
echo.

cd /d "%~dp0"
set PYTHONPATH=%CD%

echo Checking Python environment...
python --version
echo.

echo Launching Streamlit Dashboard...
echo Dashboard will open in your browser at: http://localhost:8501
echo.

streamlit run frontend/dashboard.py

pause
