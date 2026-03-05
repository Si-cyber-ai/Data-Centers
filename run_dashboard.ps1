# AI Data Center Cooling Optimization - Dashboard Launcher
Write-Host "Starting Data Center Cooling Dashboard..." -ForegroundColor Cyan
Write-Host ""

# Set working directory to script location
Set-Location -Path $PSScriptRoot

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Set PYTHONPATH to current directory
$env:PYTHONPATH = $PWD.Path

Write-Host "Checking Python environment..." -ForegroundColor Yellow
python --version
Write-Host ""

Write-Host "Launching Streamlit Dashboard..." -ForegroundColor Green
Write-Host "Dashboard will open in your browser at: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

streamlit run frontend/dashboard.py
