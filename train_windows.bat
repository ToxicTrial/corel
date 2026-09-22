@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe check_gpu.py
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe train.py
pause
