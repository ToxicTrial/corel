@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe tools\build_synthetic_dataset.py --count 20000 --size 768 --max-objects 8
pause
