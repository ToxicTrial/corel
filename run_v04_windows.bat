@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_v04_windows.bat IMAGE CHECKPOINT
  exit /b 1
)
if "%~2"=="" (
  echo Need CutoutNet trained best.pt as the second argument.
  exit /b 1
)
python cutout_v04.py "%~1" --checkpoint "%~2" --output outputs\v04 --diagnostics --cut-svg --export-instances
if errorlevel 1 pause
