@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py -3.12
) else (
  set PY=python
)

if not exist .venv (
  %PY% -m venv .venv
  if errorlevel 1 goto :err
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Base dependencies installed.
echo IMPORTANT: install CUDA-enabled PyTorch separately for your NVIDIA GPU.
echo Open: https://pytorch.org/get-started/locally/
echo Then verify with: python check_gpu.py
pause
exit /b 0

:err
echo Failed to create environment.
pause
exit /b 1
