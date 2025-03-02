@echo off
REM ===== Self-Elevate if not running as administrator =====
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~f0' -Verb runAs"
    exit /b
)

REM ===== Change to the script’s directory =====
cd /d "%~dp0"

REM ===== Locate vswhere.exe =====
set "vswherePath=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%vswherePath%" (
    echo ERROR: vswhere.exe not found! Please ensure Visual Studio is installed.
    pause
    exit /b 1
)

REM ===== Get the latest MSVC installation path using vswhere.exe =====
for /f "usebackq delims=" %%i in (`"%vswherePath%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "vcInstPath=%%i"
    goto :gotPath
)
:gotPath
if not defined vcInstPath (
    echo ERROR: MSVC installation not found.
    pause
    exit /b 1
)

set "vcvarsPath=%vcInstPath%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%vcvarsPath%" (
    echo ERROR: vcvars64.bat could not be found! Please make sure MSVC is installed properly.
    pause
    exit /b 1
)

echo Setting up MSVC environment...
call "%vcvarsPath%"
REM Environment variables from vcvars64.bat are now set.

REM ===== Activate Python virtual environment =====
if exist "venv\Scripts\activate.bat" (
    echo Activating Python venv from venv\Scripts\activate.bat...
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating Python venv from .venv\Scripts\activate.bat...
    call ".venv\Scripts\activate.bat"
) else (
    echo No Python virtual environment activation script found.
)

REM ===== Set additional environment variables =====
set "HF_HOME=%~dp0\huggingface"
set "TORCH_HOME=%~dp0\torch"
set "XFORMERS_FORCE_DISABLE_TRITON=1"
set "CUDA_HOME=%CUDA_PATH%"
set "PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll"
set "GRADIO_HOST=127.0.0.1"

REM ===== Run the Python module =====
python -m gradio_interface

pause
