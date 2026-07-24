@echo off
setlocal
rem Downloads this release's GPU-specific dependency, then registers MLFilter.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bin\install_dependency.ps1"
if errorlevel 1 (
    echo.
    echo Installation stopped because the GPU dependency could not be installed.
    pause
    exit /b 1
)
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)
echo Registering MLFilter...
regsvr32 "%~dp0bin\MLFilter_x64.ax"
echo.
echo NOTE: keep this folder where it is. Registration records the .ax's location,
echo so moving the folder will break the filter (just re-run install.bat after moving).
pause
