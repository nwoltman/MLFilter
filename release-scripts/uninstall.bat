@echo off
rem Unregisters MLFilter.
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)
echo Unregistering MLFilter...
regsvr32 /u "%~dp0bin\MLFilter_x64.ax"
pause
