@echo off
rem Registers the locally-built MLFilter .ax for testing in MPC-BE.
rem Usage: install_dev.bat [Release|Debug]   (defaults to Release)
rem
rem The dev build finds its TensorRT/CUDA DLLs via %TENSORRT_ROOT%\bin on PATH
rem (set during development setup), so no bin\ folder is bundled here.

setlocal
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Release"
set "AX=%~dp0x64\%CONFIG%\MLFilter_x64.ax"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0' -ArgumentList '%CONFIG%'"
    exit /b
)

if not exist "%AX%" (
    echo Build not found: %AX%
    echo Build it first, e.g.:  powershell -ExecutionPolicy Bypass -File "%~dp0make_dev.ps1" -Configuration %CONFIG%
    pause
    exit /b 1
)

echo Registering %CONFIG% build: %AX%
regsvr32 "%AX%"
