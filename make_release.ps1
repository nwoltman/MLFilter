# SPDX-License-Identifier: Apache-2.0
#
# Packages a redistributable MLFilter release: the .ax in the root, an install.bat,
# and a bin\ folder with every runtime DLL the filter needs (TensorRT + CUDA), so it
# works on machines that do not have the TensorRT/CUDA SDKs installed.
#
# Usage:  .\make_release.ps1   (no parameters; the build is always the same)

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

# Fixed, reproducible release. Output goes to release\.
$Configuration = "Release"
$OutputDir = Join-Path $PSScriptRoot "release"

# A single release that works for most people: TensorRT builder-resource DLLs for
# consumer GPU architectures, excluding PTX and datacenter archs (sm80/sm90/sm100).
# cuBLAS and the unused lean/dispatch/vc_plugin runtimes are also excluded.
#   sm75 Turing(20xx/16xx)  sm86 Ampere(30xx)  sm89 Ada(40xx)  sm120 Blackwell(50xx)
$ConsumerArchitectures = @("sm75", "sm86", "sm89", "sm120")

# --- Validate prerequisites ----------------------------------------------------------
if (-not $env:TENSORRT_ROOT) { throw "TENSORRT_ROOT is not set. See README 'Development setup'." }
if (-not $env:CUDA_PATH)     { throw "CUDA_PATH is not set. Install the CUDA Toolkit." }

$axPath = Join-Path $PSScriptRoot "x64\$Configuration\MLFilter_x64.ax"
if (-not (Test-Path $axPath)) {
    throw "Built filter not found: $axPath`nBuild the $Configuration|x64 configuration first (msbuild MLFilter.sln /p:Configuration=$Configuration /p:Platform=x64)."
}

$trtBin = Join-Path $env:TENSORRT_ROOT "bin"
if (-not (Test-Path $trtBin)) { throw "TensorRT bin folder not found: $trtBin" }

# CUDA 13 puts the redistributable runtime DLLs in bin\x64; older layouts use bin.
$cudaBin = @((Join-Path $env:CUDA_PATH "bin\x64"), (Join-Path $env:CUDA_PATH "bin")) |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $cudaBin) { throw "Could not find a CUDA bin folder under $env:CUDA_PATH." }

# --- Prepare output ------------------------------------------------------------------
Write-Step "Output: $OutputDir"
if (Test-Path $OutputDir) { Remove-Item $OutputDir -Recurse -Force }
$binDir = Join-Path $OutputDir "bin"
New-Item -ItemType Directory -Path $binDir -Force | Out-Null

# --- Copy the filter -----------------------------------------------------------------
Write-Step "Copying MLFilter_x64.ax"
Copy-Item $axPath -Destination $OutputDir

# --- Copy TensorRT DLLs (runtime + ONNX parser + selected builder-resource DLLs) -----
Write-Step "Copying TensorRT DLLs from $trtBin"
Write-Host ("    Builder-resource architectures: {0}" -f ($ConsumerArchitectures -join ", "))
$copied = 0
foreach ($dll in Get-ChildItem (Join-Path $trtBin "*.dll")) {
    $name = $dll.Name

    # Exclude runtimes we don't use (we link the full nvinfer runtime).
    if ($name -match "nvinfer_(lean|dispatch|vc_plugin)_") { continue }

    # Keep only builder-resource DLLs for the consumer architectures.
    if ($name -like "nvinfer_builder_resource_*") {
        $keep = $false
        foreach ($arch in $ConsumerArchitectures) {
            if ($name -like "nvinfer_builder_resource_${arch}_*") { $keep = $true; break }
        }
        if (-not $keep) { continue }
    }

    Copy-Item $dll -Destination $binDir
    $copied++
}
Write-Host ("    {0} TensorRT DLL(s) copied" -f $copied)

# --- Copy the CUDA runtime DLLs TensorRT loads dynamically ---------------------------
Write-Step "Copying CUDA runtime DLLs from $cudaBin"
# cuBLAS is excluded: TensorRT 10+ keeps cuBLAS tactics off by default.
$cudaPatterns = @(
    "cudart64_*.dll",          # CUDA runtime (required)
    "nvrtc64_*.dll",           # runtime kernel compilation
    "nvrtc-builtins64_*.dll",
    "nvJitLink_*.dll",         # JIT/PTX linking
    "nvfatbin_*.dll"
)
foreach ($pattern in $cudaPatterns) {
    $matches = Get-ChildItem (Join-Path $cudaBin $pattern) -ErrorAction SilentlyContinue
    if ($matches) {
        $matches | Copy-Item -Destination $binDir
        $matches | ForEach-Object { Write-Host ("    {0}" -f $_.Name) }
    } else {
        Write-Warning "No CUDA DLL matched '$pattern' (skipped)."
    }
}

# --- install.bat / uninstall.bat -----------------------------------------------------
Write-Step "Writing install.bat and uninstall.bat"

$installBat = @'
@echo off
rem Registers MLFilter with Windows so media players can use it.
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)
echo Registering MLFilter...
regsvr32 "%~dp0MLFilter_x64.ax"
echo.
echo NOTE: keep this folder where it is. Registration records the .ax's location,
echo so moving the folder will break the filter (just re-run install.bat after moving).
pause
'@

$uninstallBat = @'
@echo off
rem Unregisters MLFilter.
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)
echo Unregistering MLFilter...
regsvr32 /u "%~dp0MLFilter_x64.ax"
pause
'@

Set-Content -Path (Join-Path $OutputDir "install.bat")   -Value $installBat   -Encoding Ascii
Set-Content -Path (Join-Path $OutputDir "uninstall.bat") -Value $uninstallBat -Encoding Ascii

# --- Summary -------------------------------------------------------------------------
$totalBytes = (Get-ChildItem $OutputDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
$binCount = (Get-ChildItem $binDir -File).Count
Write-Host ""
Write-Step ("Release ready: {0}" -f $OutputDir)
Write-Host ("    bin\ DLLs : {0}" -f $binCount)
Write-Host ("    Total size: {0:N1} MB" -f ($totalBytes / 1MB))
Write-Host ""
Write-Host "Contents:" -ForegroundColor Green
Write-Host "    MLFilter_x64.ax     the filter"
Write-Host "    install.bat         registers the filter (run as administrator)"
Write-Host "    uninstall.bat       unregisters the filter"
Write-Host "    bin\                bundled TensorRT + CUDA runtime DLLs"
