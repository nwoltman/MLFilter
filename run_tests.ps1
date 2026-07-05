# SPDX-License-Identifier: Apache-2.0

#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

if (-not $SkipBuild) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Install Visual Studio with the C++ workload."
    }
    $msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild `
        -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
    if (-not $msbuild) {
        throw "MSBuild not found."
    }

    Write-Host "==> Building tests ($Configuration|x64)" -ForegroundColor Cyan
    & $msbuild (Join-Path $PSScriptRoot "MLFilter.sln") /t:test `
        /p:Configuration=$Configuration /p:Platform=x64 /m /v:minimal /nologo
    if ($LASTEXITCODE -ne 0) {
        throw "Test build failed (exit $LASTEXITCODE)."
    }
}

if (-not $env:CUDA_PATH) {
    throw "CUDA_PATH is not set. Install the CUDA Toolkit."
}
$cudaBin = @((Join-Path $env:CUDA_PATH "bin\x64"), (Join-Path $env:CUDA_PATH "bin")) |
    Where-Object { Test-Path $_ } | Select-Object -First 1
$env:PATH = "$cudaBin;$env:PATH"

$exe = Join-Path $PSScriptRoot "x64\$Configuration\test_x64.exe"
if (-not (Test-Path $exe)) {
    throw "Test executable not found: $exe"
}

Write-Host "==> Running conversion tests" -ForegroundColor Cyan
& $exe
exit $LASTEXITCODE
