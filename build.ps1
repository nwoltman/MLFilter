# SPDX-License-Identifier: Apache-2.0
#
# Builds MLFilter. Locates MSBuild via vswhere and builds the
# solution, so you don't need a Developer prompt.
#
# Usage:  .\build.ps1 [-Configuration Release|Debug] [-Version 1.2.3] [-Rebuild]

#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [ValidatePattern('^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$')]
    [string]$Version,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

# Windows treats environment variable names case-insensitively, but some process
# launchers fail if the inherited environment contains both Path and PATH.
$pathValue = $env:Path
[Environment]::SetEnvironmentVariable("PATH", $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable("Path", $pathValue, [EnvironmentVariableTarget]::Process)

# --- Locate MSBuild ------------------------------------------------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio (with the C++ workload)."
}

$msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" |
    Select-Object -First 1
if (-not $msbuild) {
    throw "MSBuild not found. Install the 'Desktop development with C++' workload in Visual Studio."
}

# --- Build ---------------------------------------------------------------------------
$solution = Join-Path $PSScriptRoot "MLFilter.sln"
$target = if ($Rebuild) { "Rebuild" } else { "Build" }

Write-Host "==> MSBuild: $msbuild" -ForegroundColor Cyan
Write-Host "==> $target $Configuration|x64" -ForegroundColor Cyan

$msbuildArguments = @(
    $solution,
    "/t:$target",
    "/p:Configuration=$Configuration",
    "/p:Platform=x64",
    "/m",
    "/v:minimal",
    "/nologo"
)
if ($Version) {
    $msbuildArguments += "/p:MLFilterVersion=$Version"
}

& $msbuild @msbuildArguments
if ($LASTEXITCODE -ne 0) {
    throw "Build failed (exit $LASTEXITCODE)."
}

$ax = Join-Path $PSScriptRoot "x64\$Configuration\MLFilter_x64.ax"
Write-Host ""
Write-Host "==> Build succeeded: $ax" -ForegroundColor Green
