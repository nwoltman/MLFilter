# SPDX-License-Identifier: Apache-2.0
#
# Benchmarks MLFilter's TensorRT inference pipeline on a deterministic synthetic frame.
# Builds the benchmark exe, makes the TensorRT/CUDA runtime DLLs findable, and runs it.
#
# Defaults to a 1920x1080 NV12 frame: .\benchmark.ps1

#Requires -Version 5.1
[CmdletBinding()]
param(
    [int]$Width = 1920,
    [int]$Height = 1080,
    [ValidateSet("nv12", "p010")]
    [string]$Format = "nv12",
    [ValidateSet("software", "d3d11")]
    [string]$Upload = "software",

    # ONNX model to build the engine from. Empty -> use the model configured in MLFilter's
    # settings (HKCU\Software\MLFilter\HDModelPath), same as the filter does during playback.
    [string]$Model = "",

    [int]$Frames = 400,   # frames to time after warmup
    [int]$Warmup = 24,    # frames to discard before timing (engine/autotune warmup)
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [switch]$PipelineOnly, # omit CUDA events and individual stage timing code from the build
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# Windows treats environment variable names case-insensitively, but some process
# launchers fail if the inherited environment contains both Path and PATH.
$pathValue = $env:Path
[Environment]::SetEnvironmentVariable("PATH", $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable("Path", $pathValue, [EnvironmentVariableTarget]::Process)

# --- Build the benchmark project -----------------------------------------------------
if (-not $SkipBuild) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Install Visual Studio (C++ workload)." }
    $msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" |
        Select-Object -First 1
    if (-not $msbuild) { throw "MSBuild not found. Install the 'Desktop development with C++' workload." }

    # Build via the solution (not the .vcxproj directly) so $(SolutionDir) resolves and the
    # zimg dependency lands in the shared output dir — building the project alone misplaces it.
    $solution = Join-Path $PSScriptRoot "MLFilter.sln"
    $stageTimings = if ($PipelineOnly) { "false" } else { "true" }
    Write-Host "==> Building benchmark ($Configuration|x64, stage timings: $stageTimings)" -ForegroundColor Cyan
    & $msbuild $solution /t:benchmark /p:Configuration=$Configuration /p:Platform=x64 `
        /p:MLFilterEnableStageTimings=$stageTimings /m /v:minimal /nologo
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)." }
}

$exe = Join-Path $PSScriptRoot "x64\$Configuration\benchmark_x64.exe"
if (-not (Test-Path $exe)) { throw "Benchmark exe not found: $exe (build it without -SkipBuild)." }

# --- Make the TensorRT + CUDA runtime DLLs findable ----------------------------------
# (The release bundles these in bin\; for a dev run we point PATH at the SDK folders.)
if (-not $env:TENSORRT_ROOT) { throw "TENSORRT_ROOT is not set. See README 'Development setup'." }
if (-not $env:CUDA_PATH)     { throw "CUDA_PATH is not set. Install the CUDA Toolkit." }

$trtBin  = Join-Path $env:TENSORRT_ROOT "bin"
$cudaBin = @((Join-Path $env:CUDA_PATH "bin\x64"), (Join-Path $env:CUDA_PATH "bin")) |
    Where-Object { Test-Path $_ } | Select-Object -First 1
$env:PATH = "$trtBin;$cudaBin;$env:PATH"

# --- Run -----------------------------------------------------------------------------
$arguments = @("--width", $Width, "--height", $Height, "--format", $Format, "--upload", $Upload,
               "--frames", $Frames, "--warmup", $Warmup)
if ($Model) { $arguments += @("--model", $Model) }

Write-Host "==> $exe" -ForegroundColor Cyan
& $exe @arguments
exit $LASTEXITCODE
