# SPDX-License-Identifier: Apache-2.0
#
# Benchmarks MLFilter's TensorRT inference pipeline on a real video file, the way
# mpc-vapoursynth-scripts' benchmark.vpy benchmarks the vsmlrt trt.Model pipeline.
# Builds the benchmark exe, makes the TensorRT/CUDA runtime DLLs findable, and runs it.
#
# Run with the video to benchmark:  .\benchmark.ps1 "X:\clip.mkv" [-Frames 500]
# (benchmark.bat passes a fixed path so it can be launched by double-clicking.)

#Requires -Version 5.1
[CmdletBinding()]
param(
    # The video file to benchmark with. Decoded by ffmpeg; uses its native resolution.
    # Positional, so the path can be passed without naming it: .\benchmark.ps1 "X:\clip.mkv"
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Video,

    # ONNX model to build the engine from. Empty -> use the model configured in MLFilter's
    # settings (HKCU\Software\MLFilter\ModelPath), same as the filter does during playback.
    [string]$Model = "",

    [int]$Frames = 300,   # frames to time after warmup (0 = run to end of file)
    [int]$Warmup = 10,    # frames to discard before timing (engine/autotune warmup)
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

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
    Write-Host "==> Building benchmark ($Configuration|x64)" -ForegroundColor Cyan
    & $msbuild $solution /t:benchmark /p:Configuration=$Configuration /p:Platform=x64 /m /v:minimal /nologo
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
$arguments = @($Video, "--frames", $Frames, "--warmup", $Warmup)
if ($Model) { $arguments += @("--model", $Model) }

Write-Host "==> $exe" -ForegroundColor Cyan
& $exe @arguments
exit $LASTEXITCODE
