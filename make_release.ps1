# SPDX-License-Identifier: Apache-2.0
# Creates the release folder by default. A semantic version also publishes it.
#
# Usage: .\make_release.ps1
#        .\make_release.ps1 1.2.3

#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidatePattern('^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$')]
    [string]$Version
)

$ErrorActionPreference = "Stop"
function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

$Configuration = "Release"
$OutputDir = Join-Path $PSScriptRoot "release"
$Repository = "nwoltman/MLFilter"
$ConsumerArchitectures = @("sm75", "sm86", "sm89", "sm120")
$ReleaseTag = if ($Version) { "v$Version" } else { "" }

if (-not $env:TENSORRT_ROOT) { throw "TENSORRT_ROOT is not set. See README 'Development setup'." }
if (-not $env:CUDA_PATH) { throw "CUDA_PATH is not set. Install the CUDA Toolkit." }

Write-Step "Building MLFilter $Configuration|x64"
& (Join-Path $PSScriptRoot "build.ps1") -Configuration $Configuration

$axPath = Join-Path $PSScriptRoot "x64\$Configuration\MLFilter_x64.ax"
if (-not (Test-Path $axPath)) {
    throw "Build completed without producing the expected filter: $axPath"
}
$benchmarkPath = Join-Path $PSScriptRoot "x64\$Configuration\benchmark_x64.exe"
if (-not (Test-Path $benchmarkPath)) {
    throw "Build completed without producing the expected benchmark: $benchmarkPath"
}

$trtBin = Join-Path $env:TENSORRT_ROOT "bin"
if (-not (Test-Path $trtBin)) { throw "TensorRT bin folder not found: $trtBin" }

# Resolve assets before modifying output. The installer and publisher must agree
# on the exact filenames, including the TensorRT major version.
$architectureDlls = @{}
foreach ($arch in $ConsumerArchitectures) {
    $matches = @(Get-ChildItem (Join-Path $trtBin "nvinfer_builder_resource_${arch}_*.dll"))
    if ($matches.Count -ne 1) {
        throw "Expected exactly one builder-resource DLL for $arch in $trtBin; found $($matches.Count)."
    }
    $architectureDlls[$arch] = $matches[0]
}

Write-Step "Output: $OutputDir"
if (Test-Path $OutputDir) { Remove-Item $OutputDir -Recurse -Force }
$binDir = Join-Path $OutputDir "bin"
New-Item -ItemType Directory -Path $binDir -Force | Out-Null

Write-Step "Copying MLFilter_x64.ax"
Copy-Item $axPath -Destination $binDir

Write-Step "Copying benchmark"
Copy-Item $benchmarkPath -Destination $binDir
Copy-Item (Join-Path $PSScriptRoot "release-scripts\run_benchmark.bat") -Destination $OutputDir

Write-Step "Creating release README.md"
$sourceReadme = Get-Content (Join-Path $PSScriptRoot "README.md") -Raw
$developmentSection = [regex]::Match($sourceReadme, '(?m)^## Development setup\s*$')
if (-not $developmentSection.Success) {
    throw "Could not find the '## Development setup' section in README.md."
}
$releaseReadme = $sourceReadme.Substring(0, $developmentSection.Index).TrimEnd() + [Environment]::NewLine
[System.IO.File]::WriteAllText(
    (Join-Path $OutputDir "README.md"),
    $releaseReadme,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Step "Copying architecture-independent TensorRT DLLs from $trtBin"
$copied = 0
foreach ($dll in Get-ChildItem (Join-Path $trtBin "*.dll")) {
    if ($dll.Name -match "^nvinfer_builder_resource_") { continue }
    if ($dll.Name -match "nvinfer_(lean|dispatch|vc_plugin)_") { continue }
    Copy-Item $dll -Destination $binDir
    $copied++
}
Write-Host ("    {0} TensorRT DLL(s) copied" -f $copied)

Write-Step "Writing installer scripts"
$releaseScriptsDir = Join-Path $PSScriptRoot "release-scripts"
Copy-Item (Join-Path $releaseScriptsDir "install.bat") -Destination $OutputDir
Copy-Item (Join-Path $releaseScriptsDir "uninstall.bat") -Destination $OutputDir
Copy-Item (Join-Path $releaseScriptsDir "update.bat") -Destination $OutputDir

$dependencyScript = Get-Content (Join-Path $releaseScriptsDir "bin\install_dependency.ps1") -Raw
$dependencyScript = $dependencyScript.Replace("__RELEASE_TAG__", $ReleaseTag)
$dependencyScript = $dependencyScript.Replace("__REPOSITORY__", $Repository)
$assetMap = ($ConsumerArchitectures | ForEach-Object {
    '    "{0}" = "{1}"' -f $_, $architectureDlls[$_].Name
}) -join [Environment]::NewLine
$dependencyScript = $dependencyScript.Replace("    # __ASSET_MAP__", $assetMap)

Set-Content (Join-Path $binDir "install_dependency.ps1") $dependencyScript -Encoding Ascii

$updateScript = Get-Content (Join-Path $releaseScriptsDir "bin\update.ps1") -Raw
$updateScript = $updateScript.Replace("__RELEASE_TAG__", $ReleaseTag)
$updateScript = $updateScript.Replace("__REPOSITORY__", $Repository)
Set-Content (Join-Path $binDir "update.ps1") $updateScript -Encoding Ascii

$totalBytes = (Get-ChildItem $OutputDir -Recurse -File | Measure-Object Length -Sum).Sum
$binCount = (Get-ChildItem $binDir -File).Count
Write-Host ""
Write-Step "Release folder ready: $OutputDir"
Write-Host ("    bin\ files: {0} (GPU architecture DLLs excluded)" -f $binCount)
Write-Host ("    Total size: {0:N1} MB" -f ($totalBytes / 1MB))

if (-not $Version) {
    Write-Host ""
    Write-Host "Dry run complete. Supply a semantic version (for example, .\make_release.ps1 1.2.3) to publish."
    return
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is required. Install it and run 'gh auth login'."
}
if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
    throw "Windows tar.exe is required to create the release zip."
}

$zipPath = Join-Path $PSScriptRoot "MLFilter-$ReleaseTag.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Write-Step "Creating $zipPath"
# PowerShell 5's Compress-Archive can fail while disposing archives containing
# large DLLs with "a user-mapped section open." Windows bsdtar streams them
# reliably and selects ZIP format from the .zip extension. Pass each top-level
# entry explicitly so the archive has normal relative paths and no Unix-style
# "./" root entry that can confuse Windows Explorer's compressed-folder handler.
$archiveEntries = @(Get-ChildItem $OutputDir | Select-Object -ExpandProperty Name)
& tar.exe -a -c -f $zipPath -C $OutputDir @archiveEntries
if ($LASTEXITCODE -ne 0) { throw "tar.exe failed with exit code $LASTEXITCODE." }
if (-not (Test-Path $zipPath) -or (Get-Item $zipPath).Length -eq 0) {
    throw "Archive creation completed without producing a valid file: $zipPath"
}

Write-Step "Creating GitHub release $ReleaseTag"
$assets = @($zipPath) + @($ConsumerArchitectures | ForEach-Object { $architectureDlls[$_].FullName })
& gh release create $ReleaseTag @assets --repo $Repository --title "MLFilter $ReleaseTag" --generate-notes
if ($LASTEXITCODE -ne 0) { throw "gh release create failed with exit code $LASTEXITCODE." }

$zipName = Split-Path $zipPath -Leaf
Remove-Item $zipPath -Force

Write-Host ""
Write-Step "Published $ReleaseTag"
Write-Host ("    Main download: {0}" -f $zipName)
Write-Host ("    GPU assets  : {0}" -f (($ConsumerArchitectures | ForEach-Object { $architectureDlls[$_].Name }) -join ", "))
