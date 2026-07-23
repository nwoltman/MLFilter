# SPDX-License-Identifier: Apache-2.0

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$WaitForInput
)

$ErrorActionPreference = "Stop"
$releaseTag = "__RELEASE_TAG__"
$repository = "__REPOSITORY__"

function Exit-Updater([int]$ExitCode) {
    if ($WaitForInput) {
        Write-Host ""
        Read-Host "Press Enter to close" | Out-Null
    }

    exit $ExitCode
}

function ConvertTo-SemanticVersion([string]$Tag) {
    $match = [regex]::Match(
        $Tag,
        '^v?(?<major>0|[1-9]\d*)\.(?<minor>0|[1-9]\d*)\.(?<patch>0|[1-9]\d*)(?:-(?<pre>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$'
    )
    if (-not $match.Success) {
        throw "Release tag '$Tag' is not a supported semantic version."
    }

    [pscustomobject]@{
        Major = [uint64]$match.Groups["major"].Value
        Minor = [uint64]$match.Groups["minor"].Value
        Patch = [uint64]$match.Groups["patch"].Value
        Prerelease = if ($match.Groups["pre"].Success) {
            @($match.Groups["pre"].Value.Split("."))
        } else {
            @()
        }
    }
}

function Compare-SemanticVersion($Left, $Right) {
    foreach ($property in @("Major", "Minor", "Patch")) {
        if ($Left.$property -lt $Right.$property) { return -1 }
        if ($Left.$property -gt $Right.$property) { return 1 }
    }

    if ($Left.Prerelease.Count -eq 0 -and $Right.Prerelease.Count -eq 0) { return 0 }
    if ($Left.Prerelease.Count -eq 0) { return 1 }
    if ($Right.Prerelease.Count -eq 0) { return -1 }

    $identifierCount = [Math]::Max($Left.Prerelease.Count, $Right.Prerelease.Count)
    for ($index = 0; $index -lt $identifierCount; $index++) {
        if ($index -ge $Left.Prerelease.Count) { return -1 }
        if ($index -ge $Right.Prerelease.Count) { return 1 }

        $leftIdentifier = $Left.Prerelease[$index]
        $rightIdentifier = $Right.Prerelease[$index]
        $leftNumeric = $leftIdentifier -match '^\d+$'
        $rightNumeric = $rightIdentifier -match '^\d+$'

        if ($leftNumeric -and $rightNumeric) {
            $leftNumber = [uint64]$leftIdentifier
            $rightNumber = [uint64]$rightIdentifier
            if ($leftNumber -lt $rightNumber) { return -1 }
            if ($leftNumber -gt $rightNumber) { return 1 }
        } elseif ($leftNumeric) {
            return -1
        } elseif ($rightNumeric) {
            return 1
        } else {
            $comparison = [string]::CompareOrdinal($leftIdentifier, $rightIdentifier)
            if ($comparison -lt 0) { return -1 }
            if ($comparison -gt 0) { return 1 }
        }
    }

    return 0
}

function Assert-SafeArchive([string]$ArchivePath) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem

    $archive = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        foreach ($entry in $archive.Entries) {
            $path = $entry.FullName.Replace("/", "\")
            $segments = @($path.Split("\"))
            if ([System.IO.Path]::IsPathRooted($path) -or $segments -contains "..") {
                throw "The release archive contains an unsafe path: $($entry.FullName)"
            }
        }
    } finally {
        $archive.Dispose()
    }
}

function Assert-FileReplaceable([string]$Path, [string]$DisplayPath) {
    try {
        $stream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
        $stream.Dispose()
    } catch {
        throw "Cannot replace '$DisplayPath'. Close video players and other programs using MLFilter, then try again."
    }
}

function Assert-FilesReplaceable([string]$SourceDirectory, [string]$DestinationDirectory) {
    foreach ($sourceFile in Get-ChildItem $SourceDirectory -Recurse -File) {
        $relativePath = $sourceFile.FullName.Substring($SourceDirectory.Length).TrimStart("\")
        $destination = Join-Path $DestinationDirectory $relativePath
        if (Test-Path $destination -PathType Leaf) {
            Assert-FileReplaceable $destination $relativePath
        }
    }
}

try {
    if (-not $releaseTag) {
        throw "This folder was made by a dry run and is not tied to a GitHub release."
    }

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $headers = @{
        Accept = "application/vnd.github+json"
        "User-Agent" = "MLFilter-updater"
    }
    $latestReleaseUri = "https://api.github.com/repos/$repository/releases/latest"

    Write-Host "Checking for MLFilter updates..."
    $latestRelease = Invoke-RestMethod -UseBasicParsing -Headers $headers -Uri $latestReleaseUri
    $latestTag = [string]$latestRelease.tag_name
    $currentVersion = ConvertTo-SemanticVersion $releaseTag
    $latestVersion = ConvertTo-SemanticVersion $latestTag
    $comparison = Compare-SemanticVersion $currentVersion $latestVersion

    if ($comparison -ge 0) {
        Write-Host "MLFilter $releaseTag is already up to date."
        Exit-Updater 0
    }

    Write-Host "MLFilter $latestTag is available (currently installed: $releaseTag)"
    if ($CheckOnly) { Exit-Updater 0 }

    $assetName = "MLFilter-$latestTag.zip"
    $assets = @($latestRelease.assets | Where-Object { $_.name -eq $assetName })
    if ($assets.Count -ne 1) {
        throw "Expected exactly one '$assetName' asset in release $latestTag; found $($assets.Count)."
    }

    $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("MLFilter-update-" + [guid]::NewGuid().ToString("N"))
    $archivePath = Join-Path $temporaryRoot $assetName
    $extractedPath = Join-Path $temporaryRoot "extracted"

    try {
        New-Item -ItemType Directory -Path $extractedPath -Force | Out-Null

        Write-Host "Downloading $assetName..."
        $downloadUri = [string]$assets[0].browser_download_url
        & curl.exe --fail --location --output $archivePath $downloadUri
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to download $assetName (curl exit code $LASTEXITCODE)."
        }
        if ((Get-Item $archivePath).Length -eq 0) {
            throw "The downloaded release archive is empty."
        }

        Assert-SafeArchive $archivePath
        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractedPath

        foreach ($requiredPath in @("MLFilter_x64.ax", "install_dependency.ps1", "bin")) {
            if (-not (Test-Path (Join-Path $extractedPath $requiredPath))) {
                throw "The release archive is missing '$requiredPath'"
            }
        }

        Write-Host "Downloading the GPU-specific dependency..."
        & (Join-Path $extractedPath "install_dependency.ps1")
        if ($LASTEXITCODE -ne 0) {
            throw "install_dependency.ps1 failed with exit code $LASTEXITCODE."
        }

        $stagedBuilderNames = @(
            Get-ChildItem (Join-Path $extractedPath "bin") "nvinfer_builder_resource_sm*_*.dll" |
                Select-Object -ExpandProperty Name
        )
        $obsoleteBuilderFiles = @(
            Get-ChildItem (Join-Path $PSScriptRoot "bin") "nvinfer_builder_resource_sm*_*.dll" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notin $stagedBuilderNames }
        )

        Assert-FilesReplaceable $extractedPath $PSScriptRoot
        foreach ($obsoleteBuilderFile in $obsoleteBuilderFiles) {
            Assert-FileReplaceable $obsoleteBuilderFile.FullName (Join-Path "bin" $obsoleteBuilderFile.Name)
        }

        Write-Host "Installing MLFilter $latestTag..."
        Copy-Item (Join-Path $extractedPath "*") -Destination $PSScriptRoot -Recurse -Force
        $obsoleteBuilderFiles | Remove-Item -Force
    } finally {
        Remove-Item $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host ""
    Write-Host "MLFilter was updated successfully to $latestTag"
    Exit-Updater 0
} catch {
    Write-Error $_
    Exit-Updater 1
}
