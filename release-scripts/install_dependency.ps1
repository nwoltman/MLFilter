$ErrorActionPreference = "Stop"
$releaseTag = "__RELEASE_TAG__"
$repository = "__REPOSITORY__"
$supportedArchitectures = @("__ARCHITECTURES__")
$architectureAssets = @{
    # __ASSET_MAP__
}
try {
    if (-not $releaseTag) {
        throw "This folder was made by a dry run and is not tied to a GitHub release. Run make_release.ps1 with a semantic version to create an installable release."
    }
    $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        $standardPath = Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        if (Test-Path $standardPath) { $nvidiaSmi = Get-Item $standardPath }
    }
    if (-not $nvidiaSmi) { throw "nvidia-smi.exe was not found. Install a current NVIDIA display driver." }

    $computeCapabilities = @(& $nvidiaSmi.Source --query-gpu=compute_cap --format=csv,noheader 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi could not query GPU compute capabilities: $($computeCapabilities -join ' ')" }
    $architectures = @($computeCapabilities |
        ForEach-Object { "sm" + (($_.ToString().Trim()) -replace '\.', '') } |
        Sort-Object -Unique)
    $unsupported = @($architectures | Where-Object { $_ -notin $supportedArchitectures })
    if ($unsupported.Count) {
        throw "Unsupported NVIDIA GPU architecture(s): $($unsupported -join ', '). Supported: $($supportedArchitectures -join ', ')."
    }

    $binDir = Join-Path $PSScriptRoot "bin"
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    foreach ($architecture in $architectures) {
        $assetName = $architectureAssets[$architecture]
        $destination = Join-Path $binDir $assetName
        if ((Test-Path $destination) -and (Get-Item $destination).Length -gt 0) {
            Write-Host "$assetName is already installed."
            continue
        }
        $uri = "https://github.com/$repository/releases/download/$releaseTag/$assetName"
        Write-Host "Downloading $assetName for $architecture..."
        $temporary = "$destination.download"
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $temporary
            Move-Item -Force $temporary $destination
        } finally {
            Remove-Item $temporary -Force -ErrorAction SilentlyContinue
        }
    }
} catch {
    Write-Error $_
    exit 1
}
