param(
    [string]$DestinationDirectory = "$PSScriptRoot\checkpoints_qwen3_06b"
)

$ErrorActionPreference = "Stop"
$modelFileName = "Qwen3-0.6B-Q8_0.gguf"
$modelUrl = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf?download=true"
$licenseUrl = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/LICENSE?download=true"
$expectedSha256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"

New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null
$modelPath = Join-Path $DestinationDirectory $modelFileName
$partialPath = "$modelPath.partial"
$licensePath = Join-Path $DestinationDirectory "LICENSE-QWEN3-APACHE-2.0.txt"
$curlCommand = Get-Command "curl.exe" -ErrorAction SilentlyContinue
if ($null -eq $curlCommand) {
    throw "curl.exe is required to download files with redirect and resume support."
}

if (Test-Path -LiteralPath $modelPath) {
    $actualSha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -eq $expectedSha256) {
        Write-Output "Checkpoint already exists and passed SHA-256 verification: $modelPath"
    }
    else {
        throw "Existing checkpoint hash mismatch. Expected $expectedSha256 but got $actualSha256"
    }
}
else {
    Write-Output "Downloading the official Qwen3-0.6B Q8_0 checkpoint (about 639 MB)..."
    $curlArguments = @(
        "--fail",
        "--location",
        "--retry", "3",
        "--retry-delay", "2",
        "--continue-at", "-",
        "--output", $partialPath,
        $modelUrl
    )
    & $curlCommand.Source @curlArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint download failed with curl exit code $LASTEXITCODE"
    }
    $actualSha256 = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        Remove-Item -LiteralPath $partialPath
        throw "Downloaded checkpoint hash mismatch. Expected $expectedSha256 but got $actualSha256"
    }
    Move-Item -LiteralPath $partialPath -Destination $modelPath
    Write-Output "Checkpoint downloaded and verified: $modelPath"
}

if (-not (Test-Path -LiteralPath $licensePath)) {
    & $curlCommand.Source --fail --location --retry 3 --output $licensePath $licenseUrl
    if ($LASTEXITCODE -ne 0) {
        throw "License download failed with curl exit code $LASTEXITCODE"
    }
}

Write-Output "SHA-256: $expectedSha256"
