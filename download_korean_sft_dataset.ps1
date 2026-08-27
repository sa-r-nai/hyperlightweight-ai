param(
    [string]$DestinationDirectory = "$PSScriptRoot\sft_data\external\CarrotAI-ko-instruction-dataset"
)

$ErrorActionPreference = "Stop"
$dataFileName = "instruction_korean.json"
$dataUrl = "https://huggingface.co/datasets/CarrotAI/ko-instruction-dataset/resolve/main/instruction_korean.json?download=true"
$readmeUrl = "https://huggingface.co/datasets/CarrotAI/ko-instruction-dataset/resolve/main/README.md?download=true"
$expectedSha256 = "d9c5f5277cd1ee15d847f8ad53cb762a99fbde2d767be25f2c24cf865d62a5e6"

New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null
$dataPath = Join-Path $DestinationDirectory $dataFileName
$partialPath = "$dataPath.partial"
$readmePath = Join-Path $DestinationDirectory "UPSTREAM_README.md"
$curlCommand = Get-Command "curl.exe" -ErrorAction SilentlyContinue
if ($null -eq $curlCommand) {
    throw "curl.exe is required to download files with redirect and resume support."
}

if (Test-Path -LiteralPath $dataPath) {
    $actualSha256 = (Get-FileHash -LiteralPath $dataPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw "Existing dataset hash mismatch. Expected $expectedSha256 but got $actualSha256"
    }
    Write-Output "Dataset already exists and passed SHA-256 verification: $dataPath"
}
else {
    Write-Output "Downloading the Apache-2.0 Korean instruction dataset (about 26.4 MB)..."
    $curlArguments = @(
        "--fail",
        "--location",
        "--retry", "3",
        "--retry-delay", "2",
        "--continue-at", "-",
        "--output", $partialPath,
        $dataUrl
    )
    & $curlCommand.Source @curlArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset download failed with curl exit code $LASTEXITCODE"
    }
    $actualSha256 = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        Remove-Item -LiteralPath $partialPath
        throw "Downloaded dataset hash mismatch. Expected $expectedSha256 but got $actualSha256"
    }
    Move-Item -LiteralPath $partialPath -Destination $dataPath
    Write-Output "Dataset downloaded and verified: $dataPath"
}

if (-not (Test-Path -LiteralPath $readmePath)) {
    & $curlCommand.Source --fail --location --retry 3 --output $readmePath $readmeUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset README download failed with curl exit code $LASTEXITCODE"
    }
}

Write-Output "Declared license: Apache-2.0"
Write-Output "SHA-256: $expectedSha256"
