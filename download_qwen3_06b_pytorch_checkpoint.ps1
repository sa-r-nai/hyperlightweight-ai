param(
    [string]$DestinationDirectory = "$PSScriptRoot\checkpoints_qwen3_06b_pytorch"
)

$ErrorActionPreference = "Stop"
$repositoryBase = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main"
$modelSha256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
$files = @(
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
)

$curlCommand = Get-Command "curl.exe" -ErrorAction SilentlyContinue
if ($null -eq $curlCommand) {
    throw "curl.exe is required to download files with redirect and resume support."
}
$pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) {
    $pythonCommand = Get-Command "py.exe" -ErrorAction SilentlyContinue
}
if ($null -eq $pythonCommand) {
    throw "Python is required to validate the downloaded checkpoint metadata."
}

New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null

foreach ($fileName in $files) {
    $destinationPath = Join-Path $DestinationDirectory $fileName
    $partialPath = "$destinationPath.partial"

    if ($fileName -eq "model.safetensors" -and (Test-Path -LiteralPath $destinationPath)) {
        $actualSha256 = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $modelSha256) {
            throw "Existing model hash mismatch. Expected $modelSha256 but got $actualSha256"
        }
        Write-Output "Verified existing model.safetensors"
        continue
    }
    if ($fileName -ne "model.safetensors" -and (Test-Path -LiteralPath $destinationPath)) {
        Write-Output "Keeping existing $fileName"
        continue
    }

    $encodedName = [System.Uri]::EscapeDataString($fileName).Replace("%2F", "/")
    $url = "$repositoryBase/$encodedName`?download=true"
    Write-Output "Downloading $fileName..."
    $curlArguments = @(
        "--fail",
        "--location",
        "--retry", "3",
        "--retry-delay", "2",
        "--continue-at", "-",
        "--output", $partialPath,
        $url
    )
    & $curlCommand.Source @curlArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$fileName download failed with curl exit code $LASTEXITCODE"
    }

    if ($fileName -eq "model.safetensors") {
        $actualSha256 = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $modelSha256) {
            Remove-Item -LiteralPath $partialPath
            throw "Downloaded model hash mismatch. Expected $modelSha256 but got $actualSha256"
        }
    }
    Move-Item -LiteralPath $partialPath -Destination $destinationPath
}

$requiredJsonFiles = @("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")
foreach ($jsonFile in $requiredJsonFiles) {
    $jsonPath = Join-Path $DestinationDirectory $jsonFile
    & $pythonCommand.Source -c "import json,pathlib,sys; json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))" $jsonPath
    if ($LASTEXITCODE -ne 0) {
        throw "Invalid JSON file: $jsonPath"
    }
}

$config = Get-Content -LiteralPath (Join-Path $DestinationDirectory "config.json") -Raw -Encoding UTF8 | ConvertFrom-Json
if ($config.model_type -ne "qwen3" -or $config.architectures -notcontains "Qwen3ForCausalLM") {
    throw "Downloaded config is not the expected Qwen3ForCausalLM architecture."
}

Write-Output "Python/PyTorch checkpoint is ready: $DestinationDirectory"
Write-Output "model.safetensors SHA-256: $modelSha256"
