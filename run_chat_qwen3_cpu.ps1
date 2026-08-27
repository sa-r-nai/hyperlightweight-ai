param(
    [string]$ModelPath = "$PSScriptRoot\checkpoints_qwen3_06b\Qwen3-0.6B-Q8_0.gguf",
    [int]$Threads = 6,
    [int]$ContextSize = 4096
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Checkpoint not found: $ModelPath. Run download_qwen3_06b_checkpoint.ps1 first."
}

$localLlamaCli = Join-Path $PSScriptRoot "tools\llama-cpp\llama-cli.exe"
if (Test-Path -LiteralPath $localLlamaCli) {
    $llamaCli = $localLlamaCli
}
else {
    $llamaCommand = Get-Command "llama-cli.exe" -ErrorAction SilentlyContinue
    if ($null -eq $llamaCommand) {
        throw "llama-cli.exe was not found. Install the official CPU build with: winget install llama.cpp"
    }
    $llamaCli = $llamaCommand.Source
}

$systemPromptPath = Join-Path $PSScriptRoot "qwen3_system_prompt.txt"
$arguments = @(
    "-m", $ModelPath,
    "-cnv",
    "--jinja",
    "-sysf", $systemPromptPath,
    "-rea", "off",
    "-ngl", "0",
    "-t", $Threads,
    "-c", $ContextSize,
    "-n", "512",
    "--temp", "0.7",
    "--top-k", "20",
    "--top-p", "0.8",
    "--min-p", "0",
    "--presence-penalty", "1.5"
)

& $llamaCli @arguments
exit $LASTEXITCODE
