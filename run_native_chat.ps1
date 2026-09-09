param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$Message,
    [int]$MaxNewTokens = 256
)

$arguments = @(
    "chat_native_500m.py",
    "--checkpoint", $Checkpoint,
    "--device", $Device,
    "--max-new-tokens", $MaxNewTokens
)
if ($Message) {
    $arguments += @("--message", $Message)
}
python @arguments
