param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$Message,
    [int]$MaxNewTokens = 256
)

$arguments = @(
    "chat_native_200m.py",
    "--checkpoint", $Checkpoint,
    "--device", $Device,
    "--tokenizer", "tokenizer/native_english_bpe.json",
    "--max-new-tokens", $MaxNewTokens
)
if ($Message) {
    $arguments += @("--message", $Message)
}
python @arguments
