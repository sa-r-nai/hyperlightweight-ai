# Python/PyTorch Qwen3-0.6B

This is the primary runtime path for the project. It loads the standard
Qwen3-0.6B `safetensors` checkpoint directly in Python through PyTorch and
Hugging Face Transformers. It does not use llama.cpp or a GGUF model. CUDA is
the default and required execution mode; CPU is an explicit option.

## Files

- Checkpoint directory: `checkpoints_qwen3_06b_pytorch/`
- Python runner: `terminal_chat_qwen3_python.py`
- Python dependencies: `requirements-qwen3.txt`
- Training data: `sft_data/qwen3_korean_sft_train.jsonl`
- Validation data: `sft_data/qwen3_korean_sft_validation.jsonl`

The included SFT data has not been applied to the official checkpoint. The
checkpoint's conversational ability comes from Qwen's own pretraining and
post-training.

## Prepare the checkpoint

```powershell
powershell -ExecutionPolicy Bypass -File .\download_qwen3_06b_pytorch_checkpoint.ps1
```

The downloader retrieves the official tokenizer, configuration, license, and
`model.safetensors`, then verifies the model file with SHA-256:

`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`

## Install Python dependencies

Install a CUDA-enabled PyTorch build first, then install the remaining Python
dependencies:

```powershell
python -m pip install -r .\requirements-qwen3.txt
```

## Run on GPU (default)

```powershell
python .\terminal_chat_qwen3_python.py
```

If CUDA is unavailable, the default command raises an error and exits. It does
not fall back to CPU.

Single message:

```powershell
python .\terminal_chat_qwen3_python.py --message "안녕하세요. 무엇을 할 수 있어?"
```

The GPU runner uses bfloat16 when supported and float16 otherwise.

## Run on CPU (explicit option)

```powershell
python .\terminal_chat_qwen3_python.py --device cpu
```

CPU mode loads float32 weights and uses `--threads` to control the PyTorch
thread count. The runner never selects CPU automatically.

Both modes use the Qwen chat template with thinking disabled, retain a bounded
conversation history, and perform autoregressive token generation through
`model.generate()`.

## Optional GGUF path

The earlier `QWEN3_06B_CHECKPOINT.md` and `run_chat_qwen3_cpu.ps1` files are an
optional llama.cpp path. They are not required for the Python runtime.
