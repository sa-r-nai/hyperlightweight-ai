# Optional llama.cpp/GGUF checkpoint

This is an optional alternative runtime. The project's primary path is now the
Python/PyTorch implementation documented in `QWEN3_06B_PYTHON.md`.

This project uses the official `Qwen/Qwen3-0.6B-GGUF` Q8_0 checkpoint for
usable CPU-only conversation. It is separate from the custom `TinyCausalLM`
architecture in `ultralight_llm.py`; their architectures and tokenizers are
not compatible.

## Provenance

- Model: `Qwen/Qwen3-0.6B`
- Distribution: `Qwen/Qwen3-0.6B-GGUF`
- File: `Qwen3-0.6B-Q8_0.gguf`
- Quantization: Q8_0
- Parameters: 0.6B
- Expected size: about 639 MB
- SHA-256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- License: Apache License 2.0
- Source: https://huggingface.co/Qwen/Qwen3-0.6B-GGUF

The checkpoint was pretrained and post-trained by the Qwen team. The local
file `data/korean_conversation_sft_seed.jsonl` is self-authored supplemental
material for future fine-tuning or evaluation and was **not** used to train
the official checkpoint.

An optional, larger Korean instruction corpus can be downloaded separately:

- Dataset: `CarrotAI/ko-instruction-dataset`
- Declared license: Apache-2.0
- Source: https://huggingface.co/datasets/CarrotAI/ko-instruction-dataset
- SHA-256: `d9c5f5277cd1ee15d847f8ad53cb762a99fbde2d767be25f2c24cf865d62a5e6`

```powershell
powershell -ExecutionPolicy Bypass -File .\download_korean_sft_dataset.ps1
python .\prepare_qwen3_sft_data.py
```

The preparation script combines the externally sourced instruction records
with the self-authored seed, removes exact duplicate conversations, performs a
deterministic 98/2 train-validation split, and writes a provenance manifest.

Prepared artifacts in this workspace:

- `sft_data/qwen3_korean_sft_train.jsonl`: 6,787 conversations
- `sft_data/qwen3_korean_sft_validation.jsonl`: 138 conversations
- `sft_data/MANIFEST.json`: source, license, count, and SHA-256 metadata

These files are ready for a future SFT job, but the project does not claim that
this small corpus can reproduce Qwen's pretraining or that it has already been
applied to the official checkpoint.

## Prepare

Download and verify the checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_qwen3_06b_checkpoint.ps1
```

Install the official CPU build of llama.cpp if `llama-cli.exe` is not already
available:

```powershell
winget install llama.cpp
```

## Chat on CPU

```powershell
powershell -ExecutionPolicy Bypass -File .\run_chat_qwen3_cpu.ps1
```

The runner disables GPU offload and Qwen thinking mode, uses the model's
embedded Jinja chat template, and applies the non-thinking sampling settings
recommended in the official model card.
