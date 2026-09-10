# NativeEnglishLM-200M

This branch provides an independent, English-only language-model path for
`hyperlightweight-ai`. The model starts from random weights and uses the
repository's own PyTorch model and ASCII BPE tokenizer. It does not import
pretrained weights, an external vocabulary, a model-specific chat template, or
Hugging Face model classes.

## Default architecture

| Item | Value |
|---|---:|
| Tokenizer | English ASCII byte-pair encoding |
| Target vocabulary | 8,192 tokens |
| Context length | 2,048 BPE tokens |
| Hidden size | 896 |
| Decoder blocks | 20 |
| Attention heads | 14 |
| FFN size | 3,584 |
| Position encoding | Learned absolute embeddings |
| Normalization and activation | Pre-LayerNorm and GELU |
| Input/output embeddings | Tied |
| Parameters at 8,192 tokens | 201,924,352 |

The tokenizer rejects non-ASCII text. This is intentional: this branch is an
English-only experiment, not a multilingual model with an English system
prompt.

## Install

Install a CUDA-compatible PyTorch build first when using a GPU, then run:

```powershell
python -m pip install -r .\requirements-native.txt
```

## Generate and validate seed data

```powershell
python .\generate_native_data.py
python .\prepare_native_sft.py
```

The included seed files only verify the pipeline. They are far too small to
train a useful 200M language model.

## Train the tokenizer

The tokenizer is trained from local English data and records source hashes in
its artifact:

```powershell
python .\train_native_tokenizer.py `
  --input .\data `
  --target-vocab-size 8192 `
  --output .\tokenizer\native_english_bpe.json
```

A small corpus may stop below the target when no pair meets the minimum
frequency. Production training should rebuild the tokenizer from the complete,
licensed English pretraining corpus before model training.

## Test

```powershell
python -m unittest -v .\test_native_200m.py
```

Run a CPU smoke test before a full CUDA job:

```powershell
python .\train_native_200m.py `
  --device cpu `
  --preset smoke `
  --seq-len 128 `
  --max-steps 2
```

## Train

```powershell
python .\train_native_200m.py `
  --device cuda `
  --data .\data `
  --tokenizer .\tokenizer\native_english_bpe.json `
  --seq-len 2048 `
  --batch-size 1 `
  --grad-accumulation 8 `
  --grad-checkpointing `
  --max-steps 1000 `
  --output-dir .\checkpoints_native_200m
```

One thousand steps are only an execution example. A useful model requires a
much larger token budget, held-out evaluation, and data-quality checks.

## Chat

```powershell
python .\chat_native_200m.py `
  --checkpoint .\checkpoints_native_200m\best.pt `
  --tokenizer .\tokenizer\native_english_bpe.json `
  --device cuda `
  --message "Explain the difference between a cache and a backup."
```

The checkpoint and tokenizer must have the same vocabulary size.

## Files

- `native_200m.py`: model definition and checkpoint I/O
- `native_tokenizer.py`: English ASCII BPE encoding and chat protocol
- `train_native_tokenizer.py`: tokenizer training and source manifest
- `train_native_200m.py`: causal pretraining loop
- `chat_native_200m.py`: interactive generation
- `generate_native_data.py`: self-authored English seed data
- `prepare_native_sft.py`: SFT validation, deduplication, and split
- `test_native_200m.py`: tokenizer, causality, loss, and generation tests
- `docs/native_200m_research.md`: design rationale and limitations
