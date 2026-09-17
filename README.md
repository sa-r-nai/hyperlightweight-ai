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

## Lightning AI Studio quick start

Start a GPU Studio, upload or clone this repository, and open a terminal in
the repository root. Install the dependency and verify that CUDA is visible:

```bash
python -m pip install -r requirements-native.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run a two-step GPU pipeline check before starting a long job:

```bash
python train_native_200m.py \
  --device cuda \
  --preset smoke \
  --data ./data \
  --seq-len 128 \
  --batch-size 8 \
  --grad-accumulation 1 \
  --max-steps 2 \
  --eval-every 1 \
  --checkpoint-every 1 \
  --output-dir ./checkpoints_native_smoke
```

This checks data loading, next-token target alignment, CUDA forward/backward
execution, held-out validation, and checkpoint writing.

## Generate and validate seed data

```bash
python generate_native_data.py
python prepare_native_sft.py
```

The included seed files only verify the pipeline. They are far too small to
train a useful 200M language model.

## Train the tokenizer

The tokenizer is trained from local English data and records source hashes in
its artifact:

```bash
python train_native_tokenizer.py \
  --input ./data \
  --target-vocab-size 8192 \
  --output ./tokenizer/native_english_bpe.json
```

A small corpus may stop below the target when no pair meets the minimum
frequency. Production training should rebuild the tokenizer from the complete,
licensed English pretraining corpus before model training.

The checked-in tokenizer has 1,330 tokens because the included seed corpus is
small; 8,192 is the production target rather than the size of this fixture.

## Test

```bash
python -m unittest -v test_native_200m.py
```

Run a CPU smoke test before a full CUDA job:

```bash
python train_native_200m.py \
  --device cpu \
  --preset smoke \
  --seq-len 128 \
  --max-steps 2
```

## Train

```bash
python train_native_200m.py \
  --device cuda \
  --data /path/to/large-licensed-english-corpus \
  --tokenizer ./tokenizer/native_english_bpe.json \
  --require-real-data \
  --seq-len 2048 \
  --batch-size 1 \
  --grad-accumulation 8 \
  --grad-checkpointing \
  --max-steps 1000 \
  --eval-every 250 \
  --checkpoint-every 250 \
  --output-dir ./checkpoints_native_200m
```

One thousand steps are only an execution example. A useful model requires a
much larger token budget, held-out evaluation, and data-quality checks.
When `--validation-data` is omitted, the trainer makes a deterministic
document-level split using `--validation-ratio` (default 0.02). `best.pt` is
selected by held-out validation loss; `last.pt` records the latest checkpoint.

## Chat

```bash
python chat_native_200m.py \
  --checkpoint ./checkpoints_native_200m/best.pt \
  --tokenizer ./tokenizer/native_english_bpe.json \
  --device cuda \
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
