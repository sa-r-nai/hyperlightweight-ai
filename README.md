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

For the full public-data pipeline, this is the single command to run:

```bash
python -m pip install -r requirements-native.txt && python run_lightning_pipeline.py
```

It streams and normalizes 1,000,000 FineWeb-Edu training documents and 400,000
Smol-SmolTalk conversations by default, creates held-out validation sets,
builds disk-backed token files, pretrains for four corpus passes, performs two
chat-SFT passes with assistant-only loss, and runs the basic conversation gate.
The public data itself is not committed to Git. Expect a long, storage- and
compute-intensive job; this is a from-scratch 200M model rather than a quick
fine-tune of pretrained weights.

If data preparation finished but the Studio stopped during training, reuse the
prepared artifacts and resume from `last.pt` automatically:

```bash
python run_lightning_pipeline.py --reuse-prepared
```

The public sources are pinned for reproducibility:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  configuration `sample-10BT`, ODC-By-1.0 and subject to Common Crawl terms
- [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
  the Apache-2.0 SFT subset designed for models below one billion parameters

`public_data/`, `tokenized_data/`, and checkpoints are ignored by Git. The
generated manifests retain dataset revisions, licenses, source/output hashes,
record counts, token counts, and the tokenizer fingerprint.

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

The local generator creates 5,000 deterministic, self-authored pretraining
records and 6,000 chat conversations. The chat set contains 2,094 multi-turn
examples covering everyday planning, clarification, debugging, study, writing,
basic reasoning, greetings, corrections, and safe uncertainty.
`prepare_native_sft.py` produces a deterministic 5,880/120 train-validation
split.

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

The checked-in tokenizer has 4,362 tokens after training on all included data.
The pretraining corpus contains about 353,000 BPE tokens and the chat corpus
contains about 829,000 BPE tokens, including about 348,000 supervised assistant
tokens. 8,192 remains the production vocabulary target.

## Test

```bash
python -m unittest -v
```

Run a CPU smoke test before a full CUDA job:

```bash
python train_native_200m.py \
  --device cpu \
  --preset smoke \
  --seq-len 128 \
  --max-steps 2
```

## Manual large-corpus training

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

## Chat SFT

After pretraining, start a fresh optimizer and fine-tune only on assistant
responses. The checkpoint keeps the pretraining model architecture and context
length, while `--seq-len` may be shorter for the SFT batches:

```bash
python train_native_200m.py \
  --device cuda \
  --data ./sft_data/native_sft_train.jsonl \
  --validation-data ./sft_data/native_sft_validation.jsonl \
  --tokenizer ./tokenizer/native_english_bpe.json \
  --init-from ./checkpoints_native_200m/best.pt \
  --assistant-only-loss \
  --seq-len 512 \
  --batch-size 2 \
  --grad-accumulation 8 \
  --grad-checkpointing \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --warmup-steps 50 \
  --max-steps 1000 \
  --eval-every 100 \
  --checkpoint-every 100 \
  --output-dir ./checkpoints_native_200m_chat
```

`--assistant-only-loss` masks system and user targets while retaining their
tokens as context. `--init-from` loads only model weights, so SFT starts with a
new optimizer and learning-rate schedule. The included synthetic data is useful
for pipeline development but is still too small to guarantee natural, broad
conversation from a randomly initialized 200M model.

After SFT, run the deterministic basic-conversation gate. It checks greetings,
clarification, explanations, planning, writing, arithmetic, Python, uncertainty,
non-empty output, and repetition. A passing gate is a basic sanity check rather
than proof of broad conversational quality:

```bash
python evaluate_native_chat.py \
  --checkpoint ./checkpoints_native_200m_chat/best.pt \
  --tokenizer ./tokenizer/native_english_bpe.json \
  --device cuda
```

## Chat

```bash
python chat_native_200m.py \
  --checkpoint ./checkpoints_native_200m_chat/best.pt \
  --tokenizer ./tokenizer/native_english_bpe.json \
  --device cuda \
  --message "Explain the difference between a cache and a backup."
```

The checkpoint and tokenizer must have the same vocabulary size.

## Files

- `native_200m.py`: model definition and checkpoint I/O
- `native_tokenizer.py`: English ASCII BPE encoding and chat protocol
- `train_native_tokenizer.py`: tokenizer training and source manifest
- `prepare_public_data.py`: pinned FineWeb-Edu and Smol-SmolTalk streaming/normalization
- `tokenize_native_data.py`: memory-mapped uint16 token and assistant-mask builder
- `native_data.py`: shared text and chat record readers
- `train_native_200m.py`: causal pretraining loop
- `chat_native_200m.py`: interactive generation
- `evaluate_native_chat.py`: deterministic basic-conversation quality gate
- `generate_native_data.py`: self-authored English seed data
- `generate_native_corpus.py`: deterministic 5,000-record corpus builder
- `generate_native_chat_data.py`: deterministic single- and multi-turn chat builder
- `prepare_native_sft.py`: SFT validation, deduplication, and split
- `run_lightning_pipeline.py`: one-command public-data pretraining, SFT, and evaluation
- `test_native_200m.py`: tokenizer, causality, loss, and generation tests
- `test_native_data.py`: corpus determinism, uniqueness, and metadata tests
- `docs/native_200m_research.md`: design rationale and limitations
