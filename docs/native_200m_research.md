# NativeEnglishLM-200M design notes

## Scope

This path implements a decoder-only English language model from random
initialization. It does not copy or convert external checkpoints, tokenizer
artifacts, chat templates, or model classes. Training data remains a separate
input asset whose provenance and license must be recorded.

The repository seed is a pipeline fixture, not a useful pretraining corpus.
Producing a capable model still requires large-scale licensed English data,
substantial compute, validation splits, and downstream evaluation.

## Architecture

| Component | Calculation | Parameters |
|---|---:|---:|
| Token embeddings | 8,192 x 896 | 7,340,032 |
| Position embeddings | 2,048 x 896 | 1,835,008 |
| Attention per block | 4 x 896 x 896 | 3,211,264 |
| FFN per block | 2 x 896 x 3,584 | 6,422,528 |
| Decoder blocks | 20 blocks | 192,675,840 |
| LayerNorm parameters | 41 norms | 73,472 |
| Total | tied output head | **201,924,352** |

The model uses full multi-head causal attention, learned absolute positions,
Pre-LayerNorm residual blocks, GELU feed-forward layers, and tied input/output
embeddings. These choices keep the implementation small and auditable. They do
not compensate for insufficient data or training tokens.

## English-only BPE

The tokenizer begins with eight protocol tokens and 128 ASCII byte symbols.
It learns byte-pair merges from caller-provided English files. Each learned
token therefore remains exactly decodable to ASCII without an unknown-token
fallback.

Non-ASCII text is rejected during tokenizer training, dataset loading, and
interactive input. This makes the language boundary enforceable instead of
relying only on an English system prompt.

The default target is 8,192 tokens. With a small corpus, training stops early
when no pair reaches the minimum frequency. The model configuration uses the
actual tokenizer vocabulary size at training time, so unused embedding rows
are not allocated.

Compared with the previous UTF-8 byte tokenizer, English BPE reduces sequence
length and makes the 2,048-token context materially more useful for English
prose and code. The tokenizer artifact records its input file hashes, target
size, and minimum pair frequency.

## Training

Training uses next-token cross-entropy over packed documents, AdamW with
betas=(0.9, 0.95), linear warmup, and cosine decay. CUDA AMP and gradient
checkpointing are available. CPU mode exists for smoke tests but is not a
practical way to pretrain the full model.

At 201,924,352 parameters, raw model weights require approximately:

| Representation | Approximate weight storage |
|---|---:|
| float32 | 770 MiB |
| float16 or bfloat16 | 385 MiB |
| AdamW moments in float32 | 1.50 GiB additional |

Gradients, optimizer copies, activations, allocator overhead, and attention
workspaces increase real training memory beyond these figures.

## Data policy

Every production dataset should record:

- origin or generation method;
- license and permitted use;
- language and document type;
- exact preprocessing rules and content hashes;
- duplicate and near-duplicate filtering;
- train/validation separation;
- human review policy for synthetic records.

Malformed encoding, personal secrets, unexplained synthetic claims, and
non-English text should be removed before tokenizer or model training.

## Verification

Automated tests cover:

1. BPE round-trip behavior and compression;
2. rejection of non-ASCII input;
3. the independent role-token chat protocol;
4. the expected 190M-215M parameter range;
5. causal masking and finite loss;
6. deterministic generation shape;
7. output-token filtering.

Production runs must also track validation loss, perplexity, effective tokens,
gradient norms, overflow events, checkpoint hashes, tokenizer hash, hardware,
PyTorch/CUDA versions, and random seeds.

## Current limitations

- The included corpus is much too small for useful pretraining.
- The current loader materializes tokenized data in memory.
- Generation has no KV cache and recomputes the context for every token.
- The training loop does not yet apply an assistant-only SFT loss mask.
- ASCII-only operation excludes accented English names, typographic
  punctuation, mathematical Unicode symbols, and other languages.

## References

1. Vaswani et al., "Attention Is All You Need," 2017. [arXiv](https://arxiv.org/abs/1706.03762)
2. Hendrycks and Gimpel, "Gaussian Error Linear Units," 2016. [arXiv](https://arxiv.org/abs/1606.08415)
3. Loshchilov and Hutter, "Decoupled Weight Decay Regularization," 2017. [arXiv](https://arxiv.org/abs/1711.05101)
4. PyTorch, [scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
5. PyTorch, [automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
6. PyTorch, [activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
