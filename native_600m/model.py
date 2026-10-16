"""Independent decoder-only language model with randomly initialized weights.

The default configuration contains exactly 596,788,480 unique parameters.
This module uses PyTorch building blocks only; it does not load another model's
configuration, tokenizer, or weights. Labels supplied to ``forward`` must
already be aligned with their next-token prediction positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings, including small configurations for inspection."""

    vocab_size: int = 32_000
    max_seq_len: int = 2_048
    d_model: int = 1_280
    n_layers: int = 32
    n_heads: int = 20
    n_kv_heads: int = 5
    d_ff: int = 3_456
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in ("vocab_size", "max_seq_len", "d_model", "n_layers",
                     "n_heads", "n_kv_heads", "d_ff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if self.head_dim % 2:
            raise ValueError("The attention head dimension must be even for RoPE.")
        for name in ("dropout", "rope_theta", "norm_eps", "init_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be at least zero and less than one.")
        if self.rope_theta <= 1:
            raise ValueError("rope_theta must be greater than one.")
        if self.norm_eps <= 0 or self.init_std <= 0:
            raise ValueError("norm_eps and init_std must be positive.")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a boolean.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def parameter_count(config: ModelConfig) -> int:
    """Calculate the exact unique-parameter count without allocating tensors."""

    kv_width = config.n_kv_heads * config.head_dim
    embeddings = config.vocab_size * config.d_model
    attention = 2 * config.d_model**2 + 2 * config.d_model * kv_width
    feed_forward = 3 * config.d_model * config.d_ff
    norms = 2 * config.d_model
    total = embeddings + config.n_layers * (attention + feed_forward + norms)
    total += config.d_model  # Final RMSNorm.
    if not config.tie_embeddings:
        total += embeddings
    return total


class RMSNorm(nn.Module):
    """Root-mean-square normalization with float32 accumulation."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        source = hidden.float()
        normalized = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(hidden.dtype)


class RotaryEmbedding(nn.Module):
    """Apply rotary positions to adjacent pairs of query/key channels."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(query.size(-2), device=query.device, dtype=torch.float32)
        # Build angles in float32 even when the surrounding matmuls use autocast.
        with torch.autocast(device_type=query.device.type, enabled=False):
            inv_freq = self.theta ** (-torch.arange(0, self.head_dim, 2, device=query.device,
                                                    dtype=torch.float32) / self.head_dim)
            angles = torch.outer(positions, inv_freq)
            cosine = angles.cos()[None, None, :, :]
            sine = angles.sin()[None, None, :, :]

        def rotate(values: torch.Tensor) -> torch.Tensor:
            even, odd = values.float()[..., 0::2], values.float()[..., 1::2]
            paired = torch.stack((even * cosine - odd * sine,
                                  even * sine + odd * cosine), dim=-1)
            return paired.flatten(-2).to(values.dtype)

        return rotate(query), rotate(key)


class GroupedQueryAttention(nn.Module):
    """Causal attention with independently projected shared key/value heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        kv_width = config.n_kv_heads * config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_width, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        query, key = self.rope(query, key)
        repeats = self.n_heads // self.n_kv_heads
        if repeats != 1:
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.residual_dropout(self.attention(self.attention_norm(hidden)))
        return hidden + self.residual_dropout(self.ffn(self.ffn_norm(hidden)))


class NativeLM(nn.Module):
    """Native Korean model; every parameter starts from a fresh initialization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.apply(self._initialize_weights)
        # Scale the two residual output paths to control initial residual growth.
        residual_std = config.init_std / math.sqrt(2 * config.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attention.out_proj.weight, std=residual_std)
            nn.init.normal_(layer.ffn.down_proj.weight, std=residual_std)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def enable_gradient_checkpointing(self) -> None:
        """Recompute decoder activations when the training caller enables grads."""

        self.gradient_checkpointing = True

    def _validate_input(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.size(0) == 0:
            raise ValueError("input_ids must have nonempty shape [batch, sequence].")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain int32 or int64 token IDs.")
        if not 1 <= input_ids.size(1) <= self.config.max_seq_len:
            raise ValueError(f"Sequence length must be between 1 and {self.config.max_seq_len}.")

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
        *, loss_reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and optional loss on already next-token-aligned labels.

        ``-100`` labels are ignored. ``mean`` divides by the number of supervised
        tokens; ``sum`` returns their total loss for token-weighted accumulation.
        An entirely ignored label tensor produces a finite, differentiable zero.
        """

        self._validate_input(input_ids)
        if loss_reduction not in ("mean", "sum"):
            raise ValueError("loss_reduction must be 'mean' or 'sum'.")
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.int64:
                raise ValueError("labels must be int64 with the same [batch, sequence] shape as input_ids.")
            if labels.device != input_ids.device:
                raise ValueError("labels and input_ids must be on the same device.")
        hidden = self.token_embedding(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1),
                ignore_index=-100, reduction="sum",
            )
            if loss_reduction == "mean":
                loss = loss / labels.ne(-100).sum().clamp_min(1)
        return logits, loss

    @torch.inference_mode()
    def generate(
        self, input_ids: torch.Tensor, *, max_new_tokens: int = 256,
        temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
        repetition_penalty: float = 1.05, eos_token_id: int | Sequence[int] | None = 2,
        forbidden_token_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Append sampled tokens until EOS or the configured context limit.

        Temperature zero selects greedy decoding. Generation recalculates causal
        attention on the full prefix and does not currently use a KV cache.
        EOS may be one ID or a sequence of stop IDs. Finished batch rows receive
        the first EOS ID while other rows finish. Model mode is
        restored after generation; all generation forwards run in evaluation mode.
        """

        self._validate_input(input_ids)
        for name, value in (("max_new_tokens", max_new_tokens), ("top_k", top_k)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name, value in (("temperature", temperature), ("top_p", top_p),
                            ("repetition_penalty", repetition_penalty)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if temperature < 0 or not 0 < top_p <= 1 or repetition_penalty <= 0:
            raise ValueError("temperature must be nonnegative, top_p in (0, 1], and repetition_penalty positive.")
        forbidden = list(forbidden_token_ids) if forbidden_token_ids is not None else []
        if eos_token_id is None:
            eos_ids = []
        elif isinstance(eos_token_id, int):
            eos_ids = [eos_token_id]
        else:
            try:
                eos_ids = list(eos_token_id)
            except TypeError as error:
                raise ValueError("eos_token_id must be an integer or a sequence of integers.") from error
        for token_id in forbidden + eos_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < self.config.vocab_size:
                raise ValueError("EOS and forbidden token IDs must be integers within the vocabulary.")
        if len(set(forbidden)) == self.config.vocab_size:
            raise ValueError("forbidden_token_ids cannot exclude the entire vocabulary.")
        sequence = input_ids
        remaining = min(max_new_tokens, self.config.max_seq_len - input_ids.size(1))
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        eos_tensor = torch.tensor(eos_ids, dtype=input_ids.dtype, device=input_ids.device)
        was_training = self.training
        self.eval()
        try:
            for _ in range(remaining):
                logits, _ = self(sequence)
                scores = logits[:, -1, :].float()
                if repetition_penalty != 1:
                    previous = scores.gather(1, sequence.long())
                    penalized = torch.where(previous < 0, previous * repetition_penalty,
                                            previous / repetition_penalty)
                    scores.scatter_(1, sequence.long(), penalized)
                if forbidden:
                    scores[:, forbidden] = -torch.inf
                if temperature == 0:
                    next_token = scores.argmax(dim=-1, keepdim=True)
                else:
                    scores = scores / temperature
                    if top_k:
                        threshold = scores.topk(min(top_k, self.config.vocab_size), dim=-1).values[:, -1:]
                        scores = scores.masked_fill(scores < threshold, -torch.inf)
                    if top_p < 1:
                        sorted_scores, sorted_indices = scores.sort(descending=True, dim=-1)
                        remove = sorted_scores.softmax(dim=-1).cumsum(dim=-1) > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        scores = scores.scatter(1, sorted_indices, sorted_scores.masked_fill(remove, -torch.inf))
                    next_token = torch.multinomial(scores.softmax(dim=-1), num_samples=1)
                if eos_ids:
                    next_token = torch.where(finished[:, None], eos_ids[0], next_token)
                    finished |= torch.isin(next_token.squeeze(-1), eos_tensor)
                sequence = torch.cat((sequence, next_token.to(sequence.dtype)), dim=1)
                if eos_ids and bool(finished.all()):
                    break
        finally:
            self.train(was_training)
        return sequence
