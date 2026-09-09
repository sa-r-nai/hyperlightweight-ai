"""NativeByteLM-500M: a from-scratch decoder-only language model.

This file owns the model definition used by the repository's new training and
inference paths.  It does not import Hugging Face Transformers, load external
weights, or depend on a model-specific tokenizer.

Architecture summary:
    token embedding + learned position embedding
    24 pre-LayerNorm decoder blocks
    full multi-head causal self-attention
    two-layer GELU feed-forward network
    tied language-model head

The default configuration contains approximately 498.5M trainable parameters
with the repository's 264-token UTF-8 byte vocabulary and a 2048-token context.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from native_tokenizer import NativeTokenizer


@dataclass
class Native500MConfig:
    """Configuration for the default model and small smoke-test variants."""

    vocab_size: int = NativeTokenizer.vocab_size
    max_seq_len: int = 2048
    d_model: int = 1280
    n_layers: int = 24
    n_heads: int = 20
    d_ff: int = 5504
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_bias: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size는 특수 토큰 수보다 커야 합니다.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len은 양수여야 합니다.")
        if self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("d_model과 d_ff는 양수여야 합니다.")
        if self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("n_layers와 n_heads는 양수여야 합니다.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model은 n_heads로 나누어져야 합니다.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def with_sequence_length(self, seq_len: int) -> "Native500MConfig":
        return replace(self, max_seq_len=seq_len)


def estimate_parameter_count(config: Native500MConfig) -> int:
    """Return the exact count for the bias-free/tied default parameterization."""

    embedding = config.vocab_size * config.d_model
    positions = config.max_seq_len * config.d_model
    attention = 4 * config.d_model * config.d_model
    feed_forward = 2 * config.d_model * config.d_ff
    linear_biases = 0
    if config.use_bias:
        linear_biases = 4 * config.d_model + config.d_ff + config.d_model

    # Two LayerNorms per block and one final LayerNorm.  Each has weight and
    # bias, even when the projection layers themselves are bias-free.
    layer_norms = (config.n_layers * 2 + 1) * 2 * config.d_model
    per_block = attention + feed_forward + linear_biases + 0
    total = embedding + positions + config.n_layers * per_block + layer_norms
    if not config.tie_embeddings:
        total += config.vocab_size * config.d_model
    return total


SMOKE_CONFIG = Native500MConfig(
    vocab_size=NativeTokenizer.vocab_size,
    max_seq_len=128,
    d_model=96,
    n_layers=3,
    n_heads=4,
    d_ff=384,
    dropout=0.0,
)


def _causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    """Use PyTorch's fused SDPA when present, with a compatible fallback."""

    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=True,
        )

    scale = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    sequence_length = scores.size(-1)
    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=dropout_p > 0.0)
    return torch.matmul(weights, value)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Native500MConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout
        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=config.use_bias,
        )
        self.output = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.use_bias,
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        attended = _causal_attention(
            query,
            key,
            value,
            self.dropout_p if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            hidden_size,
        )
        return self.residual_dropout(self.output(attended))


class FeedForward(nn.Module):
    def __init__(self, config: Native500MConfig) -> None:
        super().__init__()
        self.input = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.output = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.input(hidden_states), approximate="tanh")
        hidden_states = self.output(hidden_states)
        return self.dropout(hidden_states)


class DecoderBlock(nn.Module):
    def __init__(self, config: Native500MConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            config.d_model,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.feed_forward_norm = nn.LayerNorm(
            config.d_model,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.attention = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states


class NativeCausalLM(nn.Module):
    """Decoder-only causal language model trained from random initialization."""

    def __init__(self, config: Native500MConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(
            config.d_model,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )
        self._initialize_weights()
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids는 [batch, sequence] 형태여야 합니다.")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"입력 길이 {input_ids.size(1)}가 최대 길이 "
                f"{self.config.max_seq_len}를 초과했습니다."
            )
        if input_ids.numel() and int(input_ids.max()) >= self.config.vocab_size:
            raise ValueError("input_ids에 모델 어휘 크기를 벗어난 토큰이 있습니다.")

        positions = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        ).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = activation_checkpoint(
                    block,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states)

        logits = self.lm_head(self.final_norm(hidden_states))
        if labels is None:
            return logits
        if labels.shape != input_ids.shape:
            raise ValueError("labels는 input_ids와 같은 형태여야 합니다.")
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.05,
        eos_token_id: int = NativeTokenizer.eos_token_id,
    ) -> torch.Tensor:
        """Generate tokens with the same explicit sampling stages as training."""

        if input_ids.dim() != 2:
            raise ValueError("input_ids는 [batch, sequence] 형태여야 합니다.")
        if temperature < 0.0:
            raise ValueError("temperature는 0 이상이어야 합니다.")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p는 0보다 크고 1 이하여야 합니다.")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty는 0보다 커야 합니다.")

        self.eval()
        generated = input_ids
        finished = torch.zeros(
            generated.size(0),
            dtype=torch.bool,
            device=generated.device,
        )

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            logits = self(context)[:, -1, :]

            if repetition_penalty != 1.0:
                for batch_index in range(generated.size(0)):
                    seen_tokens = generated[batch_index].unique()
                    logits[batch_index, seen_tokens] = torch.where(
                        logits[batch_index, seen_tokens] < 0,
                        logits[batch_index, seen_tokens] * repetition_penalty,
                        logits[batch_index, seen_tokens] / repetition_penalty,
                    )

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                        dim=-1,
                    )
                    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
                    cumulative = sorted_probabilities.cumsum(dim=-1)
                    remove = cumulative - sorted_probabilities > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token), dim=1)
            finished = finished | next_token.squeeze(-1).eq(eos_token_id)
            if bool(finished.all()):
                break

        return generated


def config_from_checkpoint(checkpoint: dict[str, Any]) -> Native500MConfig:
    config_data = checkpoint.get("config") or checkpoint.get("model_config")
    if not isinstance(config_data, dict):
        raise ValueError("체크포인트에 모델 설정이 없습니다.")
    return Native500MConfig(**config_data)


def save_checkpoint(
    path: str | Path,
    model: NativeCausalLM,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    best_loss: Optional[float] = None,
) -> None:
    """Save a self-describing training checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "nativebytelm-checkpoint-v1",
        "step": step,
        "best_loss": best_loss,
        "config": asdict(model.config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[NativeCausalLM, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("지원하지 않는 체크포인트 형식입니다.")
    config = config_from_checkpoint(checkpoint)
    model = NativeCausalLM(config)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def write_config(path: str | Path, config: Native500MConfig) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "format": "nativebytelm-config-v1",
                "config": asdict(config),
                "estimated_parameter_count": estimate_parameter_count(config),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "Native500MConfig",
    "NativeCausalLM",
    "SMOKE_CONFIG",
    "config_from_checkpoint",
    "estimate_parameter_count",
    "load_checkpoint",
    "save_checkpoint",
    "write_config",
]
