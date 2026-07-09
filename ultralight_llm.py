"""Ultra-light decoder-only language model skeleton.

The flow matches:
input -> preprocessing -> tokenization -> LM forward / next-token sampling
-> stop-condition check -> append token -> repeat -> response output.

This module is intentionally self-contained:
- byte-level tokenizer for multilingual text
- RMSNorm + RoPE + GQA attention + SwiGLU feed-forward
- configurable presets from 5M to 2B parameters
- autoregressive generation loop with top-k / top-p sampling

The architecture is ready for training or loading pretrained weights.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPreprocessor:
    """Normalize user input before tokenization."""

    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class ByteTokenizer:
    """Very small multilingual tokenizer based on UTF-8 bytes.

    Special tokens:
    0 = <pad>
    1 = <bos>
    2 = <eos>
    3 = <unk>
    4..259 = raw UTF-8 byte values 0..255
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    byte_offset = 4
    vocab_size = 260

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        byte_tokens = [self.byte_offset + b for b in text.encode("utf-8")]
        if add_bos:
            byte_tokens.insert(0, self.bos_token_id)
        if add_eos:
            byte_tokens.append(self.eos_token_id)
        return byte_tokens

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        byte_values = bytearray()
        for token_id in token_ids:
            if token_id < self.byte_offset:
                if skip_special_tokens:
                    continue
                continue
            byte_value = token_id - self.byte_offset
            if 0 <= byte_value <= 255:
                byte_values.append(byte_value)
        return byte_values.decode("utf-8", errors="replace")


@dataclass
class ModelConfig:
    vocab_size: int = ByteTokenizer.vocab_size
    max_seq_len: int = 2048
    d_model: int = 1280
    n_layers: int = 24
    n_heads: int = 20
    n_kv_heads: int = 5
    d_ff: int = 5120
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive.")


def estimate_parameter_count(config: ModelConfig) -> int:
    """Rough parameter count for a configuration."""

    head_dim = config.d_model // config.n_heads
    kv_dim = config.n_kv_heads * head_dim

    embed = config.vocab_size * config.d_model
    attn = config.d_model * (config.d_model + kv_dim + kv_dim + config.d_model)
    ffn = 3 * config.d_model * config.d_ff
    norms = (2 * config.n_layers + 1) * config.d_model

    tied_head = 0 if config.tie_embeddings else config.vocab_size * config.d_model
    per_layer = attn + ffn + 2 * config.d_model
    return embed + tied_head + config.n_layers * per_layer + norms


MODEL_PRESETS = {
    # Approx. 5.2M parameters
    "5m": ModelConfig(
        d_model=224,
        n_layers=7,
        n_heads=8,
        n_kv_heads=2,
        d_ff=896,
        max_seq_len=2048,
    ),
    # Approx. 45.8M parameters
    "50m": ModelConfig(
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        d_ff=2048,
        max_seq_len=2048,
    ),
    # Approx. 102.9M parameters
    "100m": ModelConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=3,
        d_ff=3072,
        max_seq_len=2048,
    ),
    # Approx. 0.57B parameters
    "600m": ModelConfig(
        d_model=1280,
        n_layers=24,
        n_heads=20,
        n_kv_heads=5,
        d_ff=5120,
        max_seq_len=2048,
    ),
    # Approx. 0.96B parameters
    "1b": ModelConfig(
        d_model=1536,
        n_layers=28,
        n_heads=24,
        n_kv_heads=6,
        d_ff=6144,
        max_seq_len=2048,
    ),
    # Approx. 1.95B parameters
    "2b": ModelConfig(
        d_model=2048,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        d_ff=8192,
        max_seq_len=2048,
    ),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary dimension must be even.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class GQAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seq_len, x.device, x.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )[None, None, :, :]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        return self.down_proj(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.attn = GQAAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
        )
        self.ffn = FeedForward(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyCausalLM(nn.Module):
    """Decoder-only transformer for lightweight autoregressive generation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        if config.tie_embeddings:
            if self.token_emb.weight.shape != self.lm_head.weight.shape:
                raise ValueError("Embedding and head shapes must match to tie weights.")
            self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq].")

        if input_ids.size(1) > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len :]
            if targets is not None:
                targets = targets[:, -self.config.max_seq_len :]

        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        if targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids.")

        if input_ids.size(1) < 2:
            raise ValueError("Need at least two tokens to compute causal loss.")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_targets.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        eos_token_id: int = ByteTokenizer.eos_token_id,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            logits = self(context)
            next_logits = logits[:, -1, :]
            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            if torch.all(next_token == eos_token_id):
                break
        return generated


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    eos_token_id: int = ByteTokenizer.eos_token_id


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Sample one token id per batch item."""

    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / max(temperature, 1e-8)

    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        top_values, _ = torch.topk(logits, top_k, dim=-1)
        cutoff = top_values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True

        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sampled = torch.multinomial(filtered, num_samples=1)
        return sorted_indices.gather(-1, sampled).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_text(
    model: TinyCausalLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    *,
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> str:
    """Run the full pipeline from user input to response text."""

    generation_config = generation_config or GenerationConfig()
    model.eval()
    device = device or next(model.parameters()).device

    normalized = TextPreprocessor.normalize(prompt)
    prompt_ids = tokenizer.encode(normalized, add_bos=True)
    if len(prompt_ids) > model.config.max_seq_len:
        prompt_ids = prompt_ids[-model.config.max_seq_len :]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        top_p=generation_config.top_p,
        eos_token_id=generation_config.eos_token_id,
    )

    response_ids = generated[0].tolist()[len(prompt_ids) :]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def build_model(preset: str = "600m", *, device: Optional[torch.device] = None) -> TinyCausalLM:
    """Create a model from one of the built-in presets."""

    if preset not in MODEL_PRESETS:
        raise KeyError(f"Unknown preset: {preset}")

    model = TinyCausalLM(MODEL_PRESETS[preset])
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    tokenizer = ByteTokenizer()
    config = MODEL_PRESETS["600m"]
    model = TinyCausalLM(config)
    params = estimate_parameter_count(config)

    print(f"Preset: 600m")
    print(f"Estimated parameters: {params / 1e6:.1f}M")
    print("Architecture ready. Load trained weights before using generate_text().")
