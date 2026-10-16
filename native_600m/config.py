"""Dependency-free configuration and artifact integrity helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "native_600m.json"


def config_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: str | Path | None = None) -> dict:
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer.")
    if not isinstance(config.get("system_prompt"), str) or not config["system_prompt"].strip():
        raise ValueError("A nonempty Korean system_prompt is required.")
    for section in ("model", "data", "tokenizer", "pretrain", "sft"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing {section} configuration.")
    integer_keys = {
        "model": ("vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff"),
        "data": ("pretrain_train_documents", "pretrain_validation_documents", "chat_train_records",
                 "chat_validation_records", "min_text_characters", "max_text_characters"),
        "tokenizer": ("vocab_size", "max_documents"),
        "pretrain": ("target_tokens", "seq_len", "batch_size", "grad_accumulation", "eval_every",
                     "checkpoint_every", "max_eval_batches"),
        "sft": ("seq_len", "batch_size", "grad_accumulation", "eval_every", "checkpoint_every", "max_eval_batches"),
    }
    for section, keys in integer_keys.items():
        for key in keys:
            if type(config[section].get(key)) is not int or config[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer.")
    if not 263 <= config["model"]["vocab_size"] <= 65535:
        raise ValueError("The vocabulary must include all byte tokens and fit uint16.")
    if config["model"]["vocab_size"] != config["tokenizer"]["vocab_size"]:
        raise ValueError("Model and tokenizer vocabulary targets must match.")
    if config["data"]["min_text_characters"] > config["data"]["max_text_characters"]:
        raise ValueError("Minimum text length exceeds maximum text length.")
    ratio = config["data"].get("min_hangul_ratio")
    if not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
        raise ValueError("Invalid Korean character ratio.")
    model = config["model"]
    if model["d_model"] % model["n_heads"] or (model["d_model"] // model["n_heads"]) % 2:
        raise ValueError("Attention heads must have an even integer dimension.")
    if model["n_heads"] % model["n_kv_heads"]:
        raise ValueError("Query heads must be divisible by KV heads.")
    for stage in ("pretrain", "sft"):
        options = config[stage]
        if options["seq_len"] > model["max_seq_len"]:
            raise ValueError(f"{stage} sequence exceeds model context.")
        for key in ("lr", "min_lr"):
            value = options.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {stage}.{key}.")
        if options["min_lr"] > options["lr"]:
            raise ValueError("Minimum learning rate must not exceed peak learning rate.")
        if type(options.get("warmup_steps")) is not int or options["warmup_steps"] < 0:
            raise ValueError("warmup_steps must be nonnegative.")
    epochs = config["sft"].get("epochs")
    if not isinstance(epochs, (int, float)) or not math.isfinite(epochs) or epochs <= 0:
        raise ValueError("sft.epochs must be positive and finite.")
    return config
