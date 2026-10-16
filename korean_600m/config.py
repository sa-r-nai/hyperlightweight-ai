"""Dependency-free configuration and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "korean_600m.json"


def load_config(path: str | Path | None = None) -> dict:
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if config.get("model_name") != "Qwen/Qwen3-0.6B":
        raise ValueError("This workflow supports Qwen/Qwen3-0.6B only.")
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("model_revision", "")):
        raise ValueError("model_revision must be a pinned 40-character commit SHA.")
    if not isinstance(config.get("system_prompt"), str) or not config["system_prompt"].strip():
        raise ValueError("system_prompt must be nonempty.")
    for key in ("max_seq_length", "seed"):
        if type(config.get(key)) is not int or config[key] < (32 if key == "max_seq_length" else 0):
            raise ValueError(f"Invalid {key}.")
    if config["max_seq_length"] > 32768:
        raise ValueError("max_seq_length must not exceed 32768 for this workflow.")
    for section in ("data", "lora", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing {section} configuration.")
    positive_keys = {
        "data": ("train_records", "validation_records", "max_characters"),
        "lora": ("r", "lora_alpha"),
        "training": ("num_train_epochs", "learning_rate", "per_device_train_batch_size",
                     "per_device_eval_batch_size", "gradient_accumulation_steps", "eval_steps",
                     "save_steps", "logging_steps", "save_total_limit"),
    }
    for section, keys in positive_keys.items():
        for key in keys:
            value = config[section].get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} must be positive and finite.")
    if config["training"]["save_steps"] != config["training"]["eval_steps"]:
        raise ValueError("save_steps and eval_steps must match for best-checkpoint selection.")
    for section, key, upper_exclusive in (("data", "min_hangul_ratio", False),
                                           ("lora", "lora_dropout", True),
                                           ("training", "warmup_ratio", True)):
        value = config[section].get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1 or (upper_exclusive and value == 1):
            raise ValueError(f"Invalid {section}.{key}.")
    workers = config["training"].get("dataloader_num_workers")
    if type(workers) is not int or workers < 0:
        raise ValueError("dataloader_num_workers must be a nonnegative integer.")
    if not config["lora"].get("target_modules"):
        raise ValueError("lora.target_modules must not be empty.")
    return config


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
