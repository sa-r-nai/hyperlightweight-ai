"""Atomic, self-describing native model checkpoints without external model loaders."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

FORMAT = "native-korean-600m-checkpoint-v1"


def save_checkpoint(path: str | Path, model, **metadata) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "config": asdict(model.config), "model": model.state_dict(), **metadata}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path: str | Path) -> dict:
    import torch

    path = Path(path)
    if path.is_dir():
        from safetensors.torch import load_file
        from .config import sha256_file

        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "native-korean-600m-export-v1":
            raise ValueError("Unsupported native export format.")
        for name in ("model.safetensors", "model_config.json", "tokenizer.json"):
            if sha256_file(path / name) != manifest.get("files", {}).get(name):
                raise ValueError(f"Export file is missing or changed: {name}")
        config = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
        state = load_file(str(path / "model.safetensors"), device="cpu")
        if config.get("tie_embeddings"):
            state["lm_head.weight"] = state["token_embedding.weight"]
        payload = {"format": FORMAT, "config": config, "model": state,
                   "tokenizer_sha256": manifest["files"]["tokenizer.json"], "export_manifest": manifest,
                   "workflow_config": (manifest.get("training_run") or {}).get("workflow_config")}
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("This is not a NativeKoreanLM checkpoint.")
    if not isinstance(payload.get("config"), dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("The checkpoint is missing its model configuration or weights.")
    return payload


def load_model(path: str | Path, device: str = "cuda"):
    import torch
    from .model import ModelConfig, NativeLM

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select --device cpu explicitly for CPU inference.")
    payload = read_checkpoint(path)
    # Assignment from a CPU-mapped checkpoint avoids an extra full-size random model.
    with torch.device("meta"):
        model = NativeLM(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True, assign=True)
    # Restore sharing explicitly after assigning checkpoint tensors.
    if model.config.tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    model.to(device=device, dtype=dtype)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "model"}
    return model, metadata
