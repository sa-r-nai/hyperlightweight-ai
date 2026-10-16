"""Merge a trained LoRA adapter into a new standalone Safetensors model directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .chat import DEFAULT_ADAPTER, PROVENANCE_FILE, load_chat_model, validate_adapter
from .config import DEFAULT_CONFIG, config_fingerprint, load_config, sha256_file, write_json


def validate_output_path(output: Path, adapter: Path, *, reuse_existing: bool = False) -> Path:
    resolved = output.resolve()
    adapter_resolved = adapter.resolve()
    if resolved == adapter_resolved or resolved in adapter_resolved.parents or adapter_resolved in resolved.parents:
        raise ValueError("Export output and adapter must be separate directories, with neither containing the other.")
    if resolved.exists() and (not resolved.is_dir() or (any(resolved.iterdir()) and not reuse_existing)):
        raise ValueError(f"Export output must be a new or empty directory: {resolved}")
    return resolved


def _artifact_path(output: Path, relative: str) -> Path:
    path = Path(relative)
    resolved = (output / path).resolve()
    if path.is_absolute() or ".." in path.parts or output not in resolved.parents:
        raise ValueError("Export file manifest contains a path outside the model directory.")
    return resolved


def _validate_artifacts(output: Path, hashes: dict[str, str]) -> None:
    """Check required loadable files and every saved artifact before reuse."""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Existing export has no file hash manifest; choose a new output directory.")
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if "model.safetensors" in hashes:
        required.add("model.safetensors")
    elif "model.safetensors.index.json" in hashes:
        required.add("model.safetensors.index.json")
        index_path = _artifact_path(output, "model.safetensors.index.json")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            raise ValueError("Existing export has an invalid Safetensors shard index.")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Existing export has an invalid Safetensors shard index.")
        shard_names = list(weight_map.values())
        if any(not isinstance(shard, str) or not shard.endswith(".safetensors") for shard in shard_names):
            raise ValueError("Existing export shard index references unsafe model files.")
        shards = set(shard_names)
        required.update(shards)
    else:
        raise ValueError("Existing export is missing Safetensors model weights.")
    if not required.issubset(hashes):
        raise ValueError("Existing export is missing required model or tokenizer files in its manifest.")
    for relative, expected_hash in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("Invalid export file manifest.")
        path = _artifact_path(output, relative)
        if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != expected_hash:
            raise ValueError(f"Existing export file is missing or corrupted: {relative}")


def _reuse_export(output: Path, expected: dict[str, Any]) -> dict[str, Any]:
    provenance_path = output / PROVENANCE_FILE
    if not provenance_path.is_file():
        raise ValueError("Existing export has no provenance; choose a new output directory.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("Existing export has malformed provenance.")
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Existing export does not match the current source: {key}")
    _validate_artifacts(output, provenance.get("artifact_sha256"))
    print(f"Verified existing merged model: {output}")
    return provenance


def export_merged(
    config: dict[str, Any], *, adapter: Path, output: Path, device: str, reuse_existing: bool = False
) -> dict[str, Any]:
    output = validate_output_path(output, adapter, reuse_existing=reuse_existing)
    adapter_metadata = validate_adapter(adapter, config)
    training_provenance_path = adapter / PROVENANCE_FILE
    if not training_provenance_path.is_file():
        raise FileNotFoundError(f"Training provenance is required to retain dataset and license metadata: {training_provenance_path}")
    training_provenance = json.loads(training_provenance_path.read_text(encoding="utf-8"))
    if (not isinstance(training_provenance, dict)
            or training_provenance.get("base_model") != config["model_name"]
            or training_provenance.get("base_revision") != config["model_revision"]):
        raise ValueError("Training provenance does not match the configured base model and revision.")
    identity = {
        "base_model": config["model_name"],
        "base_revision": config["model_revision"],
        "config_sha256": config_fingerprint(config),
        "source_adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "source_adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "source_training_provenance_sha256": sha256_file(training_provenance_path),
    }
    if reuse_existing and output.is_dir() and any(output.iterdir()):
        return _reuse_export(output, identity)
    model, tokenizer = load_chat_model(config, adapter=adapter, merged=None, device=device)
    # safe_merge rejects NaN values; merging does not increase the base architecture size.
    merged_model = model.merge_and_unload(safe_merge=True)
    merged_model.eval()
    unique_parameters = sum(parameter.numel() for parameter in merged_model.parameters())
    output.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output)
    artifact_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    _validate_artifacts(output, artifact_hashes)
    provenance = {
        "provenance_version": "korean-600m-merged-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **identity,
        "source_adapter": str(adapter.resolve()),
        "adapter_config": adapter_metadata,
        "training_provenance": training_provenance,
        "artifact_sha256": artifact_hashes,
        "unique_parameters": unique_parameters,
        "format": "safetensors",
        "merge": "PEFT merge_and_unload(safe_merge=True)",
        "system_prompt": config["system_prompt"],
        "max_seq_length": config["max_seq_length"],
        "conversation_quality_verified": False,
        "training_config": config,
    }
    write_json(output / PROVENANCE_FILE, provenance)
    print(f"Merged model saved to {output} ({unique_parameters:,} unique parameters).")
    return provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained Korean 600M LoRA as a standalone model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=Path("artifacts_korean_600m/merged"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse only an intact export with matching source hashes and configuration.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    export_merged(load_config(args.config), adapter=args.adapter, output=args.output, device=args.device,
                  reuse_existing=args.reuse_existing)


if __name__ == "__main__":
    main()
