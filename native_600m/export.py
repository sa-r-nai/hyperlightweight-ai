"""Export native weights, an independent tokenizer, and provenance for inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .checkpoint import read_checkpoint
from .config import sha256_file, write_json


def export_model(checkpoint: Path, tokenizer_path: Path, output_dir: Path):
    import json

    identity = {"checkpoint_sha256": sha256_file(checkpoint), "tokenizer_sha256": sha256_file(tokenizer_path)}
    if output_dir.exists() and any(output_dir.iterdir()):
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Export directory is incomplete or unrelated. Choose a new output directory.")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source") != identity:
            raise ValueError("Existing export belongs to different native weights or tokenizer.")
        for name, digest in existing["files"].items():
            if name not in {"model.safetensors", "model_config.json", "tokenizer.json"}:
                raise ValueError("Invalid native export file manifest.")
            if sha256_file(output_dir / name) != digest:
                raise ValueError("Existing export is corrupted.")
        if set(existing["files"]) != {"model.safetensors", "model_config.json", "tokenizer.json"}:
            raise ValueError("Existing export is incomplete.")
        print(f"[export] Verified {output_dir}")
        return
    from safetensors.torch import save_file

    payload = read_checkpoint(checkpoint)
    if payload.get("tokenizer_sha256") != identity["tokenizer_sha256"]:
        raise ValueError("Checkpoint and tokenizer fingerprints differ.")
    state = dict(payload["model"])
    if payload["config"].get("tie_embeddings"):
        state.pop("lm_head.weight")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.safetensors"
    temporary = weights_path.with_suffix(".safetensors.tmp")
    save_file(state, str(temporary), metadata={"format": "native-korean-600m", "initialization": "random"})
    temporary.replace(weights_path)
    write_json(output_dir / "model_config.json", payload["config"])
    shutil.copyfile(tokenizer_path, output_dir / "tokenizer.json")
    run_path = checkpoint.parent / "run_manifest.json"
    source_manifest = payload.get("training_provenance")
    if source_manifest is None and run_path.is_file():
        source_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    write_json(output_dir / "manifest.json", {
        "format": "native-korean-600m-export-v1", "source": identity,
        "files": {name: sha256_file(output_dir / name) for name in
                  ("model.safetensors", "model_config.json", "tokenizer.json")},
        "architecture": payload["config"], "training_run": source_manifest,
        "initialization": "Random weights; no external pretrained model or vocabulary.",
        "conversation_quality_verified": False,
    })
    print(f"[export] Native model saved to {output_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts_native_600m/sft/best.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts_native_600m/tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_native_600m/model"))
    return parser.parse_args(argv)


def main():
    args = parse_args()
    export_model(args.checkpoint, args.tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
